#include "../vector_bridge.hpp"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include <stdexcept>

std::string VectorBridge::newVar() {
    return "v" + std::to_string(var_counter_++);
}

std::string VectorBridge::getVar(mlir::Value v) {
    auto it = value_map_.find(v.getAsOpaquePointer());
    if (it == value_map_.end()) return "/*undef*/";
    return it->second;
}

void VectorBridge::setVar(mlir::Value v, const std::string& name) {
    value_map_[v.getAsOpaquePointer()] = name;
}

std::string VectorBridge::getDim(mlir::Value memref, int idx) {
    auto it = dim_map_.find(memref.getAsOpaquePointer());
    if (it == dim_map_.end()) return "";
    auto it2 = it->second.find(idx);
    return it2 == it->second.end() ? "" : it2->second;
}

void VectorBridge::setDim(mlir::Value memref, int idx, const std::string& name) {
    dim_map_[memref.getAsOpaquePointer()][idx] = name;
}

std::string VectorBridge::getDimExpr(mlir::Value memref, int idx) {
    auto memType = mlir::cast<mlir::MemRefType>(memref.getType());
    int64_t shape = memType.getShape()[idx];
    if (shape != mlir::ShapedType::kDynamic) {
        return std::to_string(shape); // static: safe to inline as a literal
    }
    std::string v = getDim(memref, idx);
    if (!v.empty()) return v;
    // Dynamic dim we never saw a memref.dim for — don't silently emit 0.
    return "/*UNRESOLVED_DYNAMIC_DIM*/1";
}

void VectorBridge::collectDynamicDims(mlir::Operation* root) {
    auto func = mlir::cast<mlir::func::FuncOp>(root);
    for (auto arg : func.getArguments()) {
        auto memType = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
        if (!memType) continue;
        auto shape = memType.getShape();
        for (size_t idx = 0; idx < shape.size(); ++idx) {
            if (shape[idx] != mlir::ShapedType::kDynamic) continue;
            if (!getDim(arg, idx).empty()) continue; // 已登記過就跳過
            std::string name = "dim_p" + std::to_string(arg.getArgNumber())
                              + "_" + std::to_string(idx);
            setDim(arg, idx, name);
        }
    }
}

std::string VectorBridge::computeFlatOffset(mlir::Value memref, mlir::Operation::operand_range indices) {
    auto memType = mlir::cast<mlir::MemRefType>(memref.getType());
    int rank = memType.getRank();
    std::string offset;
    for (int d = 0; d < rank; ++d) {
        std::string idx = getVar(indices[d]);
        std::string stride = getPhysicalStride(memref, d);
        std::string term = (stride == "1") ? idx : ("(" + idx + "*" + stride + ")");
        if (offset.empty()) offset = term;
        else offset += " + " + term;
    }
    return offset.empty() ? "0" : offset;
}

std::string VectorBridge::getPhysicalStride(mlir::Value memref, int dim) {
    auto memType = mlir::cast<mlir::MemRefType>(memref.getType());
    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (mlir::succeeded(mlir::getStridesAndOffset(memType, strides, offset))) {
        if (strides[dim] != mlir::ShapedType::kDynamic) {
            return std::to_string(strides[dim]);
        }
    }
    // Dynamic stride, or layout not statically representable — fall back to
    // the row-major assumption (product of trailing dim sizes), which is
    // correct for any memref without an explicit non-default layout.
    auto shape = memType.getShape();
    std::string prod;
    for (int t = dim + 1; t < memType.getRank(); ++t) {
        std::string d = getDimExpr(memref, t);
        prod = prod.empty() ? d : (prod + "*" + d);
    }
    return prod.empty() ? "1" : prod;
}

std::string VectorBridge::computeVL(int vlen) {
    // 由內而外找第一個 step 等於 vlen 的迴圈，代表這是向量維度的那一層
    for (auto it = loop_stack_.rbegin(); it != loop_stack_.rend(); ++it) {
        if (it->step == vlen) {
            return "(" + it->iv + " + " + std::to_string(vlen) + " <= " + it->hi +
                   " ? " + std::to_string(vlen) + " : (" + it->hi + " - " + it->iv + "))";
        }
    }
    // 找不到對應迴圈（理論上不該發生），退回寫死的 vlen，至少行為跟原本一樣
    return std::to_string(vlen);
}

std::string VectorBridge::computeSafeVL(mlir::Value memref, mlir::Value lastIndexValue, int vlen) {
    std::string loopVL = computeVL(vlen);
    auto memType = mlir::cast<mlir::MemRefType>(memref.getType());
    int rank = memType.getRank();
    std::string dimSize = getDimExpr(memref, rank - 1); // 假設向量化的維度是最內層
    std::string idxVar = getVar(lastIndexValue);
    std::string remaining = "(" + dimSize + " - " + idxVar + ")";
    return "(" + loopVL + " < " + remaining + " ? " + loopVL + " : " + remaining + ")";
}

VecTypeInfo VectorBridge::getVecTypeInfo(mlir::Type elemType, int vlen) {
    bool isF64 = elemType.isF64();
    int elemBits = isF64 ? 64 : 32;
    int totalBits = elemBits * vlen;
    int lmul = (totalBits + 127) / 128; // 假設 VLEN=128，無條件進位算需要幾組暫存器
    if (lmul < 1) lmul = 1;
    // 只支援 m1/m2/m4/m8（RVV 合法的 LMUL 值），這裡簡單處理到 m8
    std::string lmulTag = "m" + std::to_string(lmul);
    std::string elemTag = isF64 ? "f64" : "f32";
    std::string bitwidth = isF64 ? "64" : "32";
    std::string scalarCType = isF64 ? "double" : "float";
    std::string fullElemName = isF64 ? "float64" : "float32";
    std::string vecCType = "v" + fullElemName + lmulTag + "_t";
    std::string suffix = elemTag + lmulTag;
    return {vecCType, bitwidth, suffix, scalarCType};
}

std::string VectorBridge::getScalarCType(mlir::Type memrefElemType) {
    return memrefElemType.isF64() ? "double" : "float";
}

void VectorBridge::emitFunc(mlir::func::FuncOp func) {
    fprintf(out_, "#include <riscv_vector.h>\n\n");

    collectDynamicDims(func.getOperation()); // 先掃一遍，決定要加哪些 size_t 參數

    fprintf(out_, "void %s(", func.getName().str().c_str());
    bool first = true;
    for (auto arg : func.getArguments()) {
        if (!first) fprintf(out_, ", ");
        first = false;
        auto name = "p" + std::to_string(arg.getArgNumber());
        setVar(arg, name);
        auto memType = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
        std::string cType = memType ? getScalarCType(memType.getElementType()) : "float";
        fprintf(out_, "%s* %s", cType.c_str(), name.c_str());
    }
    for (auto arg : func.getArguments()) {
        auto it = dim_map_.find(arg.getAsOpaquePointer());
        if (it == dim_map_.end()) continue;
        for (auto& [idx, name] : it->second) {
            fprintf(out_, ", size_t %s", name.c_str());
        }
    }
    fprintf(out_, ") {\n");
    for (auto& block : func.getBody()) {
        for (auto& op : block) emitOp(&op);
    }
    fprintf(out_, "}\n");
}

void VectorBridge::emitOp(mlir::Operation* op) {
    if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
        emitAffineFor(op);
    } else if (mlir::isa<mlir::affine::AffineApplyOp>(op)) {
        emitAffineApply(op);
    } else if (mlir::isa<mlir::vector::TransferReadOp>(op)) {
        emitTransferRead(op);
    } else if (mlir::isa<mlir::vector::TransferWriteOp>(op)) {
        emitTransferWrite(op);
    } else if (mlir::isa<mlir::vector::ReductionOp>(op)) {
        emitVectorReduction(op);
    } else if (mlir::isa<mlir::vector::BroadcastOp>(op)) {
        emitVectorBroadcast(op);
    } else if (auto mulf = mlir::dyn_cast<mlir::arith::MulFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(mulf.getType()))
            emitVectorMulf(op);
    } else if (auto addf = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(addf.getType()))
            emitVectorAddf(op);
    } else if (auto subf = mlir::dyn_cast<mlir::arith::SubFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(subf.getType()))
            emitVectorSubf(op);
    } else if (auto divf = mlir::dyn_cast<mlir::arith::DivFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(divf.getType()))
            emitVectorDivf(op);
    } else if (auto maxf = mlir::dyn_cast<mlir::arith::MaximumFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(maxf.getType()))
            emitVectorMaxf(op);
    } else if (auto minf = mlir::dyn_cast<mlir::arith::MinimumFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(minf.getType()))
            emitVectorMinf(op);
    } else if (mlir::isa<mlir::arith::SubIOp>(op)) {
        emitSubI(op);
    } else if (mlir::isa<mlir::arith::AddIOp>(op)) {
        emitAddI(op);
    } else if (mlir::isa<mlir::arith::ConstantOp>(op)) {
        emitConstant(op);
    } else if (mlir::isa<mlir::memref::DimOp>(op)) {
        emitMemrefDim(op);
    } else if (mlir::isa<mlir::func::ReturnOp>(op)) {
        fprintf(out_, "  return;\n");
    }
}

void VectorBridge::emitAffineFor(mlir::Operation* op) {
    auto forOp = mlir::cast<mlir::affine::AffineForOp>(op);
    std::string iv = newVar();
    setVar(forOp.getInductionVar(), iv);

    std::string lo, hi;
    if (forOp.hasConstantLowerBound()) {
        lo = std::to_string(forOp.getConstantLowerBound());
    } else {
        auto ops = forOp.getLowerBoundOperands();
        lo = !ops.empty() ? getVar(ops[0]) : "/*UNSUPPORTED_LB*/0";
    }
    if (forOp.hasConstantUpperBound()) {
        hi = std::to_string(forOp.getConstantUpperBound());
    } else {
        auto ops = forOp.getUpperBoundOperands();
        hi = !ops.empty() ? getVar(ops[0]) : "/*UNSUPPORTED_UB*/0";
    }

    long step = forOp.getStepAsInt();
    fprintf(out_, "  for (int %s = %s; %s < %s; %s += %ld) {\n",
        iv.c_str(), lo.c_str(), iv.c_str(), hi.c_str(), iv.c_str(), step);

    loop_stack_.push_back({iv, hi, step});
    for (auto& nested : forOp.getBody()->without_terminator()) emitOp(&nested);
    loop_stack_.pop_back();

    fprintf(out_, "  }\n");
}

std::string VectorBridge::affineExprToStr(mlir::AffineExpr expr,
                                           const std::vector<std::string>& dimVars,
                                           const std::vector<std::string>& symVars) {
    if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
        return dimVars[dim.getPosition()];
    }
    if (auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
        return symVars[sym.getPosition()];
    }
    if (auto cst = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
        return std::to_string(cst.getValue());
    }
    if (auto bin = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
        std::string lhs = affineExprToStr(bin.getLHS(), dimVars, symVars);
        std::string rhs = affineExprToStr(bin.getRHS(), dimVars, symVars);
        switch (bin.getKind()) {
            case mlir::AffineExprKind::Add:
                return "(" + lhs + " + " + rhs + ")";
            case mlir::AffineExprKind::Mul:
                return "(" + lhs + " * " + rhs + ")";
            case mlir::AffineExprKind::Mod:
                return "(" + lhs + " % " + rhs + ")";
            case mlir::AffineExprKind::FloorDiv:
                return "(" + lhs + " / " + rhs + ")"; // 注意：對負數語意跟 C 的 / 不同，先不處理
            case mlir::AffineExprKind::CeilDiv:
                return "((" + lhs + " + " + rhs + " - 1) / " + rhs + ")";
            default:
                return "/*UNSUPPORTED_AFFINE_BINOP*/0";
        }
    }
    return "/*UNSUPPORTED_AFFINE_EXPR*/0";
}

void VectorBridge::emitAffineApply(mlir::Operation* op) {
    auto applyOp = mlir::cast<mlir::affine::AffineApplyOp>(op);
    auto map = applyOp.getAffineMap();
    auto operands = applyOp.getMapOperands();

    std::vector<std::string> dimVars, symVars;
    unsigned numDims = map.getNumDims();
    for (unsigned i = 0; i < numDims; ++i)
        dimVars.push_back(getVar(operands[i]));
    for (unsigned i = numDims; i < operands.size(); ++i)
        symVars.push_back(getVar(operands[i]));

    std::string expr = affineExprToStr(map.getResult(0), dimVars, symVars);
    std::string var = newVar();
    setVar(applyOp.getResult(), var);
    fprintf(out_, "  int %s = %s;\n", var.c_str(), expr.c_str());
}

void VectorBridge::emitTransferRead(mlir::Operation* op) {
    auto readOp = mlir::cast<mlir::vector::TransferReadOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(readOp.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);
    std::string var = newVar();
    setVar(readOp.getResult(), var);
    std::string base = getVar(readOp.getSource());
    auto indices = readOp.getIndices();
    std::string offset = computeFlatOffset(readOp.getSource(), indices);

    auto permMap = readOp.getPermutationMap();
    bool isBroadcast = false;
    if (permMap.getNumResults() == 1) {
        if (auto cstExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(permMap.getResult(0))) {
            if (cstExpr.getValue() == 0) isBroadcast = true;
        }
    }

    if (isBroadcast) {
        fprintf(out_, "  %s %s_scalar = %s[%s];\n",
            tinfo.scalarCType.c_str(), var.c_str(), base.c_str(), offset.c_str());
        fprintf(out_, "  %s %s = __riscv_vfmv_v_f_%s(%s_scalar, %s);\n",
            tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
            var.c_str(), computeVL(vlen).c_str());
        return;
    }

    auto srcMemType = mlir::cast<mlir::MemRefType>(readOp.getSource().getType());
    std::string innerStride = getPhysicalStride(readOp.getSource(), srcMemType.getRank() - 1);
    std::string vl = computeSafeVL(readOp.getSource(), indices.back(), vlen);
    if (innerStride == "1") {
        fprintf(out_, "  %s %s = __riscv_vle%s_v_%s(%s + %s, %s);\n",
            tinfo.vecCType.c_str(), var.c_str(), tinfo.bitwidth.c_str(), tinfo.suffix.c_str(),
            base.c_str(), offset.c_str(), vl.c_str());
    } else {
        fprintf(out_, "  %s %s = __riscv_vlse%s_v_%s(%s + %s, %s * sizeof(%s), %s);\n",
            tinfo.vecCType.c_str(), var.c_str(), tinfo.bitwidth.c_str(), tinfo.suffix.c_str(),
            base.c_str(), offset.c_str(), innerStride.c_str(), tinfo.scalarCType.c_str(), vl.c_str());
    }
}

void VectorBridge::emitTransferWrite(mlir::Operation* op) {
    auto writeOp = mlir::cast<mlir::vector::TransferWriteOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(writeOp.getVector().getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);
    std::string vec = getVar(writeOp.getVector());
    std::string base = getVar(writeOp.getSource());
    auto indices = writeOp.getIndices();
    std::string offset = computeFlatOffset(writeOp.getSource(), indices);
    std::string vl = computeSafeVL(writeOp.getSource(), indices.back(), vlen);

    auto dstMemType = mlir::cast<mlir::MemRefType>(writeOp.getSource().getType());
    std::string innerStride = getPhysicalStride(writeOp.getSource(), dstMemType.getRank() - 1);
    if (innerStride == "1") {
        fprintf(out_, "  __riscv_vse%s_v_%s(%s + %s, %s, %s);\n",
            tinfo.bitwidth.c_str(), tinfo.suffix.c_str(),
            base.c_str(), offset.c_str(), vec.c_str(), vl.c_str());
    } else {
        fprintf(out_, "  __riscv_vsse%s_v_%s(%s + %s, %s * sizeof(%s), %s, %s);\n",
            tinfo.bitwidth.c_str(), tinfo.suffix.c_str(),
            base.c_str(), offset.c_str(), innerStride.c_str(), tinfo.scalarCType.c_str(),
            vec.c_str(), vl.c_str());
    }
}

void VectorBridge::emitVectorMulf(mlir::Operation* op) {
    auto mulf = mlir::cast<mlir::arith::MulFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(mulf.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);    
    std::string var = newVar();
    setVar(mulf.getResult(), var);
    fprintf(out_, "  %s %s = __riscv_vfmul_vv_%s(%s, %s, %s);\n",
        tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
        getVar(mulf.getLhs()).c_str(), getVar(mulf.getRhs()).c_str(), computeVL(vlen).c_str());
}

void VectorBridge::emitVectorAddf(mlir::Operation* op) {
    auto addf = mlir::cast<mlir::arith::AddFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(addf.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);    
    std::string var = newVar();
    setVar(addf.getResult(), var);
    fprintf(out_, "  %s %s = __riscv_vfadd_vv_%s(%s, %s, %s);\n",
        tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
        getVar(addf.getLhs()).c_str(), getVar(addf.getRhs()).c_str(), computeVL(vlen).c_str());
}

void VectorBridge::emitVectorSubf(mlir::Operation* op) {
    auto subf = mlir::cast<mlir::arith::SubFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(subf.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);    
    std::string var = newVar();
    setVar(subf.getResult(), var);
    fprintf(out_, "  %s %s = __riscv_vfsub_vv_%s(%s, %s, %s);\n",
        tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
        getVar(subf.getLhs()).c_str(), getVar(subf.getRhs()).c_str(),
        computeVL(vlen).c_str());
}

void VectorBridge::emitVectorDivf(mlir::Operation* op) {
    auto divf = mlir::cast<mlir::arith::DivFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(divf.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);    
    std::string var = newVar();
    setVar(divf.getResult(), var);
    fprintf(out_, "  %s %s = __riscv_vfdiv_vv_%s(%s, %s, %s);\n",
        tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
        getVar(divf.getLhs()).c_str(), getVar(divf.getRhs()).c_str(),
        computeVL(vlen).c_str());
}

void VectorBridge::emitVectorMaxf(mlir::Operation* op) {
    auto maxf = mlir::cast<mlir::arith::MaximumFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(maxf.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);
    std::string var = newVar();
    setVar(maxf.getResult(), var);
    fprintf(out_, "  %s %s = __riscv_vfmax_vv_%s(%s, %s, %s);\n",
        tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
        getVar(maxf.getLhs()).c_str(), getVar(maxf.getRhs()).c_str(), computeVL(vlen).c_str());
}

void VectorBridge::emitVectorMinf(mlir::Operation* op) {
    auto minf = mlir::cast<mlir::arith::MinimumFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(minf.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);
    std::string var = newVar();
    setVar(minf.getResult(), var);
    fprintf(out_, "  %s %s = __riscv_vfmin_vv_%s(%s, %s, %s);\n",
        tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
        getVar(minf.getLhs()).c_str(), getVar(minf.getRhs()).c_str(), computeVL(vlen).c_str());
}

void VectorBridge::emitVectorReduction(mlir::Operation* op) {
    auto redOp = mlir::cast<mlir::vector::ReductionOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(redOp.getVector().getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);
    std::string srcVec = getVar(redOp.getVector());
    std::string vl = computeVL(vlen);

    if (redOp.getKind() == mlir::vector::CombiningKind::MUL) {
        // RVV has no native product-reduction instruction.
        // Fallback: spill the vector to a scratch C array and multiply
        // scalars in a plain loop. Slower than a hardware reduction,
        // but correct — preferable to silently emitting nothing or
        // guessing at an unsupported intrinsic.
        std::string scratch = newVar();
        fprintf(out_, "  %s %s_scratch[%d];\n", tinfo.scalarCType.c_str(), scratch.c_str(), vlen);
        fprintf(out_, "  __riscv_vse%s_v_%s(%s_scratch, %s, %s);\n",
            tinfo.bitwidth.c_str(), tinfo.suffix.c_str(), scratch.c_str(), srcVec.c_str(), vl.c_str());
        std::string var = newVar();
        setVar(redOp.getResult(), var);
        fprintf(out_, "  %s %s = 1;\n", tinfo.scalarCType.c_str(), var.c_str());
        fprintf(out_, "  for (int _i = 0; _i < %s; _i++) %s *= %s_scratch[_i];\n",
            vl.c_str(), var.c_str(), scratch.c_str());
        return;
    }

    std::string kind;
    switch (redOp.getKind()) {
        case mlir::vector::CombiningKind::ADD: kind = "osum"; break;
        case mlir::vector::CombiningKind::MAXNUMF: kind = "max"; break;
        case mlir::vector::CombiningKind::MINNUMF: kind = "min"; break;
        default:
            fprintf(out_, "  /*UNSUPPORTED_REDUCTION_KIND*/\n");
            std::string var = newVar();
            setVar(redOp.getResult(), var);
            fprintf(out_, "  %s %s = 0;\n", tinfo.scalarCType.c_str(), var.c_str());
            return;
    }

    // (原本 ADD/MAX/MIN 那段完全不動，維持現狀)
    std::string seedVar = newVar();
    if (kind != "osum") {
        fprintf(out_, "  %s %s_scalar0;\n", tinfo.scalarCType.c_str(), seedVar.c_str());
        fprintf(out_, "  __riscv_vse%s_v_%s(&%s_scalar0, %s, 1);\n",
            tinfo.bitwidth.c_str(), tinfo.suffix.c_str(), seedVar.c_str(), srcVec.c_str());
        fprintf(out_, "  %s %s = __riscv_vfmv_v_f_%s(%s_scalar0, %s);\n",
            tinfo.vecCType.c_str(), seedVar.c_str(), tinfo.suffix.c_str(), seedVar.c_str(), vl.c_str());
    } else {
        fprintf(out_, "  %s %s = __riscv_vfmv_v_f_%s(0.0f, %s);\n",
            tinfo.vecCType.c_str(), seedVar.c_str(), tinfo.suffix.c_str(), vl.c_str());
    }

    std::string resultVec = newVar();
    fprintf(out_, "  %s %s = __riscv_vfred%s_vs_%s_%s(%s, %s, %s);\n",
        tinfo.vecCType.c_str(), resultVec.c_str(),
        kind.c_str(), tinfo.suffix.c_str(), tinfo.suffix.c_str(),
        srcVec.c_str(), seedVar.c_str(), vl.c_str());

    std::string var = newVar();
    setVar(redOp.getResult(), var);
    std::string scalarTag = tinfo.scalarCType == "double" ? "f64" : "f32";
    fprintf(out_, "  %s %s = __riscv_vfmv_f_s_%s_%s(%s);\n",
        tinfo.scalarCType.c_str(), var.c_str(),
        tinfo.suffix.c_str(), scalarTag.c_str(), resultVec.c_str());
}

void VectorBridge::emitConstant(mlir::Operation* op) {
    auto cst = mlir::cast<mlir::arith::ConstantOp>(op);
    if (auto idxAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
        std::string var = newVar();
        setVar(cst.getResult(), var);
        fprintf(out_, "  int %s = %ld;\n", var.c_str(), idxAttr.getInt());
    } else if (auto fAttr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
        std::string var = newVar();
        setVar(cst.getResult(), var);
        std::string cType = fAttr.getType().isF64() ? "double" : "float";
        fprintf(out_, "  %s %s = %g;\n", cType.c_str(), var.c_str(), fAttr.getValueAsDouble());
    }
}

void VectorBridge::emitSubI(mlir::Operation* op) {
    auto subi = mlir::cast<mlir::arith::SubIOp>(op);
    std::string var = newVar();
    setVar(subi.getResult(), var);
    fprintf(out_, "  int %s = %s - %s;\n", var.c_str(),
        getVar(subi.getLhs()).c_str(), getVar(subi.getRhs()).c_str());
}

void VectorBridge::emitAddI(mlir::Operation* op) {
    auto addi = mlir::cast<mlir::arith::AddIOp>(op);
    std::string var = newVar();
    setVar(addi.getResult(), var);
    fprintf(out_, "  int %s = %s + %s;\n", var.c_str(),
        getVar(addi.getLhs()).c_str(), getVar(addi.getRhs()).c_str());
}

void VectorBridge::emitVectorBroadcast(mlir::Operation* op) {
    auto bc = mlir::cast<mlir::vector::BroadcastOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(bc.getType());
    int vlen = vecType.getShape()[0];
    auto tinfo = getVecTypeInfo(vecType.getElementType(), vlen);    
    std::string var = newVar();
    setVar(bc.getResult(), var);
    fprintf(out_, "  %s %s = __riscv_vfmv_v_f_%s(%s, %s);\n",
        tinfo.vecCType.c_str(), var.c_str(), tinfo.suffix.c_str(),
        getVar(bc.getSource()).c_str(), computeVL(vlen).c_str());
}

void VectorBridge::emitMemrefDim(mlir::Operation* op) {
    auto dimOp = mlir::cast<mlir::memref::DimOp>(op);
    mlir::Value src = dimOp.getSource();
    auto idxOp = dimOp.getIndex().getDefiningOp<mlir::arith::ConstantOp>();
    if (!idxOp) return;
    int idx = mlir::cast<mlir::IntegerAttr>(idxOp.getValue()).getInt();

    std::string dimVar = getDim(src, idx);
    if (!dimVar.empty()) {
        setVar(dimOp.getResult(), dimVar); // 已經是函式參數了，SSA result 直接指過去
        return;
    }
    auto memType = mlir::cast<mlir::MemRefType>(src.getType());
    std::string var = newVar();
    setVar(dimOp.getResult(), var);
    fprintf(out_, "  size_t %s = %ld;\n", var.c_str(), memType.getShape()[idx]);
}
