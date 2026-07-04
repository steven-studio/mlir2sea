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
        std::string term = getVar(indices[d]);
        // multiply by product of all trailing dims (d+1 .. rank-1)
        for (int t = d + 1; t < rank; ++t) {
            term += "*" + getDimExpr(memref, t);
        }
        if (offset.empty()) offset = term;
        else offset += " + " + term;
    }
    return offset.empty() ? "0" : offset;
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
        fprintf(out_, "float* %s", name.c_str());
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
    } else if (mlir::isa<mlir::vector::TransferReadOp>(op)) {
        emitTransferRead(op);
    } else if (mlir::isa<mlir::vector::TransferWriteOp>(op)) {
        emitTransferWrite(op);
    } else if (mlir::isa<mlir::vector::BroadcastOp>(op)) {
        emitVectorBroadcast(op);
    } else if (auto mulf = mlir::dyn_cast<mlir::arith::MulFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(mulf.getType()))
            emitVectorMulf(op);
    } else if (auto addf = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(addf.getType()))
            emitVectorAddf(op);
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

    fprintf(out_, "  for (int %s = %s; %s < %s; %s += %ld) {\n",
        iv.c_str(), lo.c_str(), iv.c_str(), hi.c_str(), iv.c_str(), forOp.getStepAsInt());
    for (auto& nested : forOp.getBody()->without_terminator()) emitOp(&nested);
    fprintf(out_, "  }\n");
}

void VectorBridge::emitTransferRead(mlir::Operation* op) {
    auto readOp = mlir::cast<mlir::vector::TransferReadOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(readOp.getType());
    int vlen = vecType.getShape()[0];
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
        fprintf(out_, "  float %s_scalar = %s[%s];\n",
            var.c_str(), base.c_str(), offset.c_str());
        fprintf(out_, "  vfloat32m1_t %s = __riscv_vfmv_v_f_f32m1(%s_scalar, %d);\n",
            var.c_str(), var.c_str(), vlen);
    } else {
        fprintf(out_, "  vfloat32m1_t %s = __riscv_vle32_v_f32m1(%s + %s, %d);\n",
            var.c_str(), base.c_str(), offset.c_str(), vlen);
    }
}

void VectorBridge::emitTransferWrite(mlir::Operation* op) {
    auto writeOp = mlir::cast<mlir::vector::TransferWriteOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(writeOp.getVector().getType());
    int vlen = vecType.getShape()[0];
    std::string vec = getVar(writeOp.getVector());
    std::string base = getVar(writeOp.getSource());
    auto indices = writeOp.getIndices();
    std::string offset = computeFlatOffset(writeOp.getSource(), indices);
    fprintf(out_, "  __riscv_vse32_v_f32m1(%s + %s, %s, %d);\n",
        base.c_str(), offset.c_str(), vec.c_str(), vlen);
}

void VectorBridge::emitVectorMulf(mlir::Operation* op) {
    auto mulf = mlir::cast<mlir::arith::MulFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(mulf.getType());
    int vlen = vecType.getShape()[0];
    std::string var = newVar();
    setVar(mulf.getResult(), var);
    fprintf(out_, "  vfloat32m1_t %s = __riscv_vfmul_vv_f32m1(%s, %s, %d);\n",
        var.c_str(), getVar(mulf.getLhs()).c_str(), getVar(mulf.getRhs()).c_str(), vlen);
}

void VectorBridge::emitVectorAddf(mlir::Operation* op) {
    auto addf = mlir::cast<mlir::arith::AddFOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(addf.getType());
    int vlen = vecType.getShape()[0];
    std::string var = newVar();
    setVar(addf.getResult(), var);
    fprintf(out_, "  vfloat32m1_t %s = __riscv_vfadd_vv_f32m1(%s, %s, %d);\n",
        var.c_str(), getVar(addf.getLhs()).c_str(), getVar(addf.getRhs()).c_str(), vlen);
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
        fprintf(out_, "  float %s = %g;\n", var.c_str(), fAttr.getValueAsDouble());
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
    std::string var = newVar();
    setVar(bc.getResult(), var);
    fprintf(out_, "  vfloat32m1_t %s = __riscv_vfmv_v_f_f32m1(%s, %d);\n",
        var.c_str(), getVar(bc.getSource()).c_str(), vlen);
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
