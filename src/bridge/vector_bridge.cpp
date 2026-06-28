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

void VectorBridge::emitFunc(mlir::func::FuncOp func) {
    // emit function signature
    fprintf(out_, "#include <riscv_vector.h>\n\n");
    fprintf(out_, "void %s(", func.getName().str().c_str());
    bool first = true;
    for (auto arg : func.getArguments()) {
        if (!first) fprintf(out_, ", ");
        first = false;
        auto name = "p" + std::to_string(arg.getArgNumber());
        setVar(arg, name);
        fprintf(out_, "float* %s", name.c_str());
    }
    fprintf(out_, ") {\n");
    for (auto& block : func.getBody()) {
        for (auto& op : block) {
            emitOp(&op);
        }
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
    } else if (auto mulf = mlir::dyn_cast<mlir::arith::MulFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(mulf.getType()))
            emitVectorMulf(op);
    } else if (auto addf = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
        if (mlir::isa<mlir::VectorType>(addf.getType()))
            emitVectorAddf(op);
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
    fprintf(out_, "  for (int %s = %ld; %s < %ld; %s += %ld) {\n",
        iv.c_str(), forOp.getConstantLowerBound(),
        iv.c_str(), forOp.getConstantUpperBound(),
        iv.c_str(), forOp.getStepAsInt());
    for (auto& nested : forOp.getBody()->without_terminator()) {
        emitOp(&nested);
    }
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
    auto memType = mlir::cast<mlir::MemRefType>(readOp.getSource().getType());
    int cols = memType.getShape()[1];
    std::string i0 = getVar(indices[0]);
    std::string i1 = getVar(indices[1]);

    // Check permutation_map: if it maps to constant 0, it's a broadcast (scalar splat)
    auto permMap = readOp.getPermutationMap();
    bool isBroadcast = false;
    if (permMap.getNumResults() == 1) {
        if (auto cstExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(permMap.getResult(0))) {
            if (cstExpr.getValue() == 0) isBroadcast = true;
        }
    }

    if (isBroadcast) {
        // scalar broadcast: load one element and splat
        fprintf(out_, "  float %s_scalar = %s[%s*%d + %s];\n",
            var.c_str(), base.c_str(), i0.c_str(), cols, i1.c_str());
        fprintf(out_, "  vfloat32m1_t %s = __riscv_vfmv_v_f_f32m1(%s_scalar, %d);\n",
            var.c_str(), var.c_str(), vlen);
    } else {
        fprintf(out_, "  vfloat32m1_t %s = __riscv_vle32_v_f32m1(%s + %s*%d + %s, %d);\n",
            var.c_str(), base.c_str(), i0.c_str(), cols, i1.c_str(), vlen);
    }
}

void VectorBridge::emitTransferWrite(mlir::Operation* op) {
    auto writeOp = mlir::cast<mlir::vector::TransferWriteOp>(op);
    auto vecType = mlir::cast<mlir::VectorType>(writeOp.getVector().getType());
    int vlen = vecType.getShape()[0];
    std::string vec = getVar(writeOp.getVector());
    std::string base = getVar(writeOp.getSource());
    auto indices = writeOp.getIndices();
    auto memType = mlir::cast<mlir::MemRefType>(writeOp.getSource().getType());
    int cols = memType.getShape()[1];
    std::string i0 = getVar(indices[0]);
    std::string i1 = getVar(indices[1]);
    fprintf(out_, "  __riscv_vse32_v_f32m1(%s + %s*%d + %s, %s, %d);\n",
        base.c_str(), i0.c_str(), cols, i1.c_str(), vec.c_str(), vlen);
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
    }
}
void VectorBridge::emitMemrefDim(mlir::Operation* op) {
    auto dimOp = mlir::cast<mlir::memref::DimOp>(op);
    std::string var = newVar();
    setVar(dimOp.getResult(), var);
    std::string src = getVar(dimOp.getSource());
    if (auto idxOp = dimOp.getIndex().getDefiningOp<mlir::arith::ConstantOp>()) {
        int idx = mlir::cast<mlir::IntegerAttr>(idxOp.getValue()).getInt();
        fprintf(out_, "  size_t %s = _dim_%s_%d;\n", var.c_str(), src.c_str(), idx);
    }
}
