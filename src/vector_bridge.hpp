#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include <cstdio>
#include <unordered_map>
#include <string>

// Direct RVV C intrinsic emitter for vectorized MLIR
// Handles: vector.transfer_read/write, arith.*f vector<Nxf32>, affine.for
class VectorBridge {
public:
    VectorBridge(FILE* out) : out_(out) {}
    void emitFunc(mlir::func::FuncOp func);

private:
    FILE* out_;
    std::unordered_map<void*, std::string> value_map_; // mlir::Value -> C var name
    int var_counter_ = 0;

    std::string newVar();
    std::string getVar(mlir::Value v);
    void setVar(mlir::Value v, const std::string& name);

    void emitOp(mlir::Operation* op);
    void emitTransferRead(mlir::Operation* op);
    void emitTransferWrite(mlir::Operation* op);
    void emitVectorMulf(mlir::Operation* op);
    void emitVectorAddf(mlir::Operation* op);
    void emitAffineFor(mlir::Operation* op);
    void emitConstant(mlir::Operation* op);
    void emitMemrefDim(mlir::Operation* op);
};
