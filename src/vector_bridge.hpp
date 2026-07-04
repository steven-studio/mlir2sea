#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include <cstdio>
#include <unordered_map>
#include <string>
#include <map>
#include <vector>

struct LoopCtx { std::string iv; std::string hi; long step; };

class VectorBridge {
public:
    VectorBridge(FILE* out) : out_(out) {}
    void emitFunc(mlir::func::FuncOp func);

private:
    FILE* out_;
    std::unordered_map<void*, std::string> value_map_;
    std::unordered_map<void*, std::map<int, std::string>> dim_map_;
    int var_counter_ = 0;
    std::vector<LoopCtx> loop_stack_; // 追蹤目前巢狀的 affine.for，用來算 tail vl

    std::string newVar();
    std::string getVar(mlir::Value v);
    void setVar(mlir::Value v, const std::string& name);

    std::string getDim(mlir::Value memref, int idx);
    void setDim(mlir::Value memref, int idx, const std::string& name);
    std::string getDimExpr(mlir::Value memref, int idx);
    void collectDynamicDims(mlir::Operation* root);
    std::string computeFlatOffset(mlir::Value memref, mlir::Operation::operand_range indices);
    std::string computeVL(int vlen); // 算這次迭代實際該用的向量長度（處理非對齊邊界）

    void emitOp(mlir::Operation* op);
    void emitTransferRead(mlir::Operation* op);
    void emitTransferWrite(mlir::Operation* op);
    void emitVectorMulf(mlir::Operation* op);
    void emitVectorAddf(mlir::Operation* op);
    void emitVectorBroadcast(mlir::Operation* op);
    void emitAffineFor(mlir::Operation* op);
    void emitConstant(mlir::Operation* op);
    void emitSubI(mlir::Operation* op);
    void emitAddI(mlir::Operation* op);
    void emitMemrefDim(mlir::Operation* op);
};