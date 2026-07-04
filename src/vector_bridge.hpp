#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include <cstdio>
#include <unordered_map>
#include <string>
#include <map>
#include <vector>

struct LoopCtx { std::string iv; std::string hi; long step; };

struct VecTypeInfo {
    std::string vecCType;   // "vfloat32m1_t" or "vfloat64m1_t"
    std::string bitwidth;   // "32" or "64"  (used in vle32/vle64, vse32/vse64)
    std::string suffix;     // "f32m1" or "f64m1" (used in vfmul_vv_f32m1 etc.)
    std::string scalarCType; // "float" or "double"
};

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
    std::string getPhysicalStride(mlir::Value memref, int dim); // element stride for a given dim, from the memref's actual layout
    std::string computeVL(int vlen); // 算這次迭代實際該用的向量長度（處理非對齊邊界）
    std::string computeSafeVL(mlir::Value memref, mlir::Value lastIndexValue, int vlen); // per-memref clipped vl for vle/vse only
    std::string affineExprToStr(mlir::AffineExpr expr,
                            const std::vector<std::string>& dimVars,
                            const std::vector<std::string>& symVars);
    VecTypeInfo getVecTypeInfo(mlir::Type elemType, int vlen);
    std::string getScalarCType(mlir::Type memrefElemType); // for function signature (float*/double*)
    
    void emitOp(mlir::Operation* op);
    void emitTransferRead(mlir::Operation* op);
    void emitTransferWrite(mlir::Operation* op);
    void emitVectorMulf(mlir::Operation* op);
    void emitVectorAddf(mlir::Operation* op);
    void emitVectorSubf(mlir::Operation* op);
    void emitVectorDivf(mlir::Operation* op);
    void emitVectorBroadcast(mlir::Operation* op);
    void emitAffineFor(mlir::Operation* op);
    void emitAffineApply(mlir::Operation* op);
    void emitConstant(mlir::Operation* op);
    void emitSubI(mlir::Operation* op);
    void emitAddI(mlir::Operation* op);
    void emitMemrefDim(mlir::Operation* op);
};