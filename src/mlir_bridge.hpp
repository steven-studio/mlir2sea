#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <unordered_map>
#include <cstdio>

extern "C" {
#include "ir.h"
}

// mlir::Value -> ir_ref mapping
struct MLIRValueMapInfo {
    static mlir::Value getEmptyKey() { return {}; }
    static mlir::Value getTombstoneKey() { return mlir::Value::getFromOpaquePointer((void*)1); }
};

class MLIRBridge {
public:
    MLIRBridge();
    ~MLIRBridge();

    void build(mlir::func::FuncOp func);
    void emitC(const std::string& funcName, FILE* outFile);
    ir_ctx* getCtx() { return ctx_; }

private:
    ir_ctx* ctx_ = nullptr;
    std::unordered_map<void*, ir_ref> value_map_;  // mlir::Value.getAsOpaquePointer() -> ir_ref

    ir_ref getRef(mlir::Value v);
    void setRef(mlir::Value v, ir_ref ref);

    void handleFunc(mlir::func::FuncOp func);
    void handleOp(mlir::Operation* op);

    ir_type mlirTypeToIR(mlir::Type type);  // ← 加這行

    // arith
    void handleAddi(mlir::Operation* op);
    void handleSubi(mlir::Operation* op);
    void handleMuli(mlir::Operation* op);
    void handleCmpi(mlir::Operation* op);
    void handleCmpf(mlir::Operation* op);
    void handleAddf(mlir::Operation* op);
    void handleDivf(mlir::Operation* op);
    void handleNegf(mlir::Operation* op);
    void handleConstant(mlir::Operation* op);
    void handleIndexCast(mlir::Operation* op);
    void handleMathSqrt(mlir::Operation* op);
    void handleMathExp(mlir::Operation* op);
    void handleMathAbs(mlir::Operation* op);

    // func
    void handleReturn(mlir::Operation* op);

    // scf
    void handleIf(mlir::Operation* op);
    void handleFor(mlir::Operation* op);
};
