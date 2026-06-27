#include "../mlir_bridge.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <iostream>

extern "C" {
#include "ir.h"
#include "ir_builder.h"
int ir_emit_riscv(ir_ctx *ctx, const char *name, FILE *f);
}

extern ir_ctx* g_ctx;
#define _ir_CTX g_ctx

void MLIRBridge::handleMathSqrt(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "sqrt" : "sqrtf";
    ir_str name = ir_string(ctx_, fname);
    ir_str proto = ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty);
    ir_ref sqrt_func = ir_const_func(ctx_, name, proto);
    setRef(op->getResult(0), ir_CALL_1(ty, sqrt_func, operand));
}

void MLIRBridge::handleMathSin(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "sin" : "sinf";
    ir_ref func = ir_const_func(ctx_, ir_string(ctx_, fname), ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty));
    setRef(op->getResult(0), ir_CALL_1(ty, func, operand));
}

void MLIRBridge::handleMathCos(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "cos" : "cosf";
    ir_ref func = ir_const_func(ctx_, ir_string(ctx_, fname), ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty));
    setRef(op->getResult(0), ir_CALL_1(ty, func, operand));
}

void MLIRBridge::handleMathTanh(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "tanh" : "tanhf";
    ir_ref func = ir_const_func(ctx_, ir_string(ctx_, fname), ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty));
    setRef(op->getResult(0), ir_CALL_1(ty, func, operand));
}

void MLIRBridge::handleMathLog(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "log" : "logf";
    ir_ref func = ir_const_func(ctx_, ir_string(ctx_, fname), ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty));
    setRef(op->getResult(0), ir_CALL_1(ty, func, operand));
}

void MLIRBridge::handleMathLog2(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "log2" : "log2f";
    ir_ref func = ir_const_func(ctx_, ir_string(ctx_, fname), ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty));
    setRef(op->getResult(0), ir_CALL_1(ty, func, operand));
}

void MLIRBridge::handleMathPow(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "pow" : "powf";
    ir_str proto = ir_proto_2(ctx_, IR_BUILTIN_FUNC, ty, ty, ty);
    ir_ref func = ir_const_func(ctx_, ir_string(ctx_, fname), proto);
    setRef(op->getResult(0), ir_CALL_2(ty, func, lhs, rhs));
}

void MLIRBridge::handleMathExp(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "exp" : "expf";
    ir_str name = ir_string(ctx_, fname);
    ir_str proto = ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty);
    ir_ref exp_func = ir_const_func(ctx_, name, proto);
    setRef(op->getResult(0), ir_CALL_1(ty, exp_func, operand));
}

void MLIRBridge::handleMathAbs(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold1(ctx_, IR_OPT(IR_ABS, ty), operand));
}

