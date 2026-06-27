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

void MLIRBridge::handleAddi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_ADD, ty), lhs, rhs));
}

void MLIRBridge::handleSubi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_SUB, ty), lhs, rhs));
}

void MLIRBridge::handleMuli(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_MUL, ty), lhs, rhs));
}

void MLIRBridge::handleCmpi(mlir::Operation* op) {
    auto cmpi = mlir::cast<mlir::arith::CmpIOp>(op);
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getOperand(0).getType());

    ir_op ir_opcode;
    switch (cmpi.getPredicate()) {
        case mlir::arith::CmpIPredicate::eq:  ir_opcode = IR_EQ; break;
        case mlir::arith::CmpIPredicate::ne:  ir_opcode = IR_NE; break;
        case mlir::arith::CmpIPredicate::slt: ir_opcode = IR_LT; break;
        case mlir::arith::CmpIPredicate::sle: ir_opcode = IR_LE; break;
        case mlir::arith::CmpIPredicate::sgt: ir_opcode = IR_GT; break;
        case mlir::arith::CmpIPredicate::sge: ir_opcode = IR_GE; break;
        default: ir_opcode = IR_EQ; break;
    }
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(ir_opcode, ty), lhs, rhs));
}

void MLIRBridge::handleCmpf(mlir::Operation* op) {
    auto cmpf = mlir::cast<mlir::arith::CmpFOp>(op);
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getOperand(0).getType());

    ir_op ir_opcode;
    switch (cmpf.getPredicate()) {
        case mlir::arith::CmpFPredicate::OEQ: ir_opcode = IR_EQ; break;
        case mlir::arith::CmpFPredicate::ONE: ir_opcode = IR_NE; break;
        case mlir::arith::CmpFPredicate::OLT: ir_opcode = IR_LT; break;
        case mlir::arith::CmpFPredicate::OLE: ir_opcode = IR_LE; break;
        case mlir::arith::CmpFPredicate::OGT: ir_opcode = IR_GT; break;
        case mlir::arith::CmpFPredicate::OGE: ir_opcode = IR_GE; break;
        default: ir_opcode = IR_EQ; break;
    }
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(ir_opcode, ty), lhs, rhs));
}

void MLIRBridge::handleAddf(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_ADD, ty), lhs, rhs));
}

void MLIRBridge::handleSubf(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_SUB, ty), lhs, rhs));
}

void MLIRBridge::handleDivsi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_DIV, ty), lhs, rhs));
}

void MLIRBridge::handleRemsi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_MOD, ty), lhs, rhs));
}

void MLIRBridge::handleAndi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_AND, ty), lhs, rhs));
}

void MLIRBridge::handleOri(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_OR, ty), lhs, rhs));
}

void MLIRBridge::handleXori(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_XOR, ty), lhs, rhs));
}

void MLIRBridge::handleShli(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_SHL, ty), lhs, rhs));
}

void MLIRBridge::handleShrsi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_SAR, ty), lhs, rhs));
}

void MLIRBridge::handleFptosi(mlir::Operation* op) {
    ir_ref src = getRef(op->getOperand(0));
    ir_type dst_ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold1(ctx_, IR_OPT(IR_FP2INT, dst_ty), src));
}

void MLIRBridge::handleUitofp(mlir::Operation* op) {
    ir_ref src = getRef(op->getOperand(0));
    ir_type dst_ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold1(ctx_, IR_OPT(IR_INT2FP, dst_ty), src));
}

void MLIRBridge::handleDivf(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_DIV, ty), lhs, rhs));
}

void MLIRBridge::handleNegf(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold1(ctx_, IR_OPT(IR_NEG, ty), operand));
}

void MLIRBridge::handleMulf(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold2(ctx_, IR_OPT(IR_MUL, ty), lhs, rhs));
}

void MLIRBridge::handleSitofp(mlir::Operation* op) {
    ir_ref src = getRef(op->getOperand(0));
    ir_type dst_ty = mlirTypeToIR(op->getResult(0).getType());
    setRef(op->getResult(0), ir_fold1(ctx_, IR_OPT(IR_INT2FP, dst_ty), src));
}

void MLIRBridge::handleConstant(mlir::Operation* op) {
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    ir_val v;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("value")) {
        v.i64 = attr.getInt();
    } else if (auto attr = op->getAttrOfType<mlir::FloatAttr>("value")) {
        if (ty == IR_FLOAT)
            v.f = (float)attr.getValueAsDouble();
        else
            v.d = attr.getValueAsDouble();
    } else return;
    ir_ref c = ir_const(ctx_, v, ty);
    setRef(op->getResult(0), ir_fold1(ctx_, IR_OPT(IR_COPY, ty), c));
}

void MLIRBridge::handleIndexCast(mlir::Operation* op) {
    ir_ref src = getRef(op->getOperand(0));
    ir_type src_ty = mlirTypeToIR(op->getOperand(0).getType());
    ir_type dst_ty = mlirTypeToIR(op->getResult(0).getType());
    ir_ref result;
    if (src_ty == IR_ADDR && dst_ty == IR_I32)
        result = ir_fold1(ctx_, IR_OPT(IR_TRUNC, dst_ty), src);
    else
        result = ir_fold1(ctx_, IR_OPT(IR_ZEXT, dst_ty), src);
    setRef(op->getResult(0), result);
}

