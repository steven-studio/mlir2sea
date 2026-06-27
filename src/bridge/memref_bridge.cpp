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

void MLIRBridge::handleMemrefLoad(mlir::Operation* op) {
    auto loadOp = mlir::cast<mlir::memref::LoadOp>(op);
    auto memrefType = mlir::cast<mlir::MemRefType>(loadOp.getMemref().getType());
    ir_ref base = getRef(loadOp.getMemref());
    ir_type elem_ty = mlirTypeToIR(memrefType.getElementType());

    // 計算 linear index
    auto shape = memrefType.getShape();
    auto indices = loadOp.getIndices();
    ir_ref offset = getRef(indices[0]);
    for (size_t i = 1; i < indices.size(); i++) {
        ir_val v; v.i64 = shape[i];
        ir_ref dim = ir_const(ctx_, v, IR_ADDR);
        offset = ir_fold2(ctx_, IR_OPT(IR_MUL, IR_ADDR), offset, dim);
        offset = ir_fold2(ctx_, IR_OPT(IR_ADD, IR_ADDR), offset, getRef(indices[i]));
    }

    // elem_size
    ir_val sz; sz.i64 = 4; // f32 = 4 bytes
    ir_ref elem_size = ir_const(ctx_, sz, IR_ADDR);
    ir_ref byte_offset = ir_fold2(ctx_, IR_OPT(IR_MUL, IR_ADDR), offset, elem_size);
    ir_ref ptr = ir_fold2(ctx_, IR_OPT(IR_ADD, IR_ADDR), base, byte_offset);

    ir_ref result = ir_LOAD(elem_ty, ptr);
    setRef(op->getResult(0), result);
}

void MLIRBridge::handleMemrefStore(mlir::Operation* op) {
    auto storeOp = mlir::cast<mlir::memref::StoreOp>(op);
    auto memrefType = mlir::cast<mlir::MemRefType>(storeOp.getMemref().getType());
    ir_ref base = getRef(storeOp.getMemref());
    ir_ref value = getRef(storeOp.getValue());
    ir_type elem_ty = mlirTypeToIR(memrefType.getElementType());

    auto shape = memrefType.getShape();
    auto indices = storeOp.getIndices();
    ir_ref offset = getRef(indices[0]);
    for (size_t i = 1; i < indices.size(); i++) {
        ir_val v; v.i64 = shape[i];
        ir_ref dim = ir_const(ctx_, v, IR_ADDR);
        offset = ir_fold2(ctx_, IR_OPT(IR_MUL, IR_ADDR), offset, dim);
        offset = ir_fold2(ctx_, IR_OPT(IR_ADD, IR_ADDR), offset, getRef(indices[i]));
    }

    ir_val sz; sz.i64 = 4;
    ir_ref elem_size = ir_const(ctx_, sz, IR_ADDR);
    ir_ref byte_offset = ir_fold2(ctx_, IR_OPT(IR_MUL, IR_ADDR), offset, elem_size);
    ir_ref ptr = ir_fold2(ctx_, IR_OPT(IR_ADD, IR_ADDR), base, byte_offset);

    ir_STORE(ptr, value);
}

void MLIRBridge::handleReturn(mlir::Operation* op) {
    if (op->getNumOperands() == 0) {
        ir_RETURN(IR_UNUSED);  // void return
        return;
    }
    ir_RETURN(getRef(op->getOperand(0)));
}

