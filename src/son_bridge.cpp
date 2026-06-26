#include "SoN/IR/SoNDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdio>
#include <unordered_map>

extern "C" {
#include "ir.h"
#include "ir_builder.h"
}

static ir_ctx* g_ctx = nullptr;
#define _ir_CTX g_ctx

static std::unordered_map<void*, ir_ref> value_map;

static ir_ref getRef(mlir::Value v) {
    auto it = value_map.find(v.getAsOpaquePointer());
    if (it == value_map.end()) return IR_UNUSED;
    return it->second;
}

static void setRef(mlir::Value v, ir_ref ref) {
    value_map[v.getAsOpaquePointer()] = ref;
}

static ir_type mlirTypeToIR(mlir::Type type) {
    if (type.isInteger(1))  return IR_BOOL;
    if (type.isInteger(8))  return IR_I8;
    if (type.isInteger(16)) return IR_I16;
    if (type.isInteger(32)) return IR_I32;
    if (type.isInteger(64)) return IR_I64;
    if (type.isF32())       return IR_FLOAT;
    if (type.isF64())       return IR_DOUBLE;
    if (type.isa<mlir::IndexType>()) return IR_ADDR;
    return IR_I32;
}

void emitSoNFuncToIR(mlir::func::FuncOp func, ir_ctx* ctx, FILE* outFile) {
    g_ctx = ctx;
    value_map.clear();

    ir_init(ctx, IR_FUNCTION, 256, 256);
    auto resultTypes = func.getFunctionType().getResults();
    ctx->ret_type = resultTypes.empty() ? IR_VOID : mlirTypeToIR(resultTypes[0]);

    ir_START();

    int pos = 1;
    for (auto arg : func.getArguments()) {
        std::string pname = "p" + std::to_string(pos - 1);
        ir_type ty = mlirTypeToIR(arg.getType());
        ir_ref param = ir_PARAM(ty, pname.c_str(), pos++);
        setRef(arg, param);
    }

    for (auto& op : func.getBody().front()) {
        if (auto addOp = mlir::dyn_cast<mlir::son::AddOp>(&op)) {
            ir_ref lhs = getRef(addOp.getLhs());
            ir_ref rhs = getRef(addOp.getRhs());
            ir_type ty = mlirTypeToIR(addOp.getType());
            setRef(addOp.getResult(), ir_fold2(ctx, IR_OPT(IR_ADD, ty), lhs, rhs));
        } else if (auto subOp = mlir::dyn_cast<mlir::son::SubOp>(&op)) {
            ir_ref lhs = getRef(subOp.getLhs());
            ir_ref rhs = getRef(subOp.getRhs());
            ir_type ty = mlirTypeToIR(subOp.getType());
            setRef(subOp.getResult(), ir_fold2(ctx, IR_OPT(IR_SUB, ty), lhs, rhs));
        } else if (auto mulOp = mlir::dyn_cast<mlir::son::MulOp>(&op)) {
            ir_ref lhs = getRef(mulOp.getLhs());
            ir_ref rhs = getRef(mulOp.getRhs());
            ir_type ty = mlirTypeToIR(mulOp.getType());
            setRef(mulOp.getResult(), ir_fold2(ctx, IR_OPT(IR_MUL, ty), lhs, rhs));
        } else if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(&op)) {
            if (retOp.getNumOperands() > 0)
                ir_RETURN(getRef(retOp.getOperand(0)));
            else
                ir_RETURN(IR_UNUSED);
        }
    }

    ir_build_def_use_lists(ctx);
    ir_build_cfg(ctx);
    ir_build_dominators_tree(ctx);
    ir_find_loops(ctx);
    ir_gcm(ctx);
    ir_schedule(ctx);
    ir_assign_virtual_registers(ctx);
    ir_compute_live_ranges(ctx);
    ir_coalesce(ctx);
    ir_emit_c(ctx, func.getName().str().c_str(), outFile);
}
