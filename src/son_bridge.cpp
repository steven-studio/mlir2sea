#include "SoN/IR/SoNDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdio>
#include <unordered_map>
#include <string>

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

static ir_ref emitMathCall(ir_ctx* ctx, const char* fname, ir_type ty, ir_ref arg) {
    ir_str name = ir_string(ctx, fname);
    ir_str proto = ir_proto_1(ctx, IR_BUILTIN_FUNC, ty, ty);
    ir_ref func = ir_const_func(ctx, name, proto);
    return ir_CALL_1(ty, func, arg);
}

static ir_ref emitMathCall2(ir_ctx* ctx, const char* fname, ir_type ty, ir_ref a, ir_ref b) {
    ir_str name = ir_string(ctx, fname);
    ir_str proto = ir_proto_2(ctx, IR_BUILTIN_FUNC, ty, ty, ty);
    ir_ref func = ir_const_func(ctx, name, proto);
    return ir_CALL_2(ty, func, a, b);
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
        ir_type ty = op.getNumResults() > 0 ? mlirTypeToIR(op.getResult(0).getType()) : IR_VOID;

        // 整數二元運算
        if (auto o = mlir::dyn_cast<mlir::son::AddOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_ADD, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::SubOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SUB, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::MulOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_MUL, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::DivOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_DIV, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::RemOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_MOD, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::AndOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_AND, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::OrOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_OR, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::XorOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_XOR, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::ShlOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SHL, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::ShrOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SAR, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        // 浮點二元運算
        else if (auto o = mlir::dyn_cast<mlir::son::FAddOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_ADD, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::FSubOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SUB, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::FMulOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_MUL, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::FDivOp>(&op))
            setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_DIV, ty), getRef(o.getLhs()), getRef(o.getRhs())));
        else if (auto o = mlir::dyn_cast<mlir::son::FNegOp>(&op))
            setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_NEG, ty), getRef(o.getOperand())));
        // math unary
        else if (auto o = mlir::dyn_cast<mlir::son::SqrtOp>(&op))
            setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"sqrt":"sqrtf", ty, getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::ExpOp>(&op))
            setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"exp":"expf", ty, getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::SinOp>(&op))
            setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"sin":"sinf", ty, getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::CosOp>(&op))
            setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"cos":"cosf", ty, getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::TanhOp>(&op))
            setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"tanh":"tanhf", ty, getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::LogOp>(&op))
            setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"log":"logf", ty, getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::Log2Op>(&op))
            setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"log2":"log2f", ty, getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::AbsOp>(&op))
            setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_ABS, ty), getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::PowOp>(&op))
            setRef(o.getResult(), emitMathCall2(ctx, ty==IR_DOUBLE?"pow":"powf", ty, getRef(o.getLhs()), getRef(o.getRhs())));
        // 型別轉換
        else if (auto o = mlir::dyn_cast<mlir::son::FP2IntOp>(&op))
            setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_FP2INT, ty), getRef(o.getOperand())));
        else if (auto o = mlir::dyn_cast<mlir::son::Int2FPOp>(&op))
            setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_INT2FP, ty), getRef(o.getOperand())));
        // const
        else if (auto o = mlir::dyn_cast<mlir::arith::ConstantOp>(&op)) {
            ir_type ty = mlirTypeToIR(o.getType());
            ir_val v;
            if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(o.getValue()))
                v.i64 = attr.getInt();
            else if (auto attr = mlir::dyn_cast<mlir::FloatAttr>(o.getValue())) {
                if (ty == IR_FLOAT) v.f = (float)attr.getValueAsDouble();
                else v.d = attr.getValueAsDouble();
            }
            ir_ref c = ir_const(ctx, v, ty);
            setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_COPY, ty), c));
        }
        // return
        else if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(&op)) {
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
