#include "SoN/IR/SoNDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    if (type.isa<mlir::MemRefType>()) return IR_ADDR;
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

static void emitSoNOp(ir_ctx* ctx, mlir::Operation* op);

static void emitBlock(ir_ctx* ctx, mlir::Block& block) {
    for (auto& op : block)
        emitSoNOp(ctx, &op);
}

static void emitSoNOp(ir_ctx* ctx, mlir::Operation* op) {
    ir_type ty = op->getNumResults() > 0 ? mlirTypeToIR(op->getResult(0).getType()) : IR_VOID;

    // arith constant
    if (auto o = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
        ir_val v = {};
        if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(o.getValue()))
            v.i64 = attr.getInt();
        else if (auto attr = mlir::dyn_cast<mlir::FloatAttr>(o.getValue())) {
            if (ty == IR_FLOAT) v.f = (float)attr.getValueAsDouble();
            else v.d = attr.getValueAsDouble();
        }
        ir_ref c = ir_const(ctx, v, ty);
        setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_COPY, ty), c));
    }
    // arith compare
    else if (auto o = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
        ir_ref lhs = getRef(o.getLhs());
        ir_ref rhs = getRef(o.getRhs());
        ir_type cmp_ty = mlirTypeToIR(o.getLhs().getType());
        ir_op opcode;
        switch (o.getPredicate()) {
            case mlir::arith::CmpIPredicate::eq:  opcode = IR_EQ; break;
            case mlir::arith::CmpIPredicate::ne:  opcode = IR_NE; break;
            case mlir::arith::CmpIPredicate::slt: opcode = IR_LT; break;
            case mlir::arith::CmpIPredicate::sle: opcode = IR_LE; break;
            case mlir::arith::CmpIPredicate::sgt: opcode = IR_GT; break;
            case mlir::arith::CmpIPredicate::sge: opcode = IR_GE; break;
            default: opcode = IR_EQ; break;
        }
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(opcode, cmp_ty), lhs, rhs));
    }
    else if (auto o = mlir::dyn_cast<mlir::arith::CmpFOp>(op)) {
        ir_ref lhs = getRef(o.getLhs());
        ir_ref rhs = getRef(o.getRhs());
        ir_type cmp_ty = mlirTypeToIR(o.getLhs().getType());
        ir_op opcode;
        switch (o.getPredicate()) {
            case mlir::arith::CmpFPredicate::OEQ: opcode = IR_EQ; break;
            case mlir::arith::CmpFPredicate::ONE: opcode = IR_NE; break;
            case mlir::arith::CmpFPredicate::OLT: opcode = IR_LT; break;
            case mlir::arith::CmpFPredicate::OLE: opcode = IR_LE; break;
            case mlir::arith::CmpFPredicate::OGT: opcode = IR_GT; break;
            case mlir::arith::CmpFPredicate::OGE: opcode = IR_GE; break;
            default: opcode = IR_EQ; break;
        }
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(opcode, cmp_ty), lhs, rhs));
    }
    // 整數二元
    else if (auto o = mlir::dyn_cast<mlir::son::AddOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_ADD, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::SubOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SUB, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::MulOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_MUL, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::DivOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_DIV, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::RemOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_MOD, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::AndOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_AND, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::OrOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_OR, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::XorOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_XOR, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::ShlOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SHL, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::ShrOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SAR, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    // 浮點二元
    else if (auto o = mlir::dyn_cast<mlir::son::FAddOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_ADD, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::FSubOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_SUB, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::FMulOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_MUL, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::FDivOp>(op))
        setRef(o.getResult(), ir_fold2(ctx, IR_OPT(IR_DIV, ty), getRef(o.getLhs()), getRef(o.getRhs())));
    else if (auto o = mlir::dyn_cast<mlir::son::FNegOp>(op))
        setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_NEG, ty), getRef(o.getOperand())));
    // math unary
    else if (auto o = mlir::dyn_cast<mlir::son::SqrtOp>(op))
        setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"sqrt":"sqrtf", ty, getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::ExpOp>(op))
        setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"exp":"expf", ty, getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::SinOp>(op))
        setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"sin":"sinf", ty, getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::CosOp>(op))
        setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"cos":"cosf", ty, getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::TanhOp>(op))
        setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"tanh":"tanhf", ty, getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::LogOp>(op))
        setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"log":"logf", ty, getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::Log2Op>(op))
        setRef(o.getResult(), emitMathCall(ctx, ty==IR_DOUBLE?"log2":"log2f", ty, getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::AbsOp>(op))
        setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_ABS, ty), getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::PowOp>(op))
        setRef(o.getResult(), emitMathCall2(ctx, ty==IR_DOUBLE?"pow":"powf", ty, getRef(o.getLhs()), getRef(o.getRhs())));
    // 型別轉換
    else if (auto o = mlir::dyn_cast<mlir::son::FP2IntOp>(op))
        setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_FP2INT, ty), getRef(o.getOperand())));
    else if (auto o = mlir::dyn_cast<mlir::son::Int2FPOp>(op))
        setRef(o.getResult(), ir_fold1(ctx, IR_OPT(IR_INT2FP, ty), getRef(o.getOperand())));
    // scf.if
    else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        ir_ref cond = getRef(ifOp.getCondition());
        ir_ref if_ref = ir_IF(cond);

        ir_IF_TRUE(if_ref);
        ir_ref then_val = IR_UNUSED;
        for (auto& nested : ifOp.getThenRegion().front()) {
            if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(&nested)) {
                if (yield.getNumOperands() > 0)
                    then_val = getRef(yield.getOperand(0));
            } else {
                emitSoNOp(ctx, &nested);
            }
        }
        ir_ref then_end = ir_END();

        ir_IF_FALSE(if_ref);
        ir_ref else_val = IR_UNUSED;
        if (!ifOp.getElseRegion().empty()) {
            for (auto& nested : ifOp.getElseRegion().front()) {
                if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(&nested)) {
                    if (yield.getNumOperands() > 0)
                        else_val = getRef(yield.getOperand(0));
                } else {
                    emitSoNOp(ctx, &nested);
                }
            }
        }
        ir_ref else_end = ir_END();

        ir_MERGE_2(then_end, else_end);
        if (ifOp.getNumResults() > 0 && then_val != IR_UNUSED && else_val != IR_UNUSED) {
            ir_type rty = mlirTypeToIR(ifOp.getResult(0).getType());
            setRef(ifOp.getResult(0), ir_PHI_2(rty, then_val, else_val));
        }
    }
    // scf.for
    else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        ir_ref lb   = getRef(forOp.getLowerBound());
        ir_ref ub   = getRef(forOp.getUpperBound());
        ir_ref step = getRef(forOp.getStep());

        ir_ref entry_end = ir_END();
        ir_ref loop = ir_LOOP_BEGIN(entry_end);

        ir_type idx_ty = mlirTypeToIR(forOp.getInductionVar().getType());
        ir_ref i_phi = _ir_PHI_2(ctx, idx_ty, lb, IR_UNUSED);
        setRef(forOp.getInductionVar(), i_phi);

        for (auto [arg, init] : llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
            ir_type aty = mlirTypeToIR(arg.getType());
            ir_ref phi = _ir_PHI_2(ctx, aty, getRef(init), IR_UNUSED);
            setRef(arg, phi);
        }

        // pre-check
        ir_ref cond = ir_fold2(ctx, IR_OPT(IR_LT, idx_ty), i_phi, ub);
        ir_ref if_ref = ir_IF(cond);
        ir_IF_TRUE(if_ref);

        for (auto& nested : forOp.getBody()->without_terminator())
            emitSoNOp(ctx, &nested);

        ir_ref i_next = ir_fold2(ctx, IR_OPT(IR_ADD, idx_ty), i_phi, step);
        ir_PHI_SET_OP(i_phi, 2, i_next);

        auto yield = mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
        for (auto [arg, yval] : llvm::zip(forOp.getRegionIterArgs(), yield.getOperands()))
            ir_PHI_SET_OP(getRef(arg), 2, getRef(yval));

        ir_ref loop_end = ir_LOOP_END();
        ir_MERGE_SET_OP(loop, 2, loop_end);

        ir_IF_FALSE(if_ref);
        for (auto [result, arg] : llvm::zip(forOp.getResults(), forOp.getRegionIterArgs()))
            setRef(result, getRef(arg));
    }
    // scf.while
    else if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
        ir_ref loop = ir_LOOP_BEGIN(ir_END());

        auto beforeArgs = whileOp.getBeforeArguments();
        auto initArgs = whileOp.getInits();
        for (auto [arg, init] : llvm::zip(beforeArgs, initArgs)) {
            ir_type aty = mlirTypeToIR(arg.getType());
            ir_ref phi = _ir_PHI_2(ctx, aty, getRef(init), IR_UNUSED);
            setRef(arg, phi);
        }

        // before region（條件）
        for (auto& nested : whileOp.getBefore().front().without_terminator())
            emitSoNOp(ctx, &nested);

        auto condOp = mlir::cast<mlir::scf::ConditionOp>(
            whileOp.getBefore().front().getTerminator());
        ir_ref cond = getRef(condOp.getCondition());
        ir_ref if_ref = ir_IF(cond);
        ir_IF_TRUE(if_ref);

        // after block args 對應 condOp args
        for (auto [afterArg, condArg] : llvm::zip(
                whileOp.getAfterArguments(), condOp.getArgs()))
            setRef(afterArg, getRef(condArg));

        // after region（body）
        for (auto& nested : whileOp.getAfter().front().without_terminator())
            emitSoNOp(ctx, &nested);

        // yield back-edge
        auto yield = mlir::cast<mlir::scf::YieldOp>(
            whileOp.getAfter().front().getTerminator());
        for (auto [arg, yval] : llvm::zip(beforeArgs, yield.getOperands()))
            ir_PHI_SET_OP(getRef(arg), 2, getRef(yval));

        ir_ref loop_end = ir_LOOP_END();
        ir_MERGE_SET_OP(loop, 2, loop_end);

        ir_IF_FALSE(if_ref);
        for (auto [result, condArg] : llvm::zip(
                whileOp.getResults(), condOp.getArgs()))
            setRef(result, getRef(condArg));
    }
    else if (auto o = mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
        ir_ref src = getRef(o.getIn());
        ir_type dst_ty = mlirTypeToIR(o.getType());
        ir_type src_ty = mlirTypeToIR(o.getIn().getType());
        ir_ref result;
        if (src_ty == IR_ADDR && dst_ty == IR_I32)
            result = ir_fold1(ctx, IR_OPT(IR_TRUNC, dst_ty), src);
        else
            result = ir_fold1(ctx, IR_OPT(IR_ZEXT, dst_ty), src);
        setRef(o.getResult(), result);
    }
    // func.return
    else if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
        if (retOp.getNumOperands() > 0)
            ir_RETURN(getRef(retOp.getOperand(0)));
        else
            ir_RETURN(IR_UNUSED);
    }
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

    emitBlock(ctx, func.getBody().front());

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
