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

void MLIRBridge::handleIndexSwitch(mlir::Operation* op) {
    auto switchOp = mlir::cast<mlir::scf::IndexSwitchOp>(op);
    ir_ref val = getRef(switchOp.getArg());
    auto cases = switchOp.getCases();
    auto regions = switchOp.getCaseRegions();

    llvm::SmallVector<ir_ref> ends;
    llvm::SmallVector<ir_ref> vals;

    for (auto [caseVal, region] : llvm::zip(cases, regions)) {
        ir_val v; v.i64 = caseVal;
        ir_ref c = ir_const(ctx_, v, IR_ADDR);
        ir_ref cond = ir_fold2(ctx_, IR_OPT(IR_EQ, IR_ADDR), val, c);
        ir_ref if_ref = ir_IF(cond);

        ir_IF_TRUE(if_ref);
        for (auto& nested : region.front().without_terminator())
            handleOp(&nested);
        auto yield = mlir::cast<mlir::scf::YieldOp>(region.front().getTerminator());
        ir_ref case_val = yield.getNumOperands() > 0 ? getRef(yield.getOperand(0)) : IR_UNUSED;
        vals.push_back(case_val);
        ends.push_back(ir_END());

        ir_IF_FALSE(if_ref);
    }

    // default
    auto& defaultRegion = switchOp.getDefaultRegion();
    for (auto& nested : defaultRegion.front().without_terminator())
        handleOp(&nested);
    auto defaultYield = mlir::cast<mlir::scf::YieldOp>(defaultRegion.front().getTerminator());
    ir_ref default_val = defaultYield.getNumOperands() > 0 ? getRef(defaultYield.getOperand(0)) : IR_UNUSED;
    vals.push_back(default_val);
    ends.push_back(ir_END());

    // MERGE_N
    ir_ref* ends_arr = ends.data();
    ir_MERGE_N(ends.size(), ends_arr);

    if (switchOp.getNumResults() > 0 && !vals.empty()) {
        ir_type ty = mlirTypeToIR(switchOp.getResult(0).getType());
        ir_ref* vals_arr = vals.data();
        ir_ref phi = ir_PHI_N(ty, vals.size(), vals_arr);
        setRef(switchOp.getResult(0), phi);
    }
}

void MLIRBridge::handleIf(mlir::Operation* op) {
    auto ifOp = mlir::cast<mlir::scf::IfOp>(op);
    ir_ref cond = getRef(ifOp.getCondition());
    ir_ref if_ref = ir_IF(cond);

    ir_IF_TRUE(if_ref);
    ir_ref then_val = IR_UNUSED;
    for (auto& nested : ifOp.getThenRegion().front()) {
        if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(&nested)) {
            if (yield.getNumOperands() > 0)
                then_val = getRef(yield.getOperand(0));
        } else {
            handleOp(&nested);
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
                handleOp(&nested);
            }
        }
    }
    ir_ref else_end = ir_END();

    ir_MERGE_2(then_end, else_end);
    if (ifOp.getNumResults() > 0 && then_val != IR_UNUSED && else_val != IR_UNUSED) {
        ir_type ty = mlirTypeToIR(ifOp.getResult(0).getType());
        setRef(ifOp.getResult(0), ir_PHI_2(ty, then_val, else_val));
    }
}

void MLIRBridge::handleFor(mlir::Operation* op) {
    auto forOp = mlir::cast<mlir::scf::ForOp>(op);

    ir_ref lb   = getRef(forOp.getLowerBound());
    ir_ref ub   = getRef(forOp.getUpperBound());
    ir_ref step = getRef(forOp.getStep());

    ir_ref entry_end = ir_END();
    fprintf(stderr, "after END, control=%d\n", ctx_->control);
    ir_ref loop = ir_LOOP_BEGIN(entry_end);
    fprintf(stderr, "after LOOP_BEGIN, control=%d\n", ctx_->control);
    ctx_->control = loop;
    fprintf(stderr, "loop=%d control=%d op=%d\n", loop, ctx_->control, ctx_->ir_base[ctx_->control].op);

    ir_type idx_ty = mlirTypeToIR(forOp.getInductionVar().getType());
    ir_ref i_phi = _ir_PHI_2(ctx_, idx_ty, lb, IR_UNUSED);
    fprintf(stderr, "i_phi=%d\n", i_phi);
    setRef(forOp.getInductionVar(), i_phi);

    for (auto [arg, init] : llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
        ir_type ty = mlirTypeToIR(arg.getType());
        ir_ref phi = _ir_PHI_2(ctx_, ty, getRef(init), IR_UNUSED);
        setRef(arg, phi);
    }

    // pre-check
    ir_ref cond = ir_emit2(ctx_, IR_OPT(IR_LT, idx_ty), i_phi, ub);
    ir_ref if_ref = ir_IF(cond);
    ir_IF_TRUE(if_ref);
    fprintf(stderr, "after IF_TRUE, control=%d\n", ctx_->control);

    for (auto& nested : forOp.getBody()->without_terminator())
        handleOp(&nested);

    ir_ref i_next = ir_emit2(ctx_, IR_OPT(IR_ADD, idx_ty), i_phi, step);
    ir_PHI_SET_OP(i_phi, 2, i_next);

    auto yield = mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto [arg, yval] : llvm::zip(forOp.getRegionIterArgs(), yield.getOperands()))
        ir_PHI_SET_OP(getRef(arg), 2, getRef(yval));

    ir_ref loop_end = ir_LOOP_END();
    fprintf(stderr, "after LOOP_END, control=%d\n", ctx_->control);
    ir_MERGE_SET_OP(loop, 2, loop_end);
    ir_IF_FALSE(if_ref);
    fprintf(stderr, "after IF_FALSE, control=%d\n", ctx_->control);
    fprintf(stderr, "handleFor exit, control=%d\n", ctx_->control);
    for (auto [result, arg] : llvm::zip(forOp.getResults(), forOp.getRegionIterArgs()))
        setRef(result, getRef(arg));
    fprintf(stderr, "handleFor exit, control=%d\n", ctx_->control);
    fprintf(stderr, "handleFor exit\n");
}

void MLIRBridge::handleWhile(mlir::Operation* op) {
    auto whileOp = mlir::cast<mlir::scf::WhileOp>(op);

    // entry → loop
    ir_ref loop = ir_LOOP_BEGIN(ir_END());

    // iter args PHI
    auto beforeArgs = whileOp.getBeforeArguments();
    auto initArgs = whileOp.getInits();
    for (auto [arg, init] : llvm::zip(beforeArgs, initArgs)) {
        ir_type ty = mlirTypeToIR(arg.getType());
        ir_ref phi = _ir_PHI_2(ctx_, ty, getRef(init), IR_UNUSED);
        setRef(arg, phi);
    }

    // before region（條件）
    auto& beforeBlock = whileOp.getBefore().front();
    for (auto& nested : beforeBlock.without_terminator())
        handleOp(&nested);

    // condition
    auto condOp = mlir::cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());
    ir_ref cond = getRef(condOp.getCondition());
    ir_ref if_ref = ir_IF(cond);
    ir_IF_TRUE(if_ref);

    // after region（body）
    auto& afterBlock = whileOp.getAfter().front();
    
    // after block 的 args 對應 condOp 的 args
    for (auto [afterArg, condArg] : llvm::zip(whileOp.getAfterArguments(), condOp.getArgs()))
        setRef(afterArg, getRef(condArg));

    for (auto& nested : afterBlock.without_terminator())
        handleOp(&nested);

    // yield → back-edge
    auto yield = mlir::cast<mlir::scf::YieldOp>(afterBlock.getTerminator());
    for (auto [arg, yval] : llvm::zip(beforeArgs, yield.getOperands()))
        ir_PHI_SET_OP(getRef(arg), 2, getRef(yval));

    ir_ref loop_end = ir_LOOP_END();
    ir_MERGE_SET_OP(loop, 2, loop_end);

    // exit
    ir_IF_FALSE(if_ref);
    for (auto [result, condArg] : llvm::zip(whileOp.getResults(), condOp.getArgs()))
        setRef(result, getRef(condArg));
}

