#include "mlir_bridge.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <iostream>

extern "C" {
#include "ir.h"
#include "ir_builder.h"
}

// 讓所有 ir_builder macro 能找到 ctx
static ir_ctx* g_ctx = nullptr;
#define _ir_CTX g_ctx

MLIRBridge::MLIRBridge() {
    ctx_ = (ir_ctx*)malloc(sizeof(ir_ctx));
}

MLIRBridge::~MLIRBridge() {
    if (ctx_) { ir_free(ctx_); free(ctx_); }
}

ir_ref MLIRBridge::getRef(mlir::Value v) {
    auto it = value_map_.find(v.getAsOpaquePointer());
    if (it == value_map_.end()) return IR_UNUSED;
    return it->second;
}

void MLIRBridge::setRef(mlir::Value v, ir_ref ref) {
    value_map_[v.getAsOpaquePointer()] = ref;
}

ir_type MLIRBridge::mlirTypeToIR(mlir::Type type) {
    if (type.isInteger(1))  return IR_BOOL;
    if (type.isInteger(8))  return IR_I8;
    if (type.isInteger(16)) return IR_I16;
    if (type.isInteger(32)) return IR_I32;
    if (type.isInteger(64)) return IR_I64;
    if (type.isF32())       return IR_FLOAT;
    if (type.isF64())       return IR_DOUBLE;
    if (type.isa<mlir::IndexType>()) return IR_ADDR;
    if (type.isa<mlir::MemRefType>()) return IR_ADDR;
    llvm_unreachable("unsupported type");
}

void MLIRBridge::build(mlir::func::FuncOp func) {
    g_ctx = ctx_;
    ir_init(ctx_, IR_FUNCTION, 1024, 1024);

    auto resultTypes = func.getFunctionType().getResults();
    ctx_->ret_type = resultTypes.empty() ? IR_VOID : mlirTypeToIR(resultTypes[0]);

    ir_START();

    int pos = 1;
    for (auto arg : func.getArguments()) {
        std::string pname = "p" + std::to_string(pos - 1);
        ir_type ty = mlirTypeToIR(arg.getType());
        ir_ref param = ir_PARAM(ty, pname.c_str(), pos++);
        setRef(arg, param);
    }

    for (auto& op : func.getBody().front())
        handleOp(&op);
}

void MLIRBridge::handleOp(mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "arith.addi")       handleAddi(op);
    else if (name == "arith.subi")  handleSubi(op);
    else if (name == "arith.muli")  handleMuli(op);
    else if (name == "arith.cmpi")  handleCmpi(op);
    else if (name == "arith.cmpf")  handleCmpf(op);
    else if (name == "arith.addf")  handleAddf(op);
    else if (name == "arith.divf")  handleDivf(op);
    else if (name == "arith.negf")  handleNegf(op);
    else if (name == "arith.mulf")   handleMulf(op);
    else if (name == "arith.sitofp") handleSitofp(op);
    else if (name == "arith.constant") handleConstant(op);
    else if (name == "arith.index_cast") handleIndexCast(op);
    else if (name == "memref.load")  handleMemrefLoad(op);
    else if (name == "memref.store") handleMemrefStore(op);
    else if (name == "math.sqrt")   handleMathSqrt(op);
    else if (name == "math.exp")    handleMathExp(op);
    else if (name == "math.absf")    handleMathAbs(op);
    else if (name == "func.return") handleReturn(op);
    else if (name == "scf.if")      handleIf(op);
    else if (name == "scf.for")     handleFor(op);
    else if (name == "scf.while")   handleWhile(op);
    else if (name == "scf.index_switch") handleIndexSwitch(op);
    else fprintf(stderr, "unhandled op: %s\n", name.str().c_str());
}

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

void MLIRBridge::handleMathSqrt(mlir::Operation* op) {
    ir_ref operand = getRef(op->getOperand(0));
    ir_type ty = mlirTypeToIR(op->getResult(0).getType());
    const char* fname = (ty == IR_DOUBLE) ? "sqrt" : "sqrtf";
    ir_str name = ir_string(ctx_, fname);
    ir_str proto = ir_proto_1(ctx_, IR_BUILTIN_FUNC, ty, ty);
    ir_ref sqrt_func = ir_const_func(ctx_, name, proto);
    setRef(op->getResult(0), ir_CALL_1(ty, sqrt_func, operand));
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

void MLIRBridge::emitC(const std::string& funcName, FILE* outFile) {
    ir_build_def_use_lists(ctx_);
    ir_build_cfg(ctx_);
    ir_build_dominators_tree(ctx_);
    ir_find_loops(ctx_);
    ir_gcm(ctx_);
    ir_schedule(ctx_);
    ir_assign_virtual_registers(ctx_);
    ir_compute_live_ranges(ctx_);
    ir_coalesce(ctx_);
    ir_emit_c(ctx_, funcName.c_str(), outFile);
}