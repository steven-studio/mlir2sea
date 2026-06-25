#include "mlir_bridge.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <iostream>

extern "C" {
#include "ir.h"
#include "ir_builder.h"
}
#define ctx ctx_

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

void MLIRBridge::build(mlir::func::FuncOp func) {
    ir_init(ctx_, IR_FUNCTION, 128, 128);
    ctx_->ret_type = IR_I32;
    ir_START();

    // 建 PARAM nodes
    int pos = 1;
    for (auto arg : func.getArguments()) {
        std::string pname = "p" + std::to_string(pos - 1);
        ir_ref param = ir_PARAM(IR_I32, pname.c_str(), pos++);
        setRef(arg, param);
    }

    // 遍歷 body 裡的 op
    func.getBody().walk([&](mlir::Operation* op) {
        handleOp(op);
    });
}

void MLIRBridge::handleOp(mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "arith.addi")    handleAddi(op);
    else if (name == "arith.subi") handleSubi(op);
    else if (name == "arith.muli") handleMuli(op);
    else if (name == "arith.cmpi") handleCmpi(op);
    else if (name == "arith.constant") handleConstant(op);
    else if (name == "func.return")  handleReturn(op);
    else if (name == "scf.if")   handleIf(op);
    else if (name == "scf.for")  handleFor(op);
}

void MLIRBridge::handleAddi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    setRef(op->getResult(0), ir_ADD_I32(lhs, rhs));
}

void MLIRBridge::handleSubi(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    setRef(op->getResult(0), ir_SUB_I32(lhs, rhs));
}

void MLIRBridge::handleMuli(mlir::Operation* op) {
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    setRef(op->getResult(0), ir_MUL_I32(lhs, rhs));
}

void MLIRBridge::handleCmpi(mlir::Operation* op) {
    auto cmpi = mlir::cast<mlir::arith::CmpIOp>(op);
    ir_ref lhs = getRef(op->getOperand(0));
    ir_ref rhs = getRef(op->getOperand(1));
    ir_ref result;
    switch (cmpi.getPredicate()) {
        case mlir::arith::CmpIPredicate::eq:  result = ir_EQ(lhs, rhs); break;
        case mlir::arith::CmpIPredicate::ne:  result = ir_NE(lhs, rhs); break;
        case mlir::arith::CmpIPredicate::slt: result = ir_LT(lhs, rhs); break;
        case mlir::arith::CmpIPredicate::sle: result = ir_LE(lhs, rhs); break;
        case mlir::arith::CmpIPredicate::sgt: result = ir_GT(lhs, rhs); break;
        case mlir::arith::CmpIPredicate::sge: result = ir_GE(lhs, rhs); break;
        default: result = ir_EQ(lhs, rhs); break;
    }
    setRef(op->getResult(0), result);
}

void MLIRBridge::handleConstant(mlir::Operation* op) {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
    if (!attr) return;
    int32_t val = (int32_t)attr.getInt();
    setRef(op->getResult(0), ir_COPY_I32(ir_CONST_I32(val)));
}

void MLIRBridge::handleReturn(mlir::Operation* op) {
    if (op->getNumOperands() == 0) return;
    ir_ref ret = getRef(op->getOperand(0));
    ir_RETURN(ret);
}

void MLIRBridge::handleIf(mlir::Operation* op) {
    // TODO: scf.if -> IF/MERGE
    std::cerr << "TODO: scf.if\n";
}

void MLIRBridge::handleFor(mlir::Operation* op) {
    // TODO: scf.for -> LOOP_BEGIN/LOOP_END
    std::cerr << "TODO: scf.for\n";
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
