#include "mlir_bridge.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <iostream>

extern "C" {
#include "ir.h"
#include "ir_builder.h"
int ir_emit_riscv(ir_ctx *ctx, const char *name, FILE *f);
}

// 讓所有 ir_builder macro 能找到 ctx
ir_ctx* g_ctx = nullptr;
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
    else if (name == "arith.subf")   handleSubf(op);
    else if (name == "arith.divsi")  handleDivsi(op);
    else if (name == "arith.remsi")  handleRemsi(op);
    else if (name == "arith.andi")   handleAndi(op);
    else if (name == "arith.ori")    handleOri(op);
    else if (name == "arith.xori")   handleXori(op);
    else if (name == "arith.shli")   handleShli(op);
    else if (name == "arith.shrsi")  handleShrsi(op);
    else if (name == "arith.fptosi") handleFptosi(op);
    else if (name == "arith.uitofp") handleUitofp(op);
    else if (name == "math.sin")     handleMathSin(op);
    else if (name == "math.cos")     handleMathCos(op);
    else if (name == "math.tanh")    handleMathTanh(op);
    else if (name == "math.log")     handleMathLog(op);
    else if (name == "math.log2")    handleMathLog2(op);
    else if (name == "math.pow")     handleMathPow(op);
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
void MLIRBridge::emitRISCV(const std::string& funcName, FILE* outFile) {
    FILE* dbg = fopen("/tmp/dot_ir.txt", "w"); ir_save(ctx_, 0, dbg); fclose(dbg);
    ir_build_def_use_lists(ctx_);
    ir_build_cfg(ctx_);
    ir_build_dominators_tree(ctx_);
    ir_find_loops(ctx_);
    ir_gcm(ctx_);
    ir_schedule(ctx_);
    ir_assign_virtual_registers(ctx_);
    ir_compute_live_ranges(ctx_);
    ir_coalesce(ctx_);
    ir_emit_riscv(ctx_, funcName.c_str(), outFile);
}
