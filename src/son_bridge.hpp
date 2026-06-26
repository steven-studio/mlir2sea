#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdio>

extern "C" {
#include "ir.h"
}

void emitSoNFuncToIR(mlir::func::FuncOp func, ir_ctx* ctx, FILE* outFile);
