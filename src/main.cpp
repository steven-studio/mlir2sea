#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir_bridge.hpp"
#include <iostream>
#include <cstdio>

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: mlir2sea <input.mlir>\n"; return 1; }

    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::scf::SCFDialect>();

    llvm::SourceMgr srcMgr;
    auto buf = llvm::MemoryBuffer::getFile(argv[1]);
    if (!buf) { std::cerr << "Cannot open file\n"; return 1; }
    srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &ctx);
    if (!module) { std::cerr << "Parse failed\n"; return 1; }

    FILE* outFile = fopen("/tmp/mlir2sea_out.c", "w");
    fprintf(outFile, "#include <stdint.h>\n#include <stdbool.h>\n\n");

    module->walk([&](mlir::func::FuncOp func) {
        std::string name = func.getName().str();
        std::cout << "Translating: " << name << "\n";
        MLIRBridge bridge;
        bridge.build(func);
        bridge.emitC(name, outFile);
        fprintf(outFile, "\n");
    });

    fclose(outFile);
    std::cout << "Output: /tmp/mlir2sea_out.c\n";
    return 0;
}
