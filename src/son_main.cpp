#include "SoN/IR/SoNDialect.h"
#include "SoN/IR/LowerToSoN.h"
#include "son_bridge.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <iostream>
#include <cstdio>

extern "C" {
#include "ir.h"
}

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: son_mlir2sea <input.mlir>\n"; return 1; }

    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::son::SoNDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();

    llvm::SourceMgr srcMgr;
    auto buf = llvm::MemoryBuffer::getFile(argv[1]);
    if (!buf) { std::cerr << "Cannot open file\n"; return 1; }
    srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());

    auto module = mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &ctx);
    if (!module) { std::cerr << "Parse failed\n"; return 1; }

    mlir::RewritePatternSet patterns(&ctx);
    mlir::son::populateArithToSoNPatterns(patterns);
    if (mlir::applyPatternsAndFoldGreedily(*module, std::move(patterns)).failed()) {
        std::cerr << "Lowering failed\n"; return 1;
    }

    FILE* outFile = fopen("/tmp/son_out.c", "w");
    fprintf(outFile, "#include <stdint.h>\n#include <stdbool.h>\n\n");

    module->walk([&](mlir::func::FuncOp func) {
        std::cout << "Translating: " << func.getName().str() << "\n";
        ir_ctx ir;
        emitSoNFuncToIR(func, &ir, outFile);
        fprintf(outFile, "\n");
        ir_free(&ir);
    });

    fclose(outFile);
    std::cout << "Output: /tmp/son_out.c\n";
    return 0;
}
