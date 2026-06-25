#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: mlir2sea <input.mlir>\n"; return 1; }

    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();

    llvm::SourceMgr srcMgr;
    auto buf = llvm::MemoryBuffer::getFile(argv[1]);
    if (!buf) { std::cerr << "Cannot open file\n"; return 1; }
    srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &ctx);
    if (!module) { std::cerr << "Parse failed\n"; return 1; }

    module->walk([](mlir::Operation* op) {
        std::cout << "op: " << op->getName().getStringRef().str() << "\n";
    });
    return 0;
}
