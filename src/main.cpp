#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <iostream>

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

    module->walk([](mlir::Operation* op) {
        llvm::StringRef name = op->getName().getStringRef();
        std::cout << "op: " << name.str();
        
        // 顯示 SCF op 的對應 Sea-of-Nodes 節點
        if (name == "scf.if")       std::cout << " -> IF/MERGE";
        else if (name == "scf.for") std::cout << " -> LOOP_BEGIN/LOOP_END";
        else if (name == "scf.while") std::cout << " -> LOOP_BEGIN/LOOP_END";
        else if (name == "arith.addi") std::cout << " -> ADD";
        else if (name == "arith.muli") std::cout << " -> MUL";
        else if (name == "arith.cmpi") std::cout << " -> EQ/NE/LT/...";
        else if (name == "func.return") std::cout << " -> RETURN";
        
        std::cout << "\n";
    });
    return 0;
}