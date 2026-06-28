#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir_bridge.hpp"
#include "vector_bridge.hpp"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <iostream>
#include <cstdio>

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: mlir2sea <input.mlir> [--emit-riscv]\n"; return 1; }

    bool emitRISCV = false;
    bool emitRVV = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--emit-riscv") emitRISCV = true;
        if (std::string(argv[i]) == "--emit-rvv") emitRVV = true;
    }

    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::scf::SCFDialect>();
    ctx.loadDialect<mlir::math::MathDialect>();
    ctx.loadDialect<mlir::memref::MemRefDialect>();
    ctx.loadDialect<mlir::vector::VectorDialect>();
    ctx.loadDialect<mlir::affine::AffineDialect>();

    llvm::SourceMgr srcMgr;
    auto buf = llvm::MemoryBuffer::getFile(argv[1]);
    if (!buf) { std::cerr << "Cannot open file\n"; return 1; }
    srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &ctx);
    if (!module) { std::cerr << "Parse failed\n"; return 1; }

    const char* outPath = emitRVV ? "/tmp/mlir2sea_rvv.c" : (emitRISCV ? "/tmp/mlir2sea_out.s" : "/tmp/mlir2sea_out.c");
    FILE* outFile = fopen(outPath, "w");
    if (!emitRISCV)
        fprintf(outFile, "#include <stdint.h>\n#include <stdbool.h>\n\n");

    module->walk([&](mlir::func::FuncOp func) {
        std::string name = func.getName().str();
        std::cout << "Translating: " << name << "\n";
        if (emitRVV) {
            VectorBridge vbridge(outFile);
            vbridge.emitFunc(func);
        } else {
            MLIRBridge bridge;
            bridge.build(func);
            if (emitRISCV)
                bridge.emitRISCV(name, outFile);
            else
                bridge.emitC(name, outFile);
        }
        fprintf(outFile, "\n");
    });

    fclose(outFile);
    std::cout << "Output: " << outPath << "\n";
    return 0;
}
