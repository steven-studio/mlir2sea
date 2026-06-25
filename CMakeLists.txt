cmake_minimum_required(VERSION 3.16)
project(mlir2sea)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG HINTS /usr/lib/llvm-18/lib/cmake/mlir)
find_package(LLVM REQUIRED CONFIG HINTS /usr/lib/llvm-18/lib/cmake/llvm)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(mlir2sea src/main.cpp)
target_link_libraries(mlir2sea
    MLIRParser MLIRFuncDialect MLIRArithDialect MLIRSCFDialect
    MLIRIR MLIRSupport LLVMSupport)
