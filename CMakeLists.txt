cmake_minimum_required(VERSION 3.16)
project(mlir2sea)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG HINTS /usr/lib/llvm-18/lib/cmake/mlir)
find_package(LLVM REQUIRED CONFIG HINTS /usr/lib/llvm-18/lib/cmake/llvm)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/third_party/dstogov-ir)
add_definitions(${LLVM_DEFINITIONS})

add_executable(mlir2sea src/main.cpp src/mlir_bridge.cpp src/son_bridge.cpp)
target_link_libraries(mlir2sea
    MLIRParser MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRMathDialect MLIRMemRefDialect
    MLIRIR MLIRSupport LLVMSupport
    MLIRSoN
    ${CMAKE_SOURCE_DIR}/third_party/dstogov-ir/libir.a
    m dl pthread)

# son-dialect
add_subdirectory(third_party/son-dialect)
include_directories(third_party/son-dialect/include)
target_include_directories(mlir2sea PRIVATE
    ${CMAKE_BINARY_DIR}/third_party/son-dialect/lib/SoN
)
add_executable(son_mlir2sea src/son_main.cpp src/son_bridge.cpp)
target_link_libraries(son_mlir2sea
    MLIRParser MLIRFuncDialect MLIRArithDialect MLIRMathDialect MLIRTransforms
    MLIRIR MLIRSupport LLVMSupport MLIRSoN
    ${CMAKE_SOURCE_DIR}/third_party/dstogov-ir/libir.a
    m dl pthread)
target_include_directories(son_mlir2sea PRIVATE
    ${CMAKE_BINARY_DIR}/third_party/son-dialect/lib/SoN
    ${CMAKE_SOURCE_DIR}/src)
