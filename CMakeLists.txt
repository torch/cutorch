CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 5.5 REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration")

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")

SET(src Storage.c init.c Tensor.c TensorMath.c TensorOperator.c torch/utils.c)
SET(luasrc init.lua Tensor.lua FFI.lua test/test.lua)

ADD_TORCH_WRAP(cudatensormathwrap TensorMath.lua)

ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(cutorch luaT THC)
