CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 5.5 REQUIRED)

IF(${CUDA_VERSION} LESS 6.0)
  IF(APPLE)
    # work around for mac os x bug:
    # http://stackoverflow.com/questions/16286588/cuda-5-0-cmake-and-make-failing-on-osx-10-8-3
    if (NOT DEFINED CUDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
      set(CUDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by NVCC")
      message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
    endif()
  ENDIF()

  #LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;--use_fast_math")
  # bug on Apple
  IF(APPLE)
    LINK_DIRECTORIES("/usr/local/cuda/lib/")
  ENDIF(APPLE)
ELSE()
  IF(APPLE)
    SET(CUDA_HOST_COMPILER clang)
    LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -stdlib=libstdc++ -Xlinker -stdlib=libstdc++")
    IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    ENDIF()
  ENDIF(APPLE)
ENDIF()

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")

SET(src Storage.c init.c Tensor.c TensorMath.c TensorOperator.c torch/utils.c)
SET(luasrc init.lua Tensor.lua FFI.lua test/test.lua)

ADD_TORCH_WRAP(cudatensormathwrap TensorMath.lua)

ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(cutorch luaT THC)
