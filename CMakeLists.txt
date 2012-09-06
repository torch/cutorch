SET(src Storage.c init.c Tensor.c TensorMath.c)
SET(luasrc init.lua Tensor.lua)

ADD_TORCH_WRAP(cudatensormathwrap TensorMath.lua)

INCLUDE_DIRECTORIES("${Torch_SOURCE_PKG}/torch")
  
ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(cutorch luaT THC torch)
