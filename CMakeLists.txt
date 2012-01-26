SET(src Storage.c init.c Tensor.c TensorMathWrap.c)
SET(luasrc init.lua Tensor.lua)

ADD_TORCH_WRAP(cudatensormathwrap TensorMathWrap.lua)

INCLUDE_DIRECTORIES("${Torch_SOURCE_DIR}/pkg/torch")
  
ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(cutorch luaT THC)
