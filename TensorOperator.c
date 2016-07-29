#include "torch/utils.h"
#include "luaT.h"
#include "THC.h"

#include "THCTensorMath.h"

#define cutorch_TensorOperator_(NAME) TH_CONCAT_4(cutorch_,CReal,TensorOperator_,NAME)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,CReal,Tensor_,NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,CReal,Tensor)
#define cutorch_Tensor_(NAME) TH_CONCAT_4(cutorch_,CReal,Tensor_,NAME)

#include "generic/TensorOperator.c"
#include "THCGenerateAllTypes.h"
