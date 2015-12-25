#include "torch/utils.h"
#include "THC.h"
#include "THFile.h"
#include "luaT.h"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,CReal,Storage_,NAME)
#define torch_Storage TH_CONCAT_STRING_3(torch.,CReal,Storage)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,CReal,Tensor_,NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,CReal,Tensor)
#define cutorch_Tensor_(NAME) TH_CONCAT_4(cutorch_,CReal,Tensor_,NAME)

#include "generic/CTensor.c"
#include "THCGenerateAllTypes.h"
