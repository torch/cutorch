#include "torch/utils.h"
#include "THC.h"
#include "THFile.h"
#include "luaT.h"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,CReal,Storage_,NAME)
#define torch_Storage TH_CONCAT_STRING_3(torch.,CReal,Storage)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,CReal,Tensor_,NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,CReal,Tensor)
#define cutorch_Tensor_(NAME) TH_CONCAT_4(cutorch_,CReal,Tensor_,NAME)
#define cutorch_TensorCopy_(NAME) TH_CONCAT_4(cutorch_,Real,TensorCopy_,NAME)

// generate the torch types -- we could also do this via THGenerateAllTypes,
// but this allows us to be self contained.
#define FORCE_TH_HALF
#include "generic/CTensorCopy.c"
#include "THCGenerateAllTypes.h"
#undef FORCE_TH_HALF
#include "generic/CTensor.c"
#include "THCGenerateAllTypes.h"
