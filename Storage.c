#include "torch/utils.h"
#include "THC.h"
#include "THFile.h"
#include "luaT.h"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,CReal,Storage_,NAME)
#define torch_Storage TH_CONCAT_STRING_3(torch.,CReal,Storage)
#define cutorch_Storage_(NAME) TH_CONCAT_4(cutorch_,CReal,Storage_,NAME)
#define cutorch_StorageCopy_(NAME) TH_CONCAT_4(cutorch_,Real,StorageCopy_,NAME)

// generate the torch types -- we could also do this via THGenerateAllTypes,
// but this allows us to be self contained.
#define FORCE_TH_HALF
#include "generic/CStorageCopy.c"
#include "THCGenerateAllTypes.h"
#undef FORCE_TH_HALF
#include "generic/CStorage.c"
#include "THCGenerateAllTypes.h"

