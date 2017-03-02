#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CStorageCopy.c"
#else

#include "THHalf.h"

static int TH_CONCAT_3(cutorch_,Real,Storage_copy)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, TH_CONCAT_STRING_3(torch.,Real,Storage));
  void *src;
  if( (src = luaT_toudata(L, 2, TH_CONCAT_STRING_3(torch.,Real,Storage) )))
    THStorage_(copy)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THStorage_(copyByte)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THStorage_(copyChar)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THStorage_(copyShort)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THStorage_(copyInt)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THStorage_(copyLong)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THStorage_(copyFloat)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THStorage_(copyDouble)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.HalfStorage")) )
    THStorage_(copyHalf)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaStorage")) )
    THStorage_(copyCudaFloat)(cutorch_getstate(L), storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaLongStorage")) )
    THStorage_(copyCudaLong)(cutorch_getstate(L), storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaByteStorage")) )
    THStorage_(copyCudaByte)(cutorch_getstate(L), storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaCharStorage")) )
    THStorage_(copyCudaChar)(cutorch_getstate(L), storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaShortStorage")) )
    THStorage_(copyCudaShort)(cutorch_getstate(L), storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaIntStorage")) )
    THStorage_(copyCudaInt)(cutorch_getstate(L), storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaDoubleStorage")) )
    THStorage_(copyCudaDouble)(cutorch_getstate(L), storage, src);
#ifdef CUDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.CudaHalfStorage")) )
    THStorage_(copyCudaHalf)(cutorch_getstate(L), storage, src);
#endif
  else
    luaL_typerror(L, 2, "torch.*Storage");

  lua_settop(L, 1);
  return 1;
}

void cutorch_StorageCopy_(init)(lua_State* L)
{
  // torch_Storage macro is defined in Storage.c produce the CudaTensor types
  // so I have to construct the normal torch types by hand
  luaT_pushmetatable(L, TH_CONCAT_STRING_3(torch.,Real,Storage));
  lua_pushcfunction(L, TH_CONCAT_3(cutorch_,Real,Storage_copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);
}

#endif
