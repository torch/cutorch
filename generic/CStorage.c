#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CStorage.c"
#else

#include "THCHalf.h"

/* everything is as the generic Storage.c, except few things (see below) */

#ifndef THC_REAL_IS_HALF
#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    TH_CONCAT_3(THFile_read,Real,Raw)(file, fdata, size);               \
    THCudaCheck(cudaMemcpy(data, fdata, size * sizeof(real), cudaMemcpyHostToDevice)); \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    THCudaCheck(cudaMemcpy(fdata, data, size * sizeof(real), cudaMemcpyDeviceToHost)); \
    TH_CONCAT_3(THFile_write,Real,Raw)(file, fdata, size);              \
    THFree(fdata);                                                      \
  }
#else
#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    THFile_readCharRaw(file, (char *)fdata, sizeof(real) * size);       \
    THCudaCheck(cudaMemcpy(data, fdata, size * sizeof(real), cudaMemcpyHostToDevice)); \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    THCudaCheck(cudaMemcpy(fdata, data, size * sizeof(real), cudaMemcpyDeviceToHost)); \
    THFile_writeCharRaw(file, (char *)fdata, size * sizeof(real));      \
    THFree(fdata);                                                      \
  }
#endif

#define TH_GENERIC_FILE "generic/Storage.c"
#include "generic/Storage.c"

#undef TH_GENERIC_FILE
#undef THFile_readRealRaw
#undef THFile_writeRealRaw

/* now we overwrite some methods specific to CudaStorage */

static int cutorch_Storage_(copy)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.CudaByteStorage")) )
    THCStorage_(copyCudaByte)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaCharStorage")) )
    THCStorage_(copyCudaChar)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaShortStorage")) )
    THCStorage_(copyCudaShort)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaIntStorage")) )
    THCStorage_(copyCudaInt)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaLongStorage")) )
    THCStorage_(copyCudaLong)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaStorage")) )
    THCStorage_(copyCudaFloat)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaDoubleStorage")) )
    THCStorage_(copyCudaDouble)(state, storage, src);
#ifdef CUDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.CudaHalfStorage")) )
    THCStorage_(copyCudaHalf)(state, storage, src);
#endif

  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THCStorage_(copyByte)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THCStorage_(copyChar)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THCStorage_(copyShort)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THCStorage_(copyInt)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THCStorage_(copyLong)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THCStorage_(copyFloat)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THCStorage_(copyDouble)(state, storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");

  lua_settop(L, 1);
  return 1;
}

#ifndef THC_REAL_IS_HALF
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
#endif

void cutorch_Storage_(init)(lua_State* L)
{
  /* the standard stuff */
  torch_Storage_(init)(L);

  // torch_Storage macro is defined in Storage.c produce the CudaTensor types
  // so I have to construct the normal torch types by hand
#ifndef THC_REAL_IS_HALF
  luaT_pushmetatable(L, TH_CONCAT_STRING_3(torch.,Real,Storage));
  lua_pushcfunction(L, TH_CONCAT_3(cutorch_,Real,Storage_copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);
#endif

  luaT_pushmetatable(L, torch_Storage);
  lua_pushcfunction(L, cutorch_Storage_(copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);
}

#endif
