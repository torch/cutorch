#include "THC.h"
#include "THFile.h"
#include "luaT.h"

static const void *torch_File_id = NULL;
static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;
static const void *torch_CudaStorage_id = NULL;

/* everything is as the generic Storage.c, except few things (see below) */

#define real float
#define Real Cuda
#define TH_GENERIC_FILE "generic/Storage.c"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage_id TH_CONCAT_3(torch_,Real,Storage_id)
#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    float *fdata = THAlloc(sizeof(float)*size);                         \
    THFile_readFloatRaw(file, fdata, size);                             \
    THCudaCheck(cudaMemcpy(data, fdata, size * sizeof(float), cudaMemcpyHostToDevice)); \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    float *fdata = THAlloc(sizeof(float)*size);                         \
    THCudaCheck(cudaMemcpy(fdata, data, size * sizeof(float), cudaMemcpyDeviceToHost)); \
    THFile_writeFloatRaw(file, fdata, size);                            \
    THFree(fdata);                                                      \
  }

#define STRING_torchStorage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"

#undef real
#undef Real
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to CudaStorage */

#define CUDA_IMPLEMENT_STORAGE_COPY(TYPEC)                              \
  static int cutorch_##TYPEC##Storage_copy(lua_State *L)                \
  {                                                                     \
    TH##TYPEC##Storage *storage = luaT_checkudata(L, 1, torch_##TYPEC##Storage_id); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, torch_##TYPEC##Storage_id)) )         \
      TH##TYPEC##Storage_copy(storage, src);                            \
    else if( (src = luaT_toudata(L, 2, torch_ByteStorage_id)) )         \
      TH##TYPEC##Storage_copyByte(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, torch_CharStorage_id)) )         \
      TH##TYPEC##Storage_copyChar(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, torch_ShortStorage_id)) )        \
      TH##TYPEC##Storage_copyShort(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, torch_IntStorage_id)) )          \
      TH##TYPEC##Storage_copyInt(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, torch_LongStorage_id)) )         \
      TH##TYPEC##Storage_copyLong(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, torch_FloatStorage_id)) )        \
      TH##TYPEC##Storage_copyFloat(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, torch_DoubleStorage_id)) )       \
      TH##TYPEC##Storage_copyDouble(storage, src);                      \
    else if( (src = luaT_toudata(L, 2, torch_CudaStorage_id)) )         \
      TH##TYPEC##Storage_copyCuda(storage, src);                        \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Storage");                            \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
}

CUDA_IMPLEMENT_STORAGE_COPY(Byte)
CUDA_IMPLEMENT_STORAGE_COPY(Char)
CUDA_IMPLEMENT_STORAGE_COPY(Short)
CUDA_IMPLEMENT_STORAGE_COPY(Int)
CUDA_IMPLEMENT_STORAGE_COPY(Long)
CUDA_IMPLEMENT_STORAGE_COPY(Float)
CUDA_IMPLEMENT_STORAGE_COPY(Double)
CUDA_IMPLEMENT_STORAGE_COPY(Cuda)

void cutorch_CudaStorage_init(lua_State* L)
{
  /* the ids */
  torch_ByteStorage_id = luaT_checktypename2id(L, "torch.ByteStorage");
  torch_CharStorage_id = luaT_checktypename2id(L, "torch.CharStorage");
  torch_ShortStorage_id = luaT_checktypename2id(L, "torch.ShortStorage");
  torch_IntStorage_id = luaT_checktypename2id(L, "torch.IntStorage");
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");
  torch_FloatStorage_id = luaT_checktypename2id(L, "torch.FloatStorage");
  torch_DoubleStorage_id = luaT_checktypename2id(L, "torch.DoubleStorage");
  
  /* the standard stuff */
  torch_CudaStorage_init(L);

  /* the copy methods */
  {
    int i;

    const void* ids[8] = {torch_ByteStorage_id,
                          torch_CharStorage_id,
                          torch_ShortStorage_id,
                          torch_IntStorage_id,
                          torch_LongStorage_id,
                          torch_FloatStorage_id,
                          torch_DoubleStorage_id,
                          torch_CudaStorage_id};
    
    static int (*funcs[8])(lua_State*) = {cutorch_ByteStorage_copy,
                                          cutorch_CharStorage_copy,
                                          cutorch_ShortStorage_copy,
                                          cutorch_IntStorage_copy,
                                          cutorch_LongStorage_copy,
                                          cutorch_FloatStorage_copy,
                                          cutorch_DoubleStorage_copy,
                                          cutorch_CudaStorage_copy};

    for(i = 0; i < 8; i++)
    {
      luaT_pushmetaclass(L, ids[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}
