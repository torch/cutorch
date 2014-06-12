#include "THC.h"
#include "THFile.h"
#include "luaT.h"

/* everything is as the generic Storage.c, except few things (see below) */

static void THCudaTensor_maskedFill(THCudaTensor *tensor, THByteTensor *mask, float value)
{
  THError("not yet implemented for CUDA");
}

static void THCudaTensor_maskedCopy(THCudaTensor *tensor, THByteTensor *mask, THCudaTensor* src)
{
  THError("not yet implemented for CUDA");
}

void THCudaTensor_maskedSelect(THCudaTensor *tensor, THCudaTensor* src, THByteTensor *mask)
{
  THError("not yet implemented for CUDA");
}

/*
 * Based on the implementation of the THTensor_(indexCopy) in torch7
 */
static void THCudaTensor_indexFill(THCudaTensor *tensor, int dim, THLongTensor *index, float val)
{
  long i, numel;
  THCudaTensor *tSlice;
  long *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->nDimension,4,"Indexing dim is out of bounds");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);
  
  for (i=0; i<numel; i++)
  {
    if (tensor->nDimension > 1 )
    {
      // create a new CudaTensor
      tSlice = THCudaTensor_new();
      // set its storage to point to the corresponding storage in tensor
      THCudaTensor_select(tSlice, tensor,dim,index_data[i]-1);
      THCudaTensor_fill(tSlice, val);
      THCudaTensor_free(tSlice);
    }
    else
    {
      THCudaTensor_set1d(tensor,index_data[i]-1,val);
    }
  }
  THLongTensor_free(index);
}

/*
 * Based on the implementation of the THTensor_(indexCopy) in torch7
 */
static void THCudaTensor_indexCopy(THCudaTensor *tensor, int dim, THLongTensor *index, THCudaTensor *src)
{
  long i, numel;
  THCudaTensor *tSlice, *sSlice;
  long *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension,4,"Indexing dim is out of bounds");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->nDimension > 1 )
    {
      tSlice = THCudaTensor_new();
      sSlice = THCudaTensor_new();
      THCudaTensor_select(tSlice, tensor, dim, index_data[i]-1);
      THCudaTensor_select(sSlice, src, dim, i);
      THCudaTensor_copy(tSlice, sSlice);
      THCudaTensor_free(tSlice);
      THCudaTensor_free(sSlice);
    }
    else
    {
      // It's faster to copy a float from an address in the device to another address in the device than 
      // retrieving it to the host memory and recopy it to the device memory
      THCudaCheck(cudaMemcpy(tensor->storage->data + tensor->storageOffset + index_data[i]-1,\
        src->storage->data + src->storageOffset + i, sizeof(float), cudaMemcpyDeviceToDevice));
    }
  }
  THLongTensor_free(index);
}


#define real float
#define Real Cuda

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,Real,Tensor_,NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#define TH_GENERIC_FILE "generic/Tensor.c"
#include "generic/Tensor.c"
#undef TH_GENERIC_FILE

#undef real
#undef Real

/* now we overwrite some methods specific to CudaTensor */

#define CUDA_IMPLEMENT_TENSOR_COPY(TYPEC)                               \
  static int cutorch_##TYPEC##Tensor_copy(lua_State *L)                 \
  {                                                                     \
    TH##TYPEC##Tensor *storage = luaT_checkudata(L, 1, "torch." #TYPEC "Tensor"); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, "torch." #TYPEC "Tensor")) )          \
      TH##TYPEC##Tensor_copy(storage, src);                             \
    else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )           \
      TH##TYPEC##Tensor_copyByte(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )           \
      TH##TYPEC##Tensor_copyChar(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )          \
      TH##TYPEC##Tensor_copyShort(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )            \
      TH##TYPEC##Tensor_copyInt(storage, src);                          \
    else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )           \
      TH##TYPEC##Tensor_copyLong(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )          \
      TH##TYPEC##Tensor_copyFloat(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )         \
      TH##TYPEC##Tensor_copyDouble(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.CudaTensor")) )           \
      TH##TYPEC##Tensor_copyCuda(storage, src);                         \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Tensor");                             \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
  }

CUDA_IMPLEMENT_TENSOR_COPY(Byte)
CUDA_IMPLEMENT_TENSOR_COPY(Char)
CUDA_IMPLEMENT_TENSOR_COPY(Short)
CUDA_IMPLEMENT_TENSOR_COPY(Int)
CUDA_IMPLEMENT_TENSOR_COPY(Long)
CUDA_IMPLEMENT_TENSOR_COPY(Float)
CUDA_IMPLEMENT_TENSOR_COPY(Double)
CUDA_IMPLEMENT_TENSOR_COPY(Cuda)

static void THFloatTensor_computesz(THFloatTensor *self, long **sz_, long **st_)
{
  long *sz, *st, *szh;
  int i;
  
  sz = THAlloc(sizeof(long)*self->nDimension);
  st = THAlloc(sizeof(long)*self->nDimension);
  szh = THAlloc(sizeof(long)*self->nDimension);

  for(i = self->nDimension-1; i >= 0; i--)
  {
    if(i == self->nDimension-1)
      szh[i] = 1;
    else
      szh[i] = szh[i+1]*self->size[i+1];
  }

  memcpy(sz, szh, self->nDimension * sizeof(long));
  memcpy(st, self->stride, self->nDimension * sizeof(long));
  THFree(szh);

  *sz_ = sz;
  *st_ = st;
}

void THFloatTensor_kernel_copy(float *dst, 
                                         long *dst_sz, long *dst_st, int dst_dim,
                                         float *src,
                                         long *src_sz, long *src_st, int src_dim,
                                         long n_elem)
{
  long k;

  for(k = 0; k < n_elem; k++)
  {
    long src_idx = 0;
    long src_rest = k;
    long dst_idx = 0;
    long dst_rest = k;
    int dim;

    for(dim = 0; dim < dst_dim; dim++)
    {
      dst_idx += (dst_rest/dst_sz[dim])*dst_st[dim];
      dst_rest = dst_rest % dst_sz[dim];
    }

    for(dim = 0; dim < src_dim; dim++)
    {
      src_idx += (src_rest/src_sz[dim])*src_st[dim];
      src_rest = src_rest % src_sz[dim];
    }

    dst[dst_idx] = src[src_idx];
  }
}

static int cuda_FloatTensor_fakecopy(lua_State *L)
{
  THFloatTensor *self = luaT_checkudata(L, 1, "torch.FloatTensor");
  THFloatTensor *src = luaT_checkudata(L, 2, "torch.FloatTensor");
  long *d_self_sz, *d_self_st, *d_src_sz, *d_src_st;
  long nElement = THFloatTensor_nElement(self);

  THArgCheck(THFloatTensor_nElement(self) == THFloatTensor_nElement(src), 2, "sizes do not match"); 

  THFloatTensor_computesz(self, &d_self_sz, &d_self_st);
  THFloatTensor_computesz(src, &d_src_sz, &d_src_st);

  THFloatTensor_kernel_copy(THFloatTensor_data(self), 
                            d_self_sz, d_self_st, self->nDimension,
                            THFloatTensor_data(src),
                            d_src_sz, d_src_st, src->nDimension,
                            nElement);
  
  THFree(d_self_sz);
  THFree(d_self_st);
  THFree(d_src_sz);
  THFree(d_src_st);

  lua_settop(L, 1);
  return 1;
}

void cutorch_CudaTensor_init(lua_State* L)
{
  /* the standard stuff */
  torch_CudaTensor_init(L);

  /* additional methods */
  luaT_pushmetatable(L, "torch.FloatTensor");
  lua_pushcfunction(L, cuda_FloatTensor_fakecopy);
  lua_setfield(L, -2, "fakecopy");
  lua_pop(L, 1);

  /* the copy methods */
  {
    int i;

    const void* tnames[8] = {"torch.ByteTensor",
                             "torch.CharTensor",
                             "torch.ShortTensor",
                             "torch.IntTensor",
                             "torch.LongTensor",
                             "torch.FloatTensor",
                             "torch.DoubleTensor",
                             "torch.CudaTensor"};

    static int (*funcs[8])(lua_State*) = {cutorch_ByteTensor_copy,
                                          cutorch_CharTensor_copy,
                                          cutorch_ShortTensor_copy,
                                          cutorch_IntTensor_copy,
                                          cutorch_LongTensor_copy,
                                          cutorch_FloatTensor_copy,
                                          cutorch_DoubleTensor_copy,
                                          cutorch_CudaTensor_copy};

    for(i = 0; i < 8; i++)
    {
      luaT_pushmetatable(L, tnames[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}
