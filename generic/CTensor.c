#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CTensor.c"
#else

#include "THCHalf.h"

/* everything is as the generic Storage.c, except few things (see below) */

#define TH_GENERIC_FILE "generic/Tensor.c"
#include "generic/Tensor.c"
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to CudaTensor */
static int cutorch_Tensor_(copy)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.CudaTensor")) )
    THCTensor_(copyCudaFloat)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaByteTensor")) )
    THCTensor_(copyCudaByte)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaCharTensor")) )
    THCTensor_(copyCudaChar)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaShortTensor")) )
    THCTensor_(copyCudaShort)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaIntTensor")) )
    THCTensor_(copyCudaInt)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaLongTensor")) )
    THCTensor_(copyCudaLong)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaDoubleTensor")) )
    THCTensor_(copyCudaDouble)(state, tensor, src);
#ifdef CUDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.CudaHalfTensor")) )
    THCTensor_(copyCudaHalf)(state, tensor, src);
#endif

  else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )
    THCTensor_(copyByte)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )
    THCTensor_(copyChar)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )
    THCTensor_(copyShort)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )
    THCTensor_(copyInt)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )
    THCTensor_(copyLong)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THCTensor_(copyFloat)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THCTensor_(copyDouble)(state, tensor, src);
  else
    luaL_typerror(L, 2, "torch.*Tensor");

  lua_settop(L, 1);
  return 1;
}

#ifndef THC_REAL_IS_HALF
static int cutorch_Tensor_(copyAsyncCPU)(lua_State *L)
{
#define STRINGIFY_TENSOR(x) TH_CONCAT_STRING_3(torch.,x,Tensor)
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, STRINGIFY_TENSOR(CReal));
  void *src;
  if( (src = luaT_toudata(L, 2, STRINGIFY_TENSOR(CReal))))
    THCTensor_(copy)(state, tensor, src);
  else if( (src = luaT_toudata(L, 2, STRINGIFY_TENSOR(Real))))
    THCTensor_(copyAsyncCPU)(state, tensor, src);
  else
    luaL_typerror(L, 2, STRINGIFY_TENSOR(Real) " or " STRINGIFY_TENSOR(CReal));

  lua_settop(L, 1);
  return 1;
#undef STRINGIFY_TENSOR
}
#endif


#ifndef THC_REAL_IS_HALF
static int TH_CONCAT_3(cutorch_,Real,Tensor_copy)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  void *src;
  if( (src = luaT_toudata(L, 2, TH_CONCAT_STRING_3(torch.,Real,Tensor)) ))
    THTensor_(copy)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )
    THTensor_(copyByte)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )
    THTensor_(copyChar)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )
    THTensor_(copyShort)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )
    THTensor_(copyInt)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )
    THTensor_(copyLong)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THTensor_(copyFloat)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THTensor_(copyDouble)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaByteTensor")) )
    THTensor_(copyCudaByte)(cutorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaCharTensor")) )
    THTensor_(copyCudaChar)(cutorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaShortTensor")) )
    THTensor_(copyCudaShort)(cutorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaIntTensor")) )
    THTensor_(copyCudaInt)(cutorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaLongTensor")) )
    THTensor_(copyCudaLong)(cutorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaTensor")) )
    THTensor_(copyCudaFloat)(cutorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaDoubleTensor")) )
    THTensor_(copyCudaDouble)(cutorch_getstate(L), tensor, src);
#ifdef CUDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.CudaHalfTensor")) )
    THTensor_(copyCudaHalf)(cutorch_getstate(L), tensor, src);
#endif
  else
    luaL_typerror(L, 2, "torch.*Tensor");

  lua_settop(L, 1);
  return 1;
}
#endif

#ifndef THC_REAL_IS_HALF
static int TH_CONCAT_3(cutorch_,Real,Tensor_copyAsyncCuda)(lua_State *L)
{
#define STRINGIFY_TENSOR(x) TH_CONCAT_STRING_3(torch.,x,Tensor)
  THTensor *tensor = luaT_checkudata(L, 1, STRINGIFY_TENSOR(Real));
  void *src;
  if( (src = luaT_toudata(L, 2, STRINGIFY_TENSOR(CReal))))
    THTensor_(copyAsyncCuda)(cutorch_getstate(L), tensor, src);
  else
    luaL_typerror(L, 2, STRINGIFY_TENSOR(CReal));

  lua_settop(L, 1);
  return 1;
#undef STRINGIFY_TENSOR
}
#endif



#ifdef THC_REAL_IS_FLOAT
static void THFloatTensor_computesz(THFloatTensor *self, long **sz_, long **st_)
{
  long *sz, *st, *szh;
  int i;

  sz = (long*)THAlloc(sizeof(long)*self->nDimension);
  st = (long*)THAlloc(sizeof(long)*self->nDimension);
  szh = (long*)THAlloc(sizeof(long)*self->nDimension);

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
#endif

static int cutorch_Tensor_(getDevice)(lua_State *L) {
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushinteger(L, THCTensor_(getDevice)(cutorch_getstate(L), tensor) + 1);
  return 1;
}

void cutorch_Tensor_(init)(lua_State* L)
{
  /* the standard stuff */
  torch_Tensor_(init)(L);

  /* additional methods */
#ifdef THC_REAL_IS_FLOAT
  luaT_pushmetatable(L, "torch.FloatTensor");
  lua_pushcfunction(L, cuda_FloatTensor_fakecopy);
  lua_setfield(L, -2, "fakecopy");
  lua_pop(L, 1);
#endif

#ifndef THC_REAL_IS_HALF
  luaT_pushmetatable(L, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  lua_pushcfunction(L, TH_CONCAT_3(cutorch_,Real,Tensor_copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);

  // Register async copy methods.
  luaT_pushmetatable(L, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  lua_pushcfunction(L, TH_CONCAT_3(cutorch_,Real,Tensor_copyAsyncCuda));
  lua_setfield(L, -2, "copyAsync");
  lua_pop(L, 1);

  luaT_pushmetatable(L, torch_Tensor);
  lua_pushcfunction(L, cutorch_Tensor_(copyAsyncCPU));
  lua_setfield(L, -2, "copyAsync");
  lua_pop(L, 1);
#endif

  luaT_pushmetatable(L, torch_Tensor);
  lua_pushcfunction(L, cutorch_Tensor_(copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);

  luaT_pushmetatable(L, torch_Tensor);
  lua_pushcfunction(L, cutorch_Tensor_(getDevice));
  lua_setfield(L, -2, "getDevice");

  lua_pop(L, 1);
}

#endif
