#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CTensorCopy.c"
#else

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
  else if( (src = luaT_toudata(L, 2, "torch.HalfTensor")) )
    THTensor_(copyHalf)(tensor, src);
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

void cutorch_TensorCopy_(init)(lua_State* L)
{
  luaT_pushmetatable(L, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  lua_pushcfunction(L, TH_CONCAT_3(cutorch_,Real,Tensor_copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);
}

#endif
