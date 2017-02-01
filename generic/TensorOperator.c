#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TensorOperator.c"
#else

static int cutorch_TensorOperator_(__add__)(lua_State *L)
{
  THCTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor);
  THCTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor);
  THCTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCTensor_(checkGPU)(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THCTensor_(new)(state);
    luaT_pushudata(L, r, torch_Tensor);

    if(!tensor1 && tensor2)
    {
      THCTensor_(resizeAs)(state, r, tensor2);
      THCTensor_(copy)(state, r, tensor2);
      double v = luaL_checknumber(L, 1);
#ifdef THC_REAL_IS_HALF
      half value = THC_float2half((float) v);
#else
      real value = (real) v;
#endif

      THCTensor_(add)(state, r, r, value);
    }
    else if(tensor1 && !tensor2)
    {
      THCTensor_(resizeAs)(state, r, tensor1);
      THCTensor_(copy)(state, r, tensor1);

      double v = luaL_checknumber(L, 2);
#ifdef THC_REAL_IS_HALF
      half value = THC_float2half((float) v);
#else
      real value = (real) v;
#endif

      THCTensor_(add)(state, r, r, value);
    }
    else
    {
      THCTensor_(resizeAs)(state, r, tensor1);
      THCTensor_(copy)(state, r, tensor1);

#ifdef THC_REAL_IS_HALF
      half one = THC_float2half(1.0f);
#else
      real one = (real) 1;
#endif

      THCTensor_(cadd)(state, r, r, one, tensor2);
    }
  }
  return 1;
}

static int cutorch_TensorOperator_(__sub__)(lua_State *L)
{
  THCTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor);
  THCTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor);
  THCTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCTensor_(checkGPU)(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THCTensor_(new)(state);
    luaT_pushudata(L, r, torch_Tensor);

#ifdef THC_REAL_IS_HALF
      half neg = THC_float2half(-1.0f);
#else
      real neg = (real) -1;
#endif

    if(!tensor1 && tensor2)
    {
      THCTensor_(resizeAs)(state, r, tensor2);

      double v = luaL_checknumber(L, 1);
#ifdef THC_REAL_IS_HALF
      half value = THC_float2half((float) v);
#else
      real value = (real) v;
#endif

      THCTensor_(fill)(state, r, value);
      THCTensor_(cadd)(state, r, r, neg, tensor2);
    }
    else if(tensor1 && !tensor2)
    {
      THCTensor_(resizeAs)(state, r, tensor1);
      THCTensor_(copy)(state, r, tensor1);

      double v = -luaL_checknumber(L, 2);
#ifdef THC_REAL_IS_HALF
      half value = THC_float2half((float) v);
#else
      real value = (real) v;
#endif

      THCTensor_(add)(state, r, r, value);
    }
    else
    {
      THCTensor_(resizeAs)(state, r, tensor1);
      THCTensor_(copy)(state, r, tensor1);
      THCTensor_(cadd)(state, r, r, neg, tensor2);
    }
  }
  return 1;
}

static int cutorch_TensorOperator_(__unm__)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCTensor_(checkGPU)(state, 1, tensor));

  r = THCTensor_(new)(state);
  luaT_pushudata(L, r, torch_Tensor);
  THCTensor_(resizeAs)(state, r, tensor);
  THCTensor_(copy)(state, r, tensor);

#ifdef THC_REAL_IS_HALF
      half neg = THC_float2half(-1.0f);
#else
      real neg = (real) -1;
#endif

  THCTensor_(mul)(state, r, r, neg);

  return 1;
}

static int cutorch_TensorOperator_(__mul__)(lua_State *L)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor);
  THCTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor);
  THCTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCTensor_(checkGPU)(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THCTensor_(new)(state);
    luaT_pushudata(L, r, torch_Tensor);

    if(!tensor1 && tensor2)
    {
      THCTensor_(resizeAs)(state, r, tensor2);
      THCTensor_(copy)(state, r, tensor2);

      double v = luaL_checknumber(L, 1);
#ifdef THC_REAL_IS_HALF
      half value = THC_float2half((float) v);
#else
      real value = (real) v;
#endif

      THCTensor_(mul)(state, r, r, value);
    }
    else if(tensor1 && !tensor2)
    {
      THCTensor_(resizeAs)(state, r, tensor1);
      THCTensor_(copy)(state, r, tensor1);

      double v = luaL_checknumber(L, 2);
#ifdef THC_REAL_IS_HALF
      half value = THC_float2half((float) v);
#else
      real value = (real) v;
#endif

      THCTensor_(mul)(state, r, r, value);
    }
    else
    {
      int dimt = tensor1->nDimension;
      int dims = tensor2->nDimension;

      if(dimt == 1 && dims == 1)
        lua_pushnumber(L, THCTensor_(dot)(state, tensor1, tensor2)); /* ok, we wasted r, but who cares */
      else if(dimt == 2 && dims == 1)
      {
        THCTensor_(resize1d)(state, r, tensor1->size[0]);
        THCTensor_(zero)(state, r);
        THCTensor_(addmv)(state, r, 1, r, 1, tensor1, tensor2);
      }
      else if(dimt == 2 && dims == 2)
      {
        THCTensor_(resize2d)(state, r, tensor1->size[0], tensor2->size[1]);
        THCTensor_(zero)(state, r);
        THCTensor_(addmm)(state, r, 1, r, 1, tensor1, tensor2);
      }
      else
        luaL_error(L, "multiplication between %dD and %dD tensors not yet supported", tensor1->nDimension, tensor2->nDimension);
    }
  }
#else
  THError("unimplemented");
#endif
  return 1;
}

static int cutorch_TensorOperator_(__div__)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCTensor_(checkGPU)(state, 1, tensor));

  luaL_argcheck(L, lua_isnumber(L,2), 2, "number expected");

  r = THCTensor_(new)(state);
  luaT_pushudata(L, r, torch_Tensor);

  THCTensor_(resizeAs)(state, r, tensor);
  THCTensor_(copy)(state, r, tensor);

  double v = luaL_checknumber(L, 2);
#ifdef THC_REAL_IS_HALF
  half value = THC_float2half(1.0f / (float) v);
#else
  real value = (real) 1 / (real) v;
#endif

  THCTensor_(mul)(state, r, r, value);

  return 1;
}

static const struct luaL_Reg cutorch_TensorOperator_(_) [] = {
  {"__add__", cutorch_TensorOperator_(__add__)},
  {"__sub__", cutorch_TensorOperator_(__sub__)},
  {"__unm__", cutorch_TensorOperator_(__unm__)},
  {"__mul__", cutorch_TensorOperator_(__mul__)},
  {"__div__", cutorch_TensorOperator_(__div__)},
  {NULL, NULL}
};

void cutorch_TensorOperator_(init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_setfuncs(L, cutorch_TensorOperator_(_), 0);
  lua_pop(L, 1);
}

#endif
