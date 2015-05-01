#include "torch/utils.h"
#include "luaT.h"
#include "THC.h"

static int cutorch_CudaTensorOperator___add__(lua_State *L)
{
  THCudaTensor *tensor1 = luaT_toudata(L, 1, "torch.CudaTensor");
  THCudaTensor *tensor2 = luaT_toudata(L, 2, "torch.CudaTensor");
  THCudaTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCudaTensor_checkGPU(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THCudaTensor_new(state);
    luaT_pushudata(L, r, "torch.CudaTensor");

    if(!tensor1 && tensor2)
    {
      THCudaTensor_resizeAs(state, r, tensor2);
      THCudaTensor_copy(state, r, tensor2);
      THCudaTensor_add(state, r, r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THCudaTensor_resizeAs(state, r, tensor1);
      THCudaTensor_copy(state, r, tensor1);
      THCudaTensor_add(state, r, r, luaL_checknumber(L, 2));
    }
    else
    {
      THCudaTensor_resizeAs(state, r, tensor1);
      THCudaTensor_copy(state, r, tensor1);
      THCudaTensor_cadd(state, r, r, 1, tensor2);
    }
  }
  return 1;
}

static int cutorch_CudaTensorOperator___sub__(lua_State *L)
{
  THCudaTensor *tensor1 = luaT_toudata(L, 1, "torch.CudaTensor");
  THCudaTensor *tensor2 = luaT_toudata(L, 2, "torch.CudaTensor");
  THCudaTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCudaTensor_checkGPU(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THCudaTensor_new(state);
    luaT_pushudata(L, r, "torch.CudaTensor");

    if(!tensor1 && tensor2)
    {
      THCudaTensor_resizeAs(state, r, tensor2);
      THCudaTensor_fill(state, r, luaL_checknumber(L, 1));
      THCudaTensor_cadd(state, r, r, -1, tensor2);
    }
    else if(tensor1 && !tensor2)
    {
      THCudaTensor_resizeAs(state, r, tensor1);
      THCudaTensor_copy(state, r, tensor1);
      THCudaTensor_add(state, r, r, -luaL_checknumber(L, 2));
    }
    else
    {
      THCudaTensor_resizeAs(state, r, tensor1);
      THCudaTensor_copy(state, r, tensor1);
      THCudaTensor_cadd(state, r, r, -1, tensor2);
    }
  }
  return 1;
}

static int cutorch_CudaTensorOperator___unm__(lua_State *L)
{
  THCudaTensor *tensor = luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCudaTensor_checkGPU(state, 1, tensor));

  r = THCudaTensor_new(state);
  luaT_pushudata(L, r, "torch.CudaTensor");
  THCudaTensor_resizeAs(state, r, tensor);
  THCudaTensor_copy(state, r, tensor);
  THCudaTensor_mul(state, r, r, -1);

  return 1;
}

static int cutorch_CudaTensorOperator___mul__(lua_State *L)
{
  THCudaTensor *tensor1 = luaT_toudata(L, 1, "torch.CudaTensor");
  THCudaTensor *tensor2 = luaT_toudata(L, 2, "torch.CudaTensor");
  THCudaTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCudaTensor_checkGPU(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THCudaTensor_new(state);
    luaT_pushudata(L, r, "torch.CudaTensor");

    if(!tensor1 && tensor2)
    {
      THCudaTensor_resizeAs(state, r, tensor2);
      THCudaTensor_copy(state, r, tensor2);
      THCudaTensor_mul(state, r, r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THCudaTensor_resizeAs(state, r, tensor1);
      THCudaTensor_copy(state, r, tensor1);
      THCudaTensor_mul(state, r, r, luaL_checknumber(L, 2));
    }
    else
    {
      int dimt = tensor1->nDimension;
      int dims = tensor2->nDimension;

      if(dimt == 1 && dims == 1)
        lua_pushnumber(L, THCudaTensor_dot(state, tensor1, tensor2)); /* ok, we wasted r, but who cares */
      else if(dimt == 2 && dims == 1)
      {
        THCudaTensor_resize1d(state, r, tensor1->size[0]);
        THCudaTensor_zero(state, r);
        THCudaTensor_addmv(state, r, 1, r, 1, tensor1, tensor2);
      }
      else if(dimt == 2 && dims == 2)
      {
        THCudaTensor_resize2d(state, r, tensor1->size[0], tensor2->size[1]);
        THCudaTensor_zero(state, r);
        THCudaTensor_addmm(state, r, 1, r, 1, tensor1, tensor2);
      }
      else
        luaL_error(L, "multiplication between %dD and %dD tensors not yet supported", tensor1->nDimension, tensor2->nDimension);
    }
  }
  return 1;
}

static int cutorch_CudaTensorOperator___div__(lua_State *L)
{
  THCudaTensor *tensor = luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *r;
  THCState *state = cutorch_getstate(L);
  THAssert(THCudaTensor_checkGPU(state, 1, tensor));

  luaL_argcheck(L, lua_isnumber(L,2), 2, "number expected");

  r = THCudaTensor_new(state);
  luaT_pushudata(L, r, "torch.CudaTensor");

  THCudaTensor_resizeAs(state, r, tensor);
  THCudaTensor_copy(state, r, tensor);
  THCudaTensor_mul(state, r, r, 1/lua_tonumber(L, 2));

  return 1;
}

static const struct luaL_Reg cutorch_CudaTensorOperator__ [] = {
  {"__add__", cutorch_CudaTensorOperator___add__},
  {"__sub__", cutorch_CudaTensorOperator___sub__},
  {"__unm__", cutorch_CudaTensorOperator___unm__},
  {"__mul__", cutorch_CudaTensorOperator___mul__},
  {"__div__", cutorch_CudaTensorOperator___div__},
  {NULL, NULL}
};

void cutorch_CudaTensorOperator_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaL_setfuncs(L, cutorch_CudaTensorOperator__, 0);
  lua_pop(L, 1);
}
