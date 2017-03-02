#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.c"
#else

#include "THCHalf.h"

static void torch_Tensor_(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         THCStorage **storage_, ptrdiff_t *storageOffset_, THLongStorage **size_, THLongStorage **stride_);

static void torch_Tensor_(c_readSizeStride)(lua_State *L, int index, int allowStride, THLongStorage **size_, THLongStorage **stride_);

static int torch_Tensor_(size)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->size[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->size, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, "torch.LongStorage");
  }
  return 1;
}

static int torch_Tensor_(elementSize)(lua_State *L)
{
  lua_pushnumber(L, THCStorage_(elementSize)(cutorch_getstate(L)));
  return 1;
}

static int torch_Tensor_(stride)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->stride[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->stride, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, "torch.LongStorage");
  }
  return 1;
}

static int torch_Tensor_(nDimension)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushnumber(L, tensor->nDimension);
  return 1;
}

static int torch_Tensor_(storage)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  if(tensor->storage)
  {
    THCStorage_(retain)(cutorch_getstate(L), tensor->storage);
    luaT_pushudata(L, tensor->storage, torch_Storage);
  }
  else
    lua_pushnil(L);

  return 1;
}

static int torch_Tensor_(storageOffset)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushinteger(L, tensor->storageOffset+1);
  return 1;
}

static int torch_Tensor_(new)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor;
  ptrdiff_t storageOffset;
  THLongStorage *size, *stride;

  if(lua_type(L, 1) == LUA_TTABLE)
  {
    ptrdiff_t i, j;
    THLongStorage *counter;
    ptrdiff_t si = 0;
    int dimension = 0;
    int is_finished = 0;

    lua_settop(L, 1);
    size = THLongStorage_new();

    while( (lua_type(L, -1) == LUA_TTABLE) && (lua_objlen(L, -1) > 0) )
    {
      THLongStorage_resize(size, dimension+1);
      size->data[dimension] = lua_objlen(L, -1);
      dimension++;
      lua_rawgeti(L, -1, 1);
    }
    lua_pop(L, 1);

    counter = THLongStorage_newWithSize(size->size);
    THLongStorage_fill(counter, 0);

    tensor = THCTensor_(newWithSize)(state, size, NULL);

    if(size->size == 0)
      is_finished = 1;

    while(!is_finished)
    {
      if(!lua_istable(L, -1))
      {
        THLongStorage_free(size);
        THLongStorage_free(counter);
        THCTensor_(free)(state, tensor);
        luaL_error(L, "invalid tensor definition");
      }

      if(lua_objlen(L, -1) != size->data[size->size-1])
      {
        THLongStorage_free(size);
        THLongStorage_free(counter);
        THCTensor_(free)(state, tensor);
        luaL_error(L, "invalid tensor sizes");
      }

      for(i = 0; i < size->data[size->size-1]; i++)
      {
        lua_rawgeti(L, -1, i+1);
        if(!lua_isnumber(L, -1))
        {
          THLongStorage_free(size);
          THLongStorage_free(counter);
          THCTensor_(free)(state, tensor);
          luaL_error(L, "invalid element (not a number)");
        }

#ifdef THC_REAL_IS_HALF
        half value = THC_float2half((float) lua_tonumber(L, -1));
#else
        real value = (real) lua_tonumber(L, -1);
#endif

        THCStorage_(set)(state, THCTensor_(storage)(state, tensor), si++, value);
        lua_pop(L, 1);
      }

      if(size->size == 1)
        break;

      for(i = size->size-2; i >= 0; i--)
      {
        if(++counter->data[i] == size->data[i])
        {
          if(i == 0)
          {
            is_finished = 1;
            break;
          }
          else
          {
            counter->data[i] = 0;
            lua_pop(L, 1);
          }
        }
        else
        {
          lua_pop(L, 1);
          for(j = i; j < size->size-1; j++)
          {
            if(!lua_istable(L, -1))
            {
              THLongStorage_free(size);
              THLongStorage_free(counter);
              THCTensor_(free)(state, tensor);
              luaL_error(L, "invalid tensor definition");
            }
            if(lua_objlen(L, -1) != size->data[j])
            {
              THLongStorage_free(size);
              THLongStorage_free(counter);
              THCTensor_(free)(state, tensor);
              luaL_error(L, "invalid tensor sizes");
            }
            lua_rawgeti(L, -1, counter->data[j]+1);
          }
          break;
        }
      }
    }

    THLongStorage_free(size);
    THLongStorage_free(counter);
  }
  else
  {
    THCStorage *storage;

    torch_Tensor_(c_readTensorStorageSizeStride)(L, 1, 1, 1, 1, 1,
                                                 &storage, &storageOffset, &size, &stride);

    tensor = THCTensor_(newWithStorage)(state, storage, storageOffset, size, stride);

    THLongStorage_free(size);
    THLongStorage_free(stride);
  }

  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(set)(lua_State *L)
{
  THCTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  THCStorage *storage;
  ptrdiff_t storageOffset;
  THLongStorage *size, *stride;

  torch_Tensor_(c_readTensorStorageSizeStride)(L, 2, 1, 1, 1, 1,
                                               &storage, &storageOffset, &size, &stride);

  THCTensor_(setStorage)(cutorch_getstate(L), self, storage, storageOffset, size, stride);

  THLongStorage_free(size);
  THLongStorage_free(stride);

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(clone)(lua_State *L)
{
  THCTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  self = THCTensor_(newClone)(cutorch_getstate(L), self);
  luaT_pushudata(L, self, torch_Tensor);
  return 1;
}

static int torch_Tensor_(contiguous)(lua_State *L)
{
  THCTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  self = THCTensor_(newContiguous)(cutorch_getstate(L), self);
  luaT_pushudata(L, self, torch_Tensor);
  return 1;
}

/* Resize */
static int torch_Tensor_(resizeAs)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  THCTensor_(resizeAs)(cutorch_getstate(L), tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(resize)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THLongStorage *size, *stride;

  torch_Tensor_(c_readSizeStride)(L, 2, 0, &size, &stride);

  THCTensor_(resize)(cutorch_getstate(L), tensor, size, stride);

  THLongStorage_free(size);
  THLongStorage_free(stride);

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(narrow)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension = luaL_checkint(L, 2)-1;
  long firstIndex = luaL_checklong(L, 3)-1;
  long size = luaL_checklong(L, 4);

/*  THArgCheck( (dimension >= 0) && (dimension < tensor->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < tensor->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= tensor->size[dimension]), 4, "out of range");
*/
  tensor = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(narrow)(state, tensor, NULL, dimension, firstIndex, size);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(sub)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  long d0s = -1, d0e = -1, d1s = -1, d1e = -1, d2s = -1, d2e = -1, d3s = -1, d3e = -1;

  d0s = luaL_checklong(L, 2)-1;
  d0e = luaL_checklong(L, 3)-1;
  if(d0s < 0)
    d0s += tensor->size[0]+1;
  if(d0e < 0)
    d0e += tensor->size[0]+1;
  luaL_argcheck(L, tensor->nDimension > 0, 2, "invalid dimension");
  luaL_argcheck(L, d0s >= 0 && d0s < tensor->size[0], 2, "out of range");
  luaL_argcheck(L, d0e >= 0 && d0e < tensor->size[0], 3, "out of range");
  luaL_argcheck(L, d0e >= d0s, 3, "end smaller than beginning");

  if(!lua_isnone(L, 4))
  {
    d1s = luaL_checklong(L, 4)-1;
    d1e = luaL_checklong(L, 5)-1;
    if(d1s < 0)
      d1s += tensor->size[1]+1;
    if(d1e < 0)
      d1e += tensor->size[1]+1;
    luaL_argcheck(L, tensor->nDimension > 1, 4, "invalid dimension");
    luaL_argcheck(L, d1s >= 0 && d1s < tensor->size[1], 4, "out of range");
    luaL_argcheck(L, d1e >= 0 && d1e < tensor->size[1], 5, "out of range");
    luaL_argcheck(L, d1e >= d1s, 5, "end smaller than beginning");

    if(!lua_isnone(L, 6))
    {
      d2s = luaL_checklong(L, 6)-1;
      d2e = luaL_checklong(L, 7)-1;
      if(d2s < 0)
        d2s += tensor->size[2]+1;
      if(d2e < 0)
        d2e += tensor->size[2]+1;
      luaL_argcheck(L, tensor->nDimension > 2, 6, "invalid dimension");
      luaL_argcheck(L, d2s >= 0 && d2s < tensor->size[2], 6, "out of range");
      luaL_argcheck(L, d2e >= 0 && d2e < tensor->size[2], 7, "out of range");
      luaL_argcheck(L, d2e >= d2s, 7, "end smaller than beginning");

      if(!lua_isnone(L, 8))
      {
        d3s = luaL_checklong(L, 8)-1;
        d3e = luaL_checklong(L, 9)-1;
        if(d3s < 0)
          d3s += tensor->size[3]+1;
        if(d3e < 0)
          d3e += tensor->size[3]+1;
        luaL_argcheck(L, tensor->nDimension > 3, 8, "invalid dimension");
        luaL_argcheck(L, d3s >= 0 && d3s < tensor->size[3], 8, "out of range");
        luaL_argcheck(L, d3e >= 0 && d3e < tensor->size[3], 9, "out of range");
        luaL_argcheck(L, d3e >= d3s, 9, "end smaller than beginning");
      }
    }
  }

  tensor = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(narrow)(state, tensor, NULL, 0, d0s, d0e-d0s+1);
  if(d1s >= 0)
    THCTensor_(narrow)(state, tensor, NULL, 1, d1s, d1e-d1s+1);
  if(d2s >= 0)
    THCTensor_(narrow)(state, tensor, NULL, 2, d2s, d2e-d2s+1);
  if(d3s >= 0)
    THCTensor_(narrow)(state, tensor, NULL, 3, d3s, d3e-d3s+1);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(select)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension = luaL_checkint(L, 2)-1;
  long sliceIndex = luaL_checklong(L, 3)-1;

  if(tensor->nDimension > 1)
  {
    tensor = THCTensor_(newWithTensor)(state, tensor);
    THCTensor_(select)(state, tensor, NULL, dimension, sliceIndex);
    luaT_pushudata(L, tensor, torch_Tensor);
  }
  else
  {
    THArgCheck(tensor->nDimension == 1, 1, "empty Tensor");
    real v = THCTensor_(get1d)(state, tensor, sliceIndex);

#ifdef THC_REAL_IS_HALF
    double value = THC_half2float(v);
#else
    double value = (double) v;
#endif

    lua_pushnumber(L, value);
  }

  return 1;
}

static int torch_Tensor_(indexSelect)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int narg = lua_gettop(L);
  THCTensor *tensor, *src;
  THCudaLongTensor *index;
  THLongTensor *longIndex;
  THCTensor *realIndex;
  int dim;
  if (narg == 3)
  {
    tensor = THCTensor_(new)(state);
    src = luaT_checkudata(L, 1, torch_Tensor);
    dim = luaL_checkint(L, 2) - 1;
    index = luaT_toudata(L, 3, "torch.CudaLongTensor");
    longIndex = luaT_toudata(L, 3, "torch.LongTensor");
    realIndex = luaT_toudata(L, 3, torch_Tensor);
    if (!index && !longIndex && !realIndex) luaT_typerror(L, 3, "CudaLongTensor | LongTensor | Tensor");
    luaT_pushudata(L,tensor,torch_Tensor);
  }
  else if(narg == 4)
  {
    src = luaT_checkudata(L, 2, torch_Tensor);
    dim = luaL_checkint(L, 3) - 1;
    index = luaT_toudata(L, 4, "torch.CudaLongTensor");
    longIndex = luaT_toudata(L, 4, "torch.LongTensor");
    realIndex = luaT_toudata(L, 4, torch_Tensor);
    if (!index && !longIndex && !realIndex) luaT_typerror(L, 3, "CudaLongTensor | LongTensor | Tensor");
    tensor = luaT_checkudata(L,1,torch_Tensor);
  }
  else
  {
    luaL_error(L, "[Tensor,] Tensor, number, Tensor | LongTensor | CudaLongTensor expected");
    return 0;
  }

  if (index)
    THCTensor_(indexSelect)(state, tensor,src,dim,index);
  else if (longIndex)
    THCTensor_(indexSelect_long)(state, tensor,src,dim,longIndex);
  else { // realIndex
#if defined THC_REAL_IS_BYTE || defined THC_REAL_IS_CHAR
    THError("indices are in a ByteTensor or CharTensor which is not allowed. "
            "ByteTensor | CharTensor have a small range of indices to "
            "hold [0 to 255], but Tensor indices might "
            "usually be much larger.");
#endif
    THCState *state = cutorch_getstate(L);
    index = THCudaLongTensor_new(state);
    THLongStorage *indexSize = THCTensor_(newSizeOf)(state, realIndex);
    THCudaLongTensor_resize(state, index, indexSize, NULL);
    THLongStorage_free(indexSize);
    TH_CONCAT_2(THCudaLongTensor_copyCuda, Real)(state, index, realIndex);
    THCTensor_(indexSelect)(state, tensor, src, dim, index);
    THCudaLongTensor_free(state, index);
  }

  return 1;
}

static int torch_Tensor_(indexCopy)(lua_State *L)
{
  int narg = lua_gettop(L);
  THCTensor *tensor, *src;
  THCudaLongTensor *index;
  THLongTensor *longIndex;
  THCTensor *realIndex;
  int dim;
  if(narg == 4)
  {
    dim = luaL_checkint(L, 2) - 1;
    index = luaT_toudata(L, 3, "torch.CudaLongTensor");
    longIndex = luaT_toudata(L, 3, "torch.LongTensor");
    realIndex = luaT_toudata(L, 3, torch_Tensor);
    if (!index && !longIndex && !realIndex) luaT_typerror(L, 3, "CudaLongTensor | LongTensor | Tensor");
    src = luaT_checkudata(L, 4, torch_Tensor);
    tensor = luaT_checkudata(L,1,torch_Tensor);
  }
  else
  {
    luaL_error(L,"Tensor, number, Tensor | LongTensor, Tensor expected");
    return 0;
  }
  THCState *state = cutorch_getstate(L);
  if (index)
    THCTensor_(indexCopy)(state, tensor,dim,index,src);
  else if (longIndex)
    THCTensor_(indexCopy_long)(state, tensor,dim,longIndex,src);
  else { // realIndex
#if defined THC_REAL_IS_BYTE || defined THC_REAL_IS_CHAR
    THError("indices are in a ByteTensor or CharTensor which is not allowed. "
            "ByteTensor | CharTensor have a small range of indices to "
            "hold [0 to 255], but Tensor indices might "
            "usually be much larger.");
#endif
    index = THCudaLongTensor_new(state);
    THLongStorage *indexSize = THCTensor_(newSizeOf)(state, realIndex);
    THCudaLongTensor_resize(state, index, indexSize, NULL);
    THLongStorage_free(indexSize);
    TH_CONCAT_2(THCudaLongTensor_copyCuda, Real)(state, index, realIndex);
    THCTensor_(indexCopy)(state, tensor,dim,index,src);
    THCudaLongTensor_free(state, index);
  }

  return 1;
}

static int torch_Tensor_(indexAdd)(lua_State *L)
{
  int narg = lua_gettop(L);
  THCTensor *tensor, *src;
  THCudaLongTensor *index;
  THLongTensor *longIndex;
  THCTensor *realIndex;
  int dim;
  if(narg == 4)
  {
    dim = luaL_checkint(L, 2) - 1;
    index = luaT_toudata(L, 3, "torch.CudaLongTensor");
    longIndex = luaT_toudata(L, 3, "torch.LongTensor");
    realIndex = luaT_toudata(L, 3, torch_Tensor);
    if (!index && !longIndex && !realIndex) luaT_typerror(L, 3, "CudaLongTensor | LongTensor | Tensor");
    src = luaT_checkudata(L, 4, torch_Tensor);
    tensor = luaT_checkudata(L,1,torch_Tensor);
  }
  else
  {
    luaL_error(L,"Tensor, number, Tensor | LongTensor | CudaLongTensor, Tensor expected");
    return 0;
  }

  if (index)
    THCTensor_(indexAdd)(cutorch_getstate(L), tensor, dim, index, src);
  else if (longIndex)
    THCTensor_(indexAdd_long)(cutorch_getstate(L), tensor,dim,longIndex,src);
  else { // realIndex
#if defined THC_REAL_IS_BYTE || defined THC_REAL_IS_CHAR
    THError("indices are in a ByteTensor or CharTensor which is not allowed. "
            "ByteTensor | CharTensor have a small range of indices to "
            "hold [0 to 255], but Tensor indices might "
            "usually be much larger.");
#endif
    THCState *state = cutorch_getstate(L);
    index = THCudaLongTensor_new(state);
    THLongStorage *indexSize = THCTensor_(newSizeOf)(state, realIndex);
    THCudaLongTensor_resize(state, index, indexSize, NULL);
    THLongStorage_free(indexSize);
    TH_CONCAT_2(THCudaLongTensor_copyCuda, Real)(state, index, realIndex);
    THCTensor_(indexAdd)(state, tensor, dim, index, src);
    THCudaLongTensor_free(state, index);
  }

  return 1;
}

static int torch_Tensor_(indexFill)(lua_State *L)
{
  int narg = lua_gettop(L);
  THCTensor *tensor;
  THCudaLongTensor *index;
  THLongTensor *longIndex;
  THCTensor *realIndex;
  real val;
  int dim;
  if(narg == 4)
  {
    dim = luaL_checkint(L, 2) - 1;
    index = luaT_toudata(L, 3, "torch.CudaLongTensor");
    longIndex = luaT_toudata(L, 3, "torch.LongTensor");
    realIndex = luaT_toudata(L, 3, torch_Tensor);
    if (!index && !longIndex && !realIndex) luaT_typerror(L, 3, "CudaLongTensor | LongTensor | Tensor");
#ifdef THC_REAL_IS_HALF
    val = THC_float2half((float)luaL_checknumber(L, 4));
#else
    val = luaL_checknumber(L, 4);
#endif
    tensor = luaT_checkudata(L,1,torch_Tensor);
  }
  else
  {
    luaL_error(L,"Tensor, number, Tensor | LongTensor | CudaLongTensor, number expected");
    return 0;
  }

  if (index)
    THCTensor_(indexFill)(cutorch_getstate(L), tensor, dim, index, val);
  else if (longIndex)
    THCTensor_(indexFill_long)(cutorch_getstate(L), tensor, dim, longIndex, val);
  else { // realIndex
#if defined THC_REAL_IS_BYTE || defined THC_REAL_IS_CHAR
    THError("indices are in a ByteTensor or CharTensor which is not allowed. "
            "ByteTensor | CharTensor have a small range of indices to "
            "hold [0 to 255], but Tensor indices might "
            "usually be much larger.");
#endif
    THCState *state = cutorch_getstate(L);
    index = THCudaLongTensor_new(state);
    THLongStorage *indexSize = THCTensor_(newSizeOf)(state, realIndex);
    THCudaLongTensor_resize(state, index, indexSize, NULL);
    THLongStorage_free(indexSize);
    TH_CONCAT_2(THCudaLongTensor_copyCuda  , Real)(state, index, realIndex);
    THCTensor_(indexFill)(state, tensor, dim, index, val);
    THCudaLongTensor_free(state, index);
  }

  return 1;
}

static int torch_Tensor_(transpose)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension1 = luaL_checkint(L, 2)-1;
  int dimension2 = luaL_checkint(L, 3)-1;

/*
  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 2, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 3, "out of range");
*/

  tensor = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(transpose)(state, tensor, NULL, dimension1, dimension2);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(t)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);

  luaL_argcheck(L, tensor->nDimension == 2, 1, "Tensor must have 2 dimensions");

  tensor = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(transpose)(state, tensor, NULL, 0, 1);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(unfold)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension = luaL_checkint(L, 2)-1;
  long size = luaL_checklong(L, 3);
  long step = luaL_checklong(L, 4);

/*
  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
*/

  tensor = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(unfold)(state, tensor, NULL, dimension, size, step);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

/* is contiguous? [a bit like in TnXIterator] */
static int torch_Tensor_(isContiguous)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushboolean(L, THCTensor_(isContiguous)(cutorch_getstate(L), tensor));
  return 1;
}

static int torch_Tensor_(isSize)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THLongStorage *size = luaT_checkudata(L, 2, "torch.LongStorage");
  lua_pushboolean(L, THCTensor_(isSize)(cutorch_getstate(L), tensor, size));
  return 1;
}

static int torch_Tensor_(isSetTo)(lua_State *L)
{
  THCTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  lua_pushboolean(L, THCTensor_(isSetTo)(cutorch_getstate(L), self, src));
  return 1;
}

static int torch_Tensor_(isSameSizeAs)(lua_State *L)
{
  THCTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  THCTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  lua_pushboolean(L, THCTensor_(isSameSizeAs)(cutorch_getstate(L), self, src));
  return 1;
}

static int torch_Tensor_(nElement)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushinteger(L, THCTensor_(nElement)(cutorch_getstate(L), tensor));
  return 1;
}

static int torch_Tensor_(copy)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  void *src;
  if( (src = luaT_toudata(L, 2, torch_Tensor)) )
    THCTensor_(copy)(state, tensor, src);
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
  else if( (src = luaT_toudata(L, 2, "torch.HalfTensor")) )
    THCTensor_(copyHalf)(state, tensor, src);
  else
    luaL_typerror(L, 2, "torch.*Tensor");
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(__newindex__)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THLongStorage *idx = NULL;
  THByteTensor *mask;
  THCudaByteTensor *maskCuda;
  THCTensor *maskCudaReal;

  if(lua_isnumber(L, 2))
  {
    void *src;
    long index = luaL_checklong(L,2)-1;
    luaL_argcheck(L, tensor->nDimension > 0, 1, "empty tensor");
    if (index < 0) index = tensor->size[0] + index + 1;

    if (lua_isnumber(L,3)) {
#ifdef THC_REAL_IS_HALF
      half value = THC_float2half(luaL_checknumber(L, 3));
#else
      real value = (real)luaL_checknumber(L,3);
#endif

      if (tensor->nDimension == 1) {
        luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");
        THCStorage_(set)(state, tensor->storage, tensor->storageOffset+index*tensor->stride[0], value);
      } else {
        tensor = THCTensor_(newWithTensor)(state, tensor);
        THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THCTensor_(fill)(state, tensor, value);
        THCTensor_(free)(state, tensor);
      }
    } else if( (src = luaT_toudata(L, 3, torch_Tensor)) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copy)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.ByteTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyByte)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.CharTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyChar)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.ShortTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyShort)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.IntTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyInt)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.LongTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyLong)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.FloatTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyFloat)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.DoubleTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyDouble)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.HalfTensor")) ) {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, tensor, NULL, 0, index, 1);
      THCTensor_(copyHalf)(state, tensor, src);
      THCTensor_(free)(state, tensor);
    } else {
      luaL_typerror(L, 3, "torch.*Tensor");
    }
    lua_pushboolean(L, 1);
  }
  else if((idx = luaT_toudata(L, 2, "torch.LongStorage")))
  {
    ptrdiff_t index = THCTensor_(storageOffset)(state, tensor);

#ifdef THC_REAL_IS_HALF
    real value = THC_float2half((float) luaL_checknumber(L,3));
#else
    real value = (real)luaL_checknumber(L,3);
#endif

    ptrdiff_t dim;

    luaL_argcheck(L, idx->size == tensor->nDimension, 2, "invalid size");

    for(dim = 0; dim < idx->size; dim++)
    {
      long z = idx->data[dim]-1;
      if (z < 0) z = tensor->size[dim] + z + 1;
      luaL_argcheck(L, (z >= 0) && (z < tensor->size[dim]), 2, "index out of bound");
      index += z*tensor->stride[dim];
    }

    THCStorage_(set)(state, tensor->storage, index, value);
    lua_pushboolean(L, 1);
  }
  else if(lua_istable(L, 2))
  {
    int dim;
    int cdim = 0;
    int ndims;
    int done = 0;
    ndims = tensor->nDimension;
    luaL_argcheck(L, lua_objlen(L, 2) <= ndims, 2, "too many indices provided");
    tensor = THCTensor_(newWithTensor)(state, tensor);
    for(dim = 0; dim < ndims; dim++)
    {
      lua_rawgeti(L, 2, dim+1);
      if(lua_isnumber(L, -1))
      {
        long z = lua_tonumber(L, -1)-1;
        lua_pop(L, 1);
        if (z < 0) z = tensor->size[cdim] + z + 1;
        luaL_argcheck(L, (z >= 0) && (z < tensor->size[cdim]), 2, "index out of bound");
        if(tensor->nDimension == 1) {

#ifdef THC_REAL_IS_HALF
          real value = THC_float2half((float) luaL_checknumber(L,3));
#else
          real value = (real) luaL_checknumber(L,3);
#endif
          done = 1;
          THCStorage_(set)(state, tensor->storage, tensor->storageOffset+z*tensor->stride[0], value);
        } else {
          THCTensor_(select)(state, tensor, NULL, cdim, z);
        }
      }
      else if (lua_istable(L, -1))
      {
        long start = 0;
        long end = tensor->size[cdim]-1;
        lua_rawgeti(L, -1, 1);
        if(lua_isnumber(L, -1)) {
          start = lua_tonumber(L, -1)-1;
          end = start;
        }
        lua_pop(L, 1);
        if (start < 0) start = tensor->size[cdim] + start + 1;
        luaL_argcheck(L, (start >= 0) && (start < tensor->size[cdim]), 2, "start index out of bound");

        lua_rawgeti(L, -1, 2);
        if(lua_isnumber(L, -1)) {
          end = lua_tonumber(L, -1)-1;
        }
        lua_pop(L, 2);
        if (end < 0) end = tensor->size[cdim] + end + 1;
        luaL_argcheck(L, (end >= 0) && (end < tensor->size[cdim]), 2, "end index out of bound");

        luaL_argcheck(L, (end >= start), 2, "end index must be greater or equal to start index");

        THCTensor_(narrow)(state, tensor, NULL, cdim++, start, end-start+1);
      }
      else
      {
        break;
      }
    }
    if(!done) {
      /* doing a copy */
      void *src;
      if (lua_isnumber(L,3)) {

#ifdef THC_REAL_IS_HALF
        real value = THC_float2half((float) lua_tonumber(L, 3));
#else
        real value = (real) lua_tonumber(L, 3);
#endif

        THCTensor_(fill)(state, tensor, value);
      } else if( (src = luaT_toudata(L, 3, torch_Tensor)) ) {
        THCTensor_(copy)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.ByteTensor")) ) {
        THCTensor_(copyByte)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.CharTensor")) ) {
        THCTensor_(copyChar)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.ShortTensor")) ) {
        THCTensor_(copyShort)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.IntTensor")) ) {
        THCTensor_(copyInt)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.LongTensor")) ) {
        THCTensor_(copyLong)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.FloatTensor")) ) {
        THCTensor_(copyFloat)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.DoubleTensor")) ) {
        THCTensor_(copyDouble)(state, tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.HalfTensor")) ) {
        THCTensor_(copyHalf)(state, tensor, src);
      } else {
        luaL_typerror(L, 3, "torch.*Tensor");
      }
    }
    THCTensor_(free)(state, tensor);
    lua_pushboolean(L, 1);
  }
  else if((mask = luaT_toudata(L, 2, "torch.ByteTensor")))
  {
    THCTensor *vals;
    if (lua_isnumber(L, 3))
    {
#ifdef THC_REAL_IS_HALF
      real value = THC_float2half((float) luaL_checknumber(L, 3));
#else
      real value = (real) luaL_checknumber(L, 3);
#endif

      THCTensor_(maskedFillByte)(state, tensor, mask, value);
    }
    else if((vals = luaT_toudata(L, 3, torch_Tensor)))
    {
      THCTensor_(maskedCopyByte)(state, tensor, mask, vals);
    }
    else
    {
      luaL_error(L,"number or tensor expected");
    }
  }
  else if((maskCuda = luaT_toudata(L, 2, "torch.CudaByteTensor")))
  {
    THCTensor *vals;
    if (lua_isnumber(L, 3))
    {
#ifdef THC_REAL_IS_HALF
      real value = THC_float2half((float) luaL_checknumber(L, 3));
#else
      real value = (real) luaL_checknumber(L, 3);
#endif

      THCTensor_(maskedFill)(state, tensor, maskCuda, value);
    }
    else if((vals = luaT_toudata(L, 3, torch_Tensor)))
    {
      THCTensor_(maskedCopy)(state, tensor, maskCuda, vals);
    }
    else
    {
      luaL_error(L,"number or tensor expected");
    }

  }
  else if((maskCudaReal = luaT_toudata(L, 2, torch_Tensor)))
  {
    maskCuda = THCudaByteTensor_new(state);
    THLongStorage *maskCudaSize = THCTensor_(newSizeOf)(state, maskCudaReal);
    THCudaByteTensor_resize(state, maskCuda, maskCudaSize, NULL);
    THLongStorage_free(maskCudaSize);
    TH_CONCAT_2(THCudaByteTensor_copyCuda, Real)(state, maskCuda, maskCudaReal);

    THCTensor *vals;
    if (lua_isnumber(L, 3))
    {
#ifdef THC_REAL_IS_HALF
      real value = THC_float2half((float) luaL_checknumber(L, 3));
#else
      real value = (real) luaL_checknumber(L, 3);
#endif

      THCTensor_(maskedFill)(state, tensor, maskCuda, value);
    }
    else if((vals = luaT_toudata(L, 3, torch_Tensor)))
    {
      THCTensor_(maskedCopy)(state, tensor, maskCuda, vals);
    }
    else
    {
      luaL_error(L,"number or tensor expected");
    }

    THCudaByteTensor_free(state, maskCuda);
  }
  else
  {
    lua_pushboolean(L, 0);
  }

  return 1;
}

static int torch_Tensor_(__index__)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THLongStorage *idx = NULL;
  THByteTensor *mask;
  THCudaByteTensor *maskCuda;
  THCTensor *maskCudaReal;

  if(lua_isnumber(L, 2))
  {
    ptrdiff_t index = luaL_checkinteger(L,2)-1;

    luaL_argcheck(L, tensor->nDimension > 0, 1, "empty tensor");
    if (index < 0) index = tensor->size[0] + index + 1;
    luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");

    if(tensor->nDimension == 1)
    {
      real v =
        THCStorage_(get)(state, tensor->storage,
                         tensor->storageOffset+index*tensor->stride[0]);

#ifdef THC_REAL_IS_HALF
      double value = THC_half2float(v);
#else
      double value = (double) v;
#endif

      lua_pushnumber(L, value);
    }
    else
    {
      tensor = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(select)(state, tensor, NULL, 0, index);
      luaT_pushudata(L, tensor, torch_Tensor);
    }
    lua_pushboolean(L, 1);
    return 2;
  }
  else if((idx = luaT_toudata(L, 2, "torch.LongStorage")))
  {
    ptrdiff_t index = THCTensor_(storageOffset)(state, tensor);
    ptrdiff_t dim;

    luaL_argcheck(L, idx->size == tensor->nDimension, 2, "invalid size");

    for(dim = 0; dim < idx->size; dim++)
    {
      long z = idx->data[dim]-1;
      if (z < 0) z = tensor->size[dim] + z + 1;
      luaL_argcheck(L, (z >= 0) && (z < tensor->size[dim]), 2, "index out of bound");
      index += z*tensor->stride[dim];
    }

    real v =
      THCStorage_(get)(state, THCTensor_(storage)(state, tensor), index);

#ifdef THC_REAL_IS_HALF
    double value = (double) THC_half2float(v);
#else
    double value = (double) v;
#endif

    lua_pushnumber(L, value);
    lua_pushboolean(L, 1);
    return 2;
  }
  else if(lua_istable(L, 2))
  {
    int dim;
    int cdim = 0;
    int ndims;
    int done = 0;

    ndims = tensor->nDimension;
    luaL_argcheck(L, lua_objlen(L, 2) <= ndims, 2, "too many indices provided");
    tensor = THCTensor_(newWithTensor)(state, tensor);

    for(dim = 0; dim < ndims; dim++)
    {
      lua_rawgeti(L, 2, dim+1);
      if(lua_isnumber(L, -1))
      {
        long z = lua_tonumber(L, -1)-1;
        lua_pop(L, 1);
        if (z < 0) z = tensor->size[cdim] + z + 1;
        luaL_argcheck(L, (z >= 0) && (z < tensor->size[cdim]), 2, "index out of bound");
        if(tensor->nDimension == 1) {
          done = 1;

          real v =
            THCStorage_(get)(state, tensor->storage,
                             tensor->storageOffset+z*tensor->stride[0]);
#ifdef THC_REAL_IS_HALF
          double value = (double) THC_half2float(v);
#else
          double value = (double) v;
#endif

          lua_pushnumber(L, value);
        } else {
          THCTensor_(select)(state, tensor, NULL, cdim, z);
        }
      }
      else if (lua_istable(L, -1))
      {
        long start = 0;
        long end = tensor->size[cdim]-1;
        lua_rawgeti(L, -1, 1);
        if(lua_isnumber(L, -1)) {
          start = lua_tonumber(L, -1)-1;
          end = start;
        }
        lua_pop(L, 1);
        if (start < 0) start = tensor->size[cdim] + start + 1;
        luaL_argcheck(L, (start >= 0) && (start < tensor->size[cdim]), 2, "start index out of bound");

        lua_rawgeti(L, -1, 2);
        if(lua_isnumber(L, -1)) {
          end = lua_tonumber(L, -1)-1;
        }
        lua_pop(L, 2);
        if (end < 0) end = tensor->size[cdim] + end + 1;
        luaL_argcheck(L, (end >= 0) && (end < tensor->size[cdim]), 2, "end index out of bound");

        luaL_argcheck(L, (end >= start), 2, "end index must be greater or equal to start index");

        THCTensor_(narrow)(state, tensor, NULL, cdim++, start, end-start+1);
      }
      else
      {
        break;
      }
    }
    if(!done) {
      luaT_pushudata(L, tensor, torch_Tensor);
    } else {
      THCTensor_(free)(state, tensor);
    }
    lua_pushboolean(L, 1);
    return 2;
  }
  else if((mask = luaT_toudata(L, 2, "torch.ByteTensor")))
  {
    THCTensor *vals = THCTensor_(new)(state);
    THCTensor_(maskedSelectByte)(state, vals, tensor, mask);
    luaT_pushudata(L, vals, torch_Tensor);
    lua_pushboolean(L, 1);
    return 2;
  }
  else if((maskCuda = luaT_toudata(L, 2, "torch.CudaByteTensor")))
  {
    THCTensor *vals = THCTensor_(new)(state);
    THCTensor_(maskedSelect)(state, vals, tensor, maskCuda);
    luaT_pushudata(L, vals, torch_Tensor);
    lua_pushboolean(L, 1);

    return 2;
  }
  else if((maskCudaReal = luaT_toudata(L, 2, torch_Tensor)))
  {
    maskCuda = THCudaByteTensor_new(state);
    THLongStorage *maskCudaSize = THCTensor_(newSizeOf)(state, maskCudaReal);
    THCudaByteTensor_resize(state, maskCuda, maskCudaSize, NULL);
    THLongStorage_free(maskCudaSize);
    TH_CONCAT_2(THCudaByteTensor_copyCuda, Real)(state, maskCuda, maskCudaReal);

    THCTensor *vals = THCTensor_(new)(state);
    THCTensor_(maskedSelect)(state, vals, tensor, maskCuda);
    luaT_pushudata(L, vals, torch_Tensor);
    lua_pushboolean(L, 1);

    THCudaByteTensor_free(state, maskCuda);

    return 2;
  }
  else
  {
    lua_pushboolean(L, 0);
    return 1;
  }
}

static int torch_Tensor_(retain)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THCTensor_(retain)(cutorch_getstate(L), tensor);
  return 0;
}

static int torch_Tensor_(free)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THCTensor_(free)(cutorch_getstate(L), tensor);
  return 0;
}

/* helpful functions */
static void torch_Tensor_(c_readSizeStride)(lua_State *L, int index, int allowStride, THLongStorage **size_, THLongStorage **stride_)
{
  THLongStorage *size = NULL;
  THLongStorage *stride = NULL;

  if( (size = luaT_toudata(L, index, "torch.LongStorage")) )
  {
    if(!lua_isnoneornil(L, index+1))
    {
      if( (stride = luaT_toudata(L, index+1, "torch.LongStorage")) )
        luaL_argcheck(L, stride->size == size->size, index+1, "provided stride and size are inconsistent");
      else
        luaL_argcheck(L, 0, index+1, "torch.LongStorage expected");
    }
    THLongStorage_retain(size);
    if(stride)
      THLongStorage_retain(stride);
  }
  else
  {
    int i;

    size = THLongStorage_newWithSize(8);
    stride = THLongStorage_newWithSize(8);
    THLongStorage_fill(size, -1);
    THLongStorage_fill(stride, -1);

    if(allowStride)
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+2*i))
          break;
        size->data[i] = luaL_checklong(L, index+2*i);

        if(lua_isnone(L, index+2*i+1))
          break;
        stride->data[i] = luaL_checklong(L, index+2*i+1);
      }
    }
    else
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+i))
          break;
        size->data[i] = luaL_checklong(L, index+i);
      }
    }
  }

  *size_ = size;
  *stride_ = stride;
}

static void torch_Tensor_(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         THCStorage **storage_, ptrdiff_t *storageOffset_, THLongStorage **size_, THLongStorage **stride_)
{
  THCState *state = cutorch_getstate(L);
  THCTensor *src = NULL;
  THCStorage *storage = NULL;

  int arg1Type = lua_type(L, index);

  if( allowNone && (arg1Type == LUA_TNONE) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    *size_ = NULL;
    *stride_ = NULL;
    return;
  }
  else if( allowTensor && (arg1Type == LUA_TUSERDATA) && (src = luaT_toudata(L, index, torch_Tensor)) )
  {
    *storage_ = src->storage;
    *storageOffset_ = src->storageOffset;
    *size_ = THCTensor_(newSizeOf)(state, src);
    *stride_ = THCTensor_(newStrideOf)(state, src);
    return;
  }
  else if( allowStorage && (arg1Type == LUA_TUSERDATA) && (storage = luaT_toudata(L, index, torch_Storage)) )
  {
    *storage_ = storage;
    if(lua_isnone(L, index+1))
    {
      *storageOffset_ = 0;
      *size_ = THLongStorage_newWithSize1(storage->size);
      *stride_ = THLongStorage_newWithSize1(1);
    }
    else
    {
      *storageOffset_ = luaL_checkinteger(L, index+1)-1;
      torch_Tensor_(c_readSizeStride)(L, index+2, allowStride, size_, stride_);
    }
    return;
  }
  else if( (arg1Type == LUA_TNUMBER) || (luaT_toudata(L, index, "torch.LongStorage")) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    torch_Tensor_(c_readSizeStride)(L, index, 0, size_, stride_);

    return;
  }

  *storage_ = NULL;
  *storageOffset_ = 0;

  if(allowTensor && allowStorage)
      luaL_argcheck(L, 0, index, "expecting number or Tensor or Storage");
  else if(allowTensor)
      luaL_argcheck(L, 0, index, "expecting number or Tensor");
  else if(allowStorage)
      luaL_argcheck(L, 0, index, "expecting number or Storage");
  else
      luaL_argcheck(L, 0, index, "expecting number");
}

static int torch_Tensor_(factory)(lua_State *L)
{
  THCTensor *tensor = THCTensor_(new)(cutorch_getstate(L));
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(write)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  THFile_writeIntScalar(file, tensor->nDimension);
  THFile_writeLongRaw(file, tensor->size, tensor->nDimension);
  THFile_writeLongRaw(file, tensor->stride, tensor->nDimension);
#ifdef _MSC_VER
  THAssert(tensor->storageOffset+1 < LONG_MAX);
#endif
  THFile_writeLongScalar(file, tensor->storageOffset+1); /* to respect Lua convention */

  lua_getfield(L, 2, "writeObject"); /* the method */
  lua_pushvalue(L, 2); /* the file */
  /* the storage */
  if(tensor->storage)
  {
    THCStorage_(retain)(cutorch_getstate(L), tensor->storage);
    luaT_pushudata(L, tensor->storage, torch_Storage);
  }
  else
    lua_pushnil(L);

  lua_call(L, 2, 0); /* call the method */

  return 0;
}

static int torch_Tensor_(read)(lua_State *L)
{
  THCTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  tensor->nDimension = THFile_readIntScalar(file);
  tensor->size = THAlloc(sizeof(long)*tensor->nDimension);
  tensor->stride = THAlloc(sizeof(long)*tensor->nDimension);
  THFile_readLongRaw(file, tensor->size, tensor->nDimension);
  THFile_readLongRaw(file, tensor->stride, tensor->nDimension);
  tensor->storageOffset = THFile_readLongScalar(file);
  tensor->storageOffset--;  /* to respect Lua convention */

  lua_getfield(L, 2, "readObject"); /* the method */
  lua_pushvalue(L, 2); /* the file */
  lua_call(L, 1, 1); /* call the method */

  tensor->storage = luaT_toudata(L, -1, torch_Storage);
  if(tensor->storage)
    THCStorage_(retain)(cutorch_getstate(L), tensor->storage);

  return 0;
}

static const struct luaL_Reg torch_Tensor_(_) [] = {
  {"retain", torch_Tensor_(retain)},
  {"free", torch_Tensor_(free)},
  {"contiguous", torch_Tensor_(contiguous)},
  {"size", torch_Tensor_(size)},
  {"elementSize", torch_Tensor_(elementSize)},
  {"__len__", torch_Tensor_(size)},
  {"stride", torch_Tensor_(stride)},
  {"dim", torch_Tensor_(nDimension)},
  {"nDimension", torch_Tensor_(nDimension)},
  {"set", torch_Tensor_(set)},
  {"storage", torch_Tensor_(storage)},
  {"storageOffset", torch_Tensor_(storageOffset)},
  {"clone", torch_Tensor_(clone)},
  {"contiguous", torch_Tensor_(contiguous)},
  {"resizeAs", torch_Tensor_(resizeAs)},
  {"resize", torch_Tensor_(resize)},
  {"narrow", torch_Tensor_(narrow)},
  {"sub", torch_Tensor_(sub)},
  {"select", torch_Tensor_(select)},
  {"index", torch_Tensor_(indexSelect)},
  {"indexCopy", torch_Tensor_(indexCopy)},
  {"indexAdd", torch_Tensor_(indexAdd)},
  {"indexFill", torch_Tensor_(indexFill)},
  {"transpose", torch_Tensor_(transpose)},
  {"t", torch_Tensor_(t)},
  {"unfold", torch_Tensor_(unfold)},
  {"isContiguous", torch_Tensor_(isContiguous)},
  {"isSize", torch_Tensor_(isSize)},
  {"isSetTo", torch_Tensor_(isSetTo)},
  {"isSameSizeAs", torch_Tensor_(isSameSizeAs)},
  {"nElement", torch_Tensor_(nElement)},
  {"copy", torch_Tensor_(copy)},
  {"read", torch_Tensor_(read)},
  {"write", torch_Tensor_(write)},
  {"__index__", torch_Tensor_(__index__)},
  {"__newindex__", torch_Tensor_(__newindex__)},
  {NULL, NULL}
};

void torch_Tensor_(init)(lua_State *L)
{
  luaT_newmetatable(L, torch_Tensor, NULL,
                    torch_Tensor_(new), torch_Tensor_(free), torch_Tensor_(factory));
  luaL_setfuncs(L, torch_Tensor_(_), 0);
  lua_pop(L, 1);
}

#endif
