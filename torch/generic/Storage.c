#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.c"
#else

static int torch_Storage_(new)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCStorage *storage;
  if(lua_type(L, 1) == LUA_TSTRING)
  {
    const char *fileName = luaL_checkstring(L, 1);
    int isShared = luaT_optboolean(L, 2, 0);
    long size = luaL_optlong(L, 3, 0);
    storage = THCStorage_(newWithMapping)(state, fileName, size, isShared);
  }
  else if(lua_type(L, 1) == LUA_TTABLE)
  {
    long size = lua_objlen(L, 1);
    long i;
    storage = THCStorage_(newWithSize)(state, size);
    for(i = 1; i <= size; i++)
    {
      lua_rawgeti(L, 1, i);
      if(!lua_isnumber(L, -1))
      {
        THCStorage_(free)(state, storage);
        luaL_error(L, "element at index %d is not a number", i);
      }
#ifdef THC_REAL_IS_HALF
      half v = THC_float2half((float) lua_tonumber(L, -1));
      THCStorage_(set)(state, storage, i-1, v);
#else
      THCStorage_(set)(state, storage, i-1, (real)lua_tonumber(L, -1));
#endif
      lua_pop(L, 1);
    }
  }
  else if(lua_type(L, 1) == LUA_TUSERDATA)
  {
    THCStorage *src = luaT_checkudata(L, 1, torch_Storage);
    real *ptr = src->data;
    long offset = luaL_optlong(L, 2, 1) - 1;
    if (offset < 0 || offset >= src->size) {
      luaL_error(L, "offset out of bounds");
    }
    long size = luaL_optlong(L, 3, src->size - offset);
    if (size < 1 || size > (src->size - offset)) {
      luaL_error(L, "size out of bounds");
    }
    storage = THCStorage_(newWithData)(state, ptr + offset, size);
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_VIEW;
    storage->view = src;
    THCStorage_(retain)(state, storage->view);
  }
  else if(lua_type(L, 2) == LUA_TNUMBER)
  {
    long size = luaL_optlong(L, 1, 0);
    real *ptr = (real *)luaL_optlong(L, 2, 0);
    storage = THCStorage_(newWithData)(state, ptr, size);
    storage->flag = TH_STORAGE_REFCOUNTED;
  }
  else
  {
    long size = luaL_optlong(L, 1, 0);
    storage = THCStorage_(newWithSize)(state, size);
  }
  luaT_pushudata(L, storage, torch_Storage);
  return 1;
}

static int torch_Storage_(retain)(lua_State *L)
{
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THCStorage_(retain)(cutorch_getstate(L), storage);
  return 0;
}

static int torch_Storage_(free)(lua_State *L)
{
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THCStorage_(free)(cutorch_getstate(L), storage);
  return 0;
}

static int torch_Storage_(resize)(lua_State *L)
{
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  long size = luaL_checklong(L, 2);
/*  int keepContent = luaT_optboolean(L, 3, 0); */
  THCStorage_(resize)(cutorch_getstate(L), storage, size);/*, keepContent); */
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(copy)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  void *src;
  if( (src = luaT_toudata(L, 2, torch_Storage)) )
    THCStorage_(copy)(state, storage, src);
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

static int torch_Storage_(fill)(lua_State *L)
{
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
#ifdef THC_REAL_IS_HALF
  half value = THC_float2half((float) luaL_checknumber(L, 2));
#else
  real value = (real) luaL_checknumber(L, 2);
#endif
  THCStorage_(fill)(cutorch_getstate(L), storage, value);
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(elementSize)(lua_State *L)
{
  lua_pushnumber(L, THCStorage_(elementSize)(cutorch_getstate(L)));
  return 1;
}

static int torch_Storage_(__len__)(lua_State *L)
{
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  lua_pushnumber(L, storage->size);
  return 1;
}

static int torch_Storage_(__newindex__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
    long index = luaL_checklong(L, 2) - 1;
    double number = luaL_checknumber(L, 3);

#ifdef THC_REAL_IS_HALF
    half value = THC_float2half((float) number);
#else
    real value = (real) number;
#endif
    THCStorage_(set)(cutorch_getstate(L), storage, index, value);
    lua_pushboolean(L, 1);
  }
  else
    lua_pushboolean(L, 0);

  return 1;
}

static int torch_Storage_(__index__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
    long index = luaL_checklong(L, 2) - 1;
    real v = THCStorage_(get)(cutorch_getstate(L), storage, index);

#ifdef THC_REAL_IS_HALF
    double value = THC_half2float(v);
#else
    double value = (double) v;
#endif

    lua_pushnumber(L, value);
    lua_pushboolean(L, 1);
    return 2;
  }
  else
  {
    lua_pushboolean(L, 0);
    return 1;
  }
}

static int torch_Storage_(totable)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  long i;

  /* Copy storage from device to host. */
#ifndef THC_REAL_IS_HALF
  THStorage *host_storage =
      THStorage_(newWithSize)(THCStorage_(size)(state, storage));
  THStorage_(copyCuda)(state, host_storage, storage);
#else
  THFloatStorage *host_storage =
      THFloatStorage_newWithSize(THCStorage_(size)(state, storage));
  THFloatStorage_copyCudaHalf(state, host_storage, storage);
#endif

  lua_newtable(L);
  for(i = 0; i < storage->size; i++)
  {
    lua_pushnumber(L, (lua_Number)host_storage->data[i]);
    lua_rawseti(L, -2, i+1);
  }
#ifndef THC_REAL_IS_HALF
  THStorage_(free)(host_storage);
#else
  THFloatStorage_free(host_storage);
#endif
  return 1;
}

static int torch_Storage_(factory)(lua_State *L)
{
  THCStorage *storage = THCStorage_(new)(cutorch_getstate(L));
  luaT_pushudata(L, storage, torch_Storage);
  return 1;
}

static int torch_Storage_(write)(lua_State *L)
{
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  THFile_writeLongScalar(file, storage->size);
  THFile_writeRealRaw(file, storage->data, storage->size);

  return 0;
}

static int torch_Storage_(read)(lua_State *L)
{
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THFile *file = luaT_checkudata(L, 2, "torch.File");
  long size = THFile_readLongScalar(file);

  THCStorage_(resize)(cutorch_getstate(L), storage, size);
  THFile_readRealRaw(file, storage->data, storage->size);

  return 0;
}

static const struct luaL_Reg torch_Storage_(_) [] = {
  {"retain", torch_Storage_(retain)},
  {"free", torch_Storage_(free)},
  {"size", torch_Storage_(__len__)},
  {"elementSize", torch_Storage_(elementSize)},
  {"__len__", torch_Storage_(__len__)},
  {"__newindex__", torch_Storage_(__newindex__)},
  {"__index__", torch_Storage_(__index__)},
  {"resize", torch_Storage_(resize)},
  {"fill", torch_Storage_(fill)},
  {"copy", torch_Storage_(copy)},
  {"totable", torch_Storage_(totable)},
  {"write", torch_Storage_(write)},
  {"read", torch_Storage_(read)},
  {NULL, NULL}
};

void torch_Storage_(init)(lua_State *L)
{
  luaT_newmetatable(L, torch_Storage, NULL,
                    torch_Storage_(new), torch_Storage_(free), torch_Storage_(factory));
  luaL_setfuncs(L, torch_Storage_(_), 0);
  lua_pop(L, 1);
}

#endif
