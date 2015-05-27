#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.c"
#else

static int torch_Storage_(new)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THStorage *storage;
  if(lua_type(L, 1) == LUA_TSTRING)
  {
    const char *fileName = luaL_checkstring(L, 1);
    int isShared = luaT_optboolean(L, 2, 0);
    long size = luaL_optlong(L, 3, 0);
    storage = THStorage_(newWithMapping)(state, fileName, size, isShared);
  }
  else if(lua_type(L, 1) == LUA_TTABLE)
  {
    long size = lua_objlen(L, 1);
    long i;
    storage = THStorage_(newWithSize)(state, size);
    for(i = 1; i <= size; i++)
    {
      lua_rawgeti(L, 1, i);
      if(!lua_isnumber(L, -1))
      {
        THStorage_(free)(state, storage);
        luaL_error(L, "element at index %d is not a number", i);
      }
      THStorage_(set)(state, storage, i-1, (real)lua_tonumber(L, -1));
      lua_pop(L, 1);
    }
  }
  else if(lua_type(L, 1) == LUA_TUSERDATA)
  {
    THStorage *src = luaT_checkudata(L, 1, torch_Storage);
    real *ptr = src->data;
    long offset = luaL_optlong(L, 2, 1) - 1;
    if (offset < 0 || offset >= src->size) {
      luaL_error(L, "offset out of bounds");
    }
    long size = luaL_optlong(L, 3, src->size - offset);
    if (size < 1 || size > (src->size - offset)) {
      luaL_error(L, "size out of bounds");
    }
    storage = THStorage_(newWithData)(state, ptr + offset, size);
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_VIEW;
    storage->view = src;
    THStorage_(retain)(state, storage->view);
  }
  else if(lua_type(L, 2) == LUA_TNUMBER)
  {
    long size = luaL_optlong(L, 1, 0);
    real *ptr = (real *)luaL_optlong(L, 2, 0);
    storage = THStorage_(newWithData)(state, ptr, size);
    storage->flag = TH_STORAGE_REFCOUNTED;
  }
  else
  {
    long size = luaL_optlong(L, 1, 0);
    storage = THStorage_(newWithSize)(state, size);
  }
  luaT_pushudata(L, storage, torch_Storage);
  return 1;
}

static int torch_Storage_(free)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THStorage_(free)(cutorch_getstate(L), storage);
  return 0;
}

static int torch_Storage_(resize)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  long size = luaL_checklong(L, 2);
/*  int keepContent = luaT_optboolean(L, 3, 0); */
  THStorage_(resize)(cutorch_getstate(L), storage, size);/*, keepContent); */
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(copy)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  void *src;
  if( (src = luaT_toudata(L, 2, torch_Storage)) )
    THStorage_(copy)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THStorage_(copyByte)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THStorage_(copyChar)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THStorage_(copyShort)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THStorage_(copyInt)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THStorage_(copyLong)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THStorage_(copyFloat)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THStorage_(copyDouble)(state, storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(fill)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  double value = luaL_checknumber(L, 2);
  THStorage_(fill)(cutorch_getstate(L), storage, (real)value);
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(__len__)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  lua_pushnumber(L, storage->size);
  return 1;
}

static int torch_Storage_(__newindex__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
    long index = luaL_checklong(L, 2) - 1;
    double number = luaL_checknumber(L, 3);
    THStorage_(set)(cutorch_getstate(L), storage, index, (real)number);
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
    THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
    long index = luaL_checklong(L, 2) - 1;
    lua_pushnumber(L, THStorage_(get)(cutorch_getstate(L), storage, index));
    lua_pushboolean(L, 1);
    return 2;
  }
  else
  {
    lua_pushboolean(L, 0);
    return 1;
  }
}

#if defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_BYTE)
static int torch_Storage_(string)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  if(lua_isstring(L, -1))
  {
    size_t len = 0;
    const char *str = lua_tolstring(L, -1, &len);
    THStorage_(resize)(cutorch_getstate(L), storage, len);
    memmove(storage->data, str, len);
    lua_settop(L, 1);
  }
  else
    lua_pushlstring(L, (char*)storage->data, storage->size);

  return 1; /* either storage or string */
}
#endif

static int torch_Storage_(totable)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  long i;

  lua_newtable(L);
  for(i = 0; i < storage->size; i++)
  {
    lua_pushnumber(L, (lua_Number)storage->data[i]);
    lua_rawseti(L, -2, i+1);
  }
  return 1;
}

static int torch_Storage_(factory)(lua_State *L)
{
  THStorage *storage = THStorage_(new)(cutorch_getstate(L));
  luaT_pushudata(L, storage, torch_Storage);
  return 1;
}

static int torch_Storage_(write)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  THFile_writeLongScalar(file, storage->size);
  THFile_writeRealRaw(file, storage->data, storage->size);

  return 0;
}

static int torch_Storage_(read)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THFile *file = luaT_checkudata(L, 2, "torch.File");
  long size = THFile_readLongScalar(file);

  THStorage_(resize)(cutorch_getstate(L), storage, size);
  THFile_readRealRaw(file, storage->data, storage->size);

  return 0;
}

static const struct luaL_Reg torch_Storage_(_) [] = {
  {"size", torch_Storage_(__len__)},
  {"__len__", torch_Storage_(__len__)},
  {"__newindex__", torch_Storage_(__newindex__)},
  {"__index__", torch_Storage_(__index__)},
  {"resize", torch_Storage_(resize)},
  {"fill", torch_Storage_(fill)},
  {"copy", torch_Storage_(copy)},
  {"totable", torch_Storage_(totable)},
  {"write", torch_Storage_(write)},
  {"read", torch_Storage_(read)},
#if defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_BYTE)
  {"string", torch_Storage_(string)},
#endif
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
