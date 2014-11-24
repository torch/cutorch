#include "utils.h"

struct THCudaState* getState(lua_State* L)
{
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "getState");
  lua_call(L, 0, 1);
  struct THCudaState *state = lua_touserdata(L, -1);
  lua_pop(L, 2);
  return state;
}
