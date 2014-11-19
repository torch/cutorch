#include "utils.h"

struct THCudaState* getState(lua_State* L)
{
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "_state");
  struct THCudaState *state = lua_touserdata(L, -1);
  lua_pop(L, 2);
  return state;
}
