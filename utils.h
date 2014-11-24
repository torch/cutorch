#ifndef CUTORCH_UTILS_H
#define CUTORCH_UTILS_H

#include <lua.h>

struct THCudaState;

struct THCudaState* getState(lua_State* L);

#endif
