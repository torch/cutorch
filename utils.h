#ifndef CUTORCH_UTILS_H
#define CUTORCH_UTILS_H

#include <lua.h>

#ifdef __cplusplus
# define CUTORCH_EXTERNC extern "C"
#else
# define CUTORCH_EXTERNC extern
#endif

#ifdef WIN32
# ifdef CUTORCH_EXPORTS
#  define CUTORCH_API CUTORCH_EXTERNC __declspec(dllexport)
# else
#  define CUTORCH_API CUTORCH_EXTERNC __declspec(dllimport)
# endif
#else
# define CUTORCH_API CUTORCH_EXTERNC
#endif


struct THCudaState;

CUTORCH_API struct THCudaState* getState(lua_State* L);

#endif
