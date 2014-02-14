#ifndef CUTORCH_UTILS_INC
#define CUTORCH_UTILS_INC

#include "luaT.h"
#include "TH.h"

#ifdef __cplusplus
# define TORCH_EXTERNC extern "C"
#else
# define TORCH_EXTERNC extern
#endif

#ifdef WIN32
# ifdef torch_EXPORTS
#  define TORCH_API TORCH_EXTERNC __declspec(dllexport)
# else
#  define TORCH_API TORCH_EXTERNC __declspec(dllimport)
# endif
#else
# define TORCH_API TORCH_EXTERNC
#endif


TORCH_API THLongStorage* cutorch_checklongargs(lua_State *L, int index);
TORCH_API int cutorch_islongargs(lua_State *L, int index);

#endif
