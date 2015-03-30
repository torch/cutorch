#include "utils.h"
#include "luaT.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"

extern void cutorch_CudaStorage_init(lua_State* L);
extern void cutorch_CudaTensor_init(lua_State* L);
extern void cutorch_CudaTensorMath_init(lua_State* L);
extern void cutorch_CudaTensorOperator_init(lua_State* L);

static int cutorch_synchronize(lua_State *L)
{
  cudaDeviceSynchronize();
  return 0;
}

static int cutorch_getDevice(lua_State *L)
{
  int device;
  THCudaCheck(cudaGetDevice(&device));
  device++;
  lua_pushnumber(L, device);
  return 1;
}

static int cutorch_deviceReset(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCudaCheck(cudaDeviceReset());
  THCudaBlas_reset(state);
  THCRandom_resetGenerator(state);
  return 0;
}

static int cutorch_getDeviceCount(lua_State *L)
{
  int ndevice;
  THCudaCheck(cudaGetDeviceCount(&ndevice));
  lua_pushnumber(L, ndevice);
  return 1;
}

static int cutorch_getMemoryUsage(lua_State *L) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  THCudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes));
  lua_pushnumber(L, freeBytes);
  lua_pushnumber(L, totalBytes);
  return 2;
}

static int cutorch_setDevice(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int device = (int)luaL_checknumber(L, 1)-1;
  THCudaCheck(cudaSetDevice(device));
  THCRandom_setGenerator(state, device);
  THCudaBlas_setHandle(state, device);
  return 0;
}

#define SET_DEVN_PROP(NAME) \
  lua_pushnumber(L, prop.NAME); \
  lua_setfield(L, -2, #NAME);

static int cutorch_getDeviceProperties(lua_State *L)
{
  struct cudaDeviceProp prop;
  int device = (int)luaL_checknumber(L, 1)-1;
  THCudaCheck(cudaGetDeviceProperties(&prop, device));
  lua_newtable(L);
  SET_DEVN_PROP(canMapHostMemory);
  SET_DEVN_PROP(clockRate);
  SET_DEVN_PROP(computeMode);
  SET_DEVN_PROP(deviceOverlap);
  SET_DEVN_PROP(integrated);
  SET_DEVN_PROP(kernelExecTimeoutEnabled);
  SET_DEVN_PROP(major);
  SET_DEVN_PROP(maxThreadsPerBlock);
  SET_DEVN_PROP(memPitch);
  SET_DEVN_PROP(minor);
  SET_DEVN_PROP(multiProcessorCount);
  SET_DEVN_PROP(regsPerBlock);
  SET_DEVN_PROP(sharedMemPerBlock);
  SET_DEVN_PROP(textureAlignment);
  SET_DEVN_PROP(totalConstMem);
  SET_DEVN_PROP(totalGlobalMem);
  SET_DEVN_PROP(warpSize);
  SET_DEVN_PROP(pciBusID);
  SET_DEVN_PROP(pciDeviceID);
  SET_DEVN_PROP(pciDomainID);
  SET_DEVN_PROP(maxTexture1D);
  SET_DEVN_PROP(maxTexture1DLinear);

  size_t freeMem;
  THCudaCheck(cudaMemGetInfo (&freeMem, NULL));
  lua_pushnumber(L, freeMem);
  lua_setfield(L, -2, "freeGlobalMem");

  lua_pushstring(L, prop.name);
  lua_setfield(L, -2, "name");

  return 1;
}

static int cutorch_seed(lua_State *L)
{
  unsigned long seed = THCRandom_seed(cutorch_getstate(L));
  lua_pushnumber(L, seed);
  return 1;
}

static int cutorch_seedAll(lua_State *L)
{
  unsigned long seed = THCRandom_seedAll(cutorch_getstate(L));
  lua_pushnumber(L, seed);
  return 1;
}

static int cutorch_initialSeed(lua_State *L)
{
  unsigned long seed = THCRandom_initialSeed(cutorch_getstate(L));
  lua_pushnumber(L, seed);
  return 1;
}

static int cutorch_manualSeed(lua_State *L)
{
  unsigned long seed = luaL_checknumber(L, 1);
  THCRandom_manualSeed(cutorch_getstate(L), seed);
  return 0;
}

static int cutorch_manualSeedAll(lua_State* L)
{
  unsigned long seed = luaL_checknumber(L, 1);
  THCRandom_manualSeedAll(cutorch_getstate(L), seed);
  return 0;
}

static int cutorch_getRNGState(lua_State *L)
{
  THByteTensor* t = THByteTensor_new();
  THCRandom_getRNGState(cutorch_getstate(L), t);
  luaT_pushudata(L, t, "torch.ByteTensor");
  return 1;
}

static int cutorch_setRNGState(lua_State *L)
{
  THByteTensor* t = luaT_checkudata(L, 1, "torch.ByteTensor");
  THCRandom_setRNGState(cutorch_getstate(L), t);
  return 0;
}

static int cutorch_getState(lua_State *L)
{
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "_state");
  lua_remove(L, -2);
  return 1;
}

static const struct luaL_Reg cutorch_stuff__ [] = {
  {"synchronize", cutorch_synchronize},
  {"getDevice", cutorch_getDevice},
  {"deviceReset", cutorch_deviceReset},
  {"getDeviceCount", cutorch_getDeviceCount},
  {"getDeviceProperties", cutorch_getDeviceProperties},
  {"getMemoryUsage", cutorch_getMemoryUsage},
  {"setDevice", cutorch_setDevice},
  {"seed", cutorch_seed},
  {"seedAll", cutorch_seedAll},
  {"initialSeed", cutorch_initialSeed},
  {"manualSeed", cutorch_manualSeed},
  {"manualSeedAll", cutorch_manualSeedAll},
  {"getRNGState", cutorch_getRNGState},
  {"setRNGState", cutorch_setRNGState},
  {"getState", cutorch_getState},
  {NULL, NULL}
};

LUA_EXTERNC DLL_EXPORT int luaopen_libcutorch(lua_State *L);

int luaopen_libcutorch(lua_State *L)
{
  lua_newtable(L);
  luaL_register(L, NULL, cutorch_stuff__);

  THCState* state = (THCState*)malloc(sizeof(THCState));
  THCudaInit(state);

  cutorch_CudaStorage_init(L);
  cutorch_CudaTensor_init(L);
  cutorch_CudaTensorMath_init(L);
  cutorch_CudaTensorOperator_init(L);

  /* Store state in cutorch table. */
  lua_pushlightuserdata(L, state);
  lua_setfield(L, -2, "_state");

  return 1;
}
