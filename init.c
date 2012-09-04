#include "luaT.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"

extern void cutorch_CudaStorage_init(lua_State* L);
extern void cutorch_CudaTensor_init(lua_State* L);
extern void cutorch_CudaTensorMath_init(lua_State* L);

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
  THCudaCheck(cudaDeviceReset());
  return 0;
}

static int cutorch_getDeviceCount(lua_State *L)
{
  int ndevice;
  THCudaCheck(cudaGetDeviceCount(&ndevice));
  lua_pushnumber(L, ndevice);
  return 1;
}

static int cutorch_setDevice(lua_State *L)
{
  int device = (int)luaL_checknumber(L, 1)-1;
  THCudaCheck(cudaSetDevice(device));
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
  SET_DEVN_PROP(kernelExecTimeoutEnabled)
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

  lua_pushstring(L, prop.name);
  lua_setfield(L, -2, "name");

  return 1;
}

static const struct luaL_Reg cutorch_stuff__ [] = {
  {"getDevice", cutorch_getDevice},
  {"deviceReset", cutorch_deviceReset},
  {"getDeviceCount", cutorch_getDeviceCount},
  {"getDeviceProperties", cutorch_getDeviceProperties},
  {"setDevice", cutorch_setDevice},
  {NULL, NULL}
};

DLL_EXPORT int luaopen_libcutorch(lua_State *L)
{
  lua_newtable(L);
  luaL_register(L, NULL, cutorch_stuff__);

  THCudaInit();

  THCRandom_initialSeed();

  cutorch_CudaStorage_init(L);
  cutorch_CudaTensor_init(L);
  cutorch_CudaTensorMath_init(L);

  return 1;
}
