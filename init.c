#include "utils.h"
#include "luaT.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"
#include "THCHalf.h" // for CUDA_HALF_TENSOR

extern void cutorch_CudaByteStorage_init(lua_State* L);
extern void cutorch_CudaCharStorage_init(lua_State* L);
extern void cutorch_CudaShortStorage_init(lua_State* L);
extern void cutorch_CudaIntStorage_init(lua_State* L);
extern void cutorch_CudaLongStorage_init(lua_State* L);
extern void cutorch_CudaStorage_init(lua_State* L);
extern void cutorch_CudaDoubleStorage_init(lua_State* L);
#ifdef CUDA_HALF_TENSOR
extern void cutorch_CudaHalfStorage_init(lua_State* L);
#endif

extern void cutorch_CudaByteTensor_init(lua_State* L);
extern void cutorch_CudaCharTensor_init(lua_State* L);
extern void cutorch_CudaShortTensor_init(lua_State* L);
extern void cutorch_CudaIntTensor_init(lua_State* L);
extern void cutorch_CudaLongTensor_init(lua_State* L);
extern void cutorch_CudaTensor_init(lua_State* L);
extern void cutorch_CudaDoubleTensor_init(lua_State* L);
#ifdef CUDA_HALF_TENSOR
extern void cutorch_CudaHalfTensor_init(lua_State* L);
#endif

extern void cutorch_CudaTensorOperator_init(lua_State* L);

extern void cutorch_CudaByteTensorMath_init(lua_State* L);
extern void cutorch_CudaCharTensorMath_init(lua_State* L);
extern void cutorch_CudaShortTensorMath_init(lua_State* L);
extern void cutorch_CudaIntTensorMath_init(lua_State* L);
extern void cutorch_CudaLongTensorMath_init(lua_State* L);
extern void cutorch_CudaTensorMath_init(lua_State* L);
extern void cutorch_CudaDoubleTensorMath_init(lua_State* L);
#ifdef CUDA_HALF_TENSOR
extern void cutorch_CudaHalfTensorMath_init(lua_State* L);
#endif


/*
   Iteration utilities for lists of streams and lists of gpus with streams
*/

int checkAndCountListOfStreams(lua_State *L, THCState *state, int arg,
                               int device)
{
  if (!lua_istable(L, arg)) {
    THError("expecting array of device streams");
  }

  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Check that all values in the table are numeric and in bounds */
  int streams = 0;
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    if (!lua_isnumber(L, -2)) {
      THError("expected array of streams, not table");
    }
    if (!lua_isnumber(L, -1)) {
      THError("array of stream ids must contain numeric ids");
    }
    int streamId = (int) lua_tonumber(L, -1);

    /* This will error out if the stream is not in bounds */
    THCState_getDeviceStream(state, device, streamId);

    ++streams;
    lua_pop(L, 1);
  }

  /* Pop table from top */
  lua_pop(L, 1);
  return streams;
}

void checkAndCountListOfGPUStreamPairs(lua_State *L, THCState *state, int arg,
                                       int* gpus,
                                       int* streams)
{
  if (!lua_istable(L, arg)) {
    THError("expecting table of gpu={streams...}");
  }

  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Check that all values in the table are tables of numeric and in bounds */
  *gpus = 0;
  *streams = 0;

  lua_pushnil(L);
  while (lua_next(L, -2)) {
    /* -2 is key (device), -1 is value, in the form device={streams...} */
    if (!lua_isnumber(L, -2) || !lua_istable(L, -1)) {
      THError("expecting table of gpu={streams...}");
    }

    int device = (int) lua_tonumber(L, -2) - 1;
    /* Verify device is in range */
    if (device < 0 || device >= THCState_getNumDevices(state)) {
      THError("%d is not a device", device + 1);
    }

    /* Verify that the list is a list of streams */
    *streams += checkAndCountListOfStreams(L, state, -1, device);
    ++(*gpus);
    lua_pop(L, 1);
  }

  /* Pop table from top */
  lua_pop(L, 1);
}

int createSingleDeviceEvents(lua_State *L, THCState *state, int arg,
                             int device, cudaEvent_t* event)
{

  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Record events */
  lua_pushnil(L);
  int i = 0;
  while (lua_next(L, -2)) {
    int streamId = (int) lua_tonumber(L, -1);
    cudaStream_t streamWaitingOn =
      THCState_getDeviceStream(state, device, streamId);
    THCudaCheck(cudaEventCreateWithFlags(&event[i], cudaEventDisableTiming));
    THCudaCheck(cudaEventRecord(event[i], streamWaitingOn));
    lua_pop(L, 1);
    i++;
  }
  /* Pop table from top */
  lua_pop(L, 1);
  return i;
}

void createMultiDeviceEvents(lua_State *L, THCState *state, int arg,
                             cudaEvent_t* events)
{
  /* Push {gpu={streams...}} table */
  lua_pushvalue(L, arg);

  /* Create and record events per each GPU */
  int gpu = 0;
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    int device = (int) lua_tonumber(L, -2) - 1;
    THCudaCheck(cudaSetDevice(device));
    events += createSingleDeviceEvents(L, state, -1, device, events);
    ++gpu;

    lua_pop(L, 1);
  }

  /* Pop {gpu={streams...}} table */
  lua_pop(L, 1);
}

void waitSingleDeviceEvents(lua_State *L, THCState *state, int arg,
                           int device, cudaEvent_t * event, int numEvents)
{
  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Then, wait on the events. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    int streamId = (int) lua_tonumber(L, -1);
    cudaStream_t stream =
      THCState_getDeviceStream(state, device, streamId);
    for (int i = 0; i < numEvents; i++) {
      THCudaCheck(cudaStreamWaitEvent(stream, event[i], 0));
    }
    lua_pop(L, 1);
  }

  /* Pop table from top */
  lua_pop(L, 1);
}


void waitMultiDeviceEvents(lua_State *L, THCState *state, int arg,
                           cudaEvent_t* events, int streams)
{
  /* Push {gpu={streams...}} table */
  lua_pushvalue(L, arg);

  /* Then, wait on the events. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    int device = (int) lua_tonumber(L, -2) - 1;
    THCudaCheck(cudaSetDevice(device));

    /* Push stream table */
    lua_pushvalue(L, -1);
    lua_pushnil(L);
    while (lua_next(L, -2)) {
      int streamId = (int) lua_tonumber(L, -1);

      cudaStream_t stream =
        THCState_getDeviceStream(state, device, streamId);

      /* Each stream waits on all events */
      for (int i = 0; i < streams; ++i) {
        THCudaCheck(cudaStreamWaitEvent(stream, events[i], 0));
      }

      lua_pop(L, 1);
    }

    /* Pop stream table and GPU entry */
    lua_pop(L, 2);
  }

  /* Pop {gpu={streams...}} table */
  lua_pop(L, 1);
}

/* Synchronizes the host with respect to the current device */
static int cutorch_synchronize(lua_State *L)
{
  THCudaCheck(cudaDeviceSynchronize());
  return 0;
}

/* Synchronizes the host with respect to all devices */
static int cutorch_synchronizeAll(lua_State *L)
{
  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  int devices = -1;
  THCudaCheck(cudaGetDeviceCount(&devices));

  for (int i = 0; i < devices; ++i) {
    THCudaCheck(cudaSetDevice(i));
    THCudaCheck(cudaDeviceSynchronize());
  }

  THCudaCheck(cudaSetDevice(prevDev));

  return 0;
}

/*
   Usage:
   cutorch.reserveStreams(n)
   Allocates n user streams for every device present. If fewer than
   n streams are currently allocated, an additional number will be added.
   If more than n streams are currently allocated, does nothing.
   The default CUDA stream is assumed to be stream 0 and is always present;
   the allocated streams are user streams on top of the CUDA streams
   (thus, reserveStreams(1) will create 1 user stream with two being available,
   the default stream 0 and the user stream 1, on each device).
*/
static int cutorch_reserveStreams(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int numStreams = (int) luaL_checknumber(L, 1);
  int nonBlocking = lua_toboolean(L, 2);
  THCState_reserveStreams(state, numStreams, nonBlocking);

  return 0;
}

/*
   Usage:
   cutorch.reserveBlasHandles(n)
   Allocates n blasHandles for every device present. If fewer than
   n blasHandles are currently allocated, an additional number will be added.
   If more than n blasHandles are currently allocated, does nothing.
   Unlike for streams, there is no default blasHandle.
*/
static int cutorch_reserveBlasHandles(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int numHandles = (int) luaL_checknumber(L, 1);
  THCState_reserveBlasHandles(state, numHandles);

  return 0;
}

/*
   Usage:
   n = cutorch.getNumStreams()
   Returns the number of user streams allocated for every device present.
   By default, is 0.
*/
static int cutorch_getNumStreams(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  lua_pushnumber(L, THCState_getNumStreams(state));

  return 1;
}

/*
   Usage:
   n = cutorch.getNumBlasHandles()
   Returns the number of user blasHandles allocated for every device present.
   By default, is 1.
*/
static int cutorch_getNumBlasHandles(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  lua_pushnumber(L, THCState_getNumBlasHandles(state));

  return 1;
}

/*
   Usage:
   cutorch.setStream(n)
   For all devices, sets the current user stream in use to the index
   specified. e.g.,
   ---
   cutorch.setDevice(1)
   cutorch.setStream(3)
   -- device 1 stream 3 in use here
   cutorch.setDevice(2)
   -- device 2 stream 3 in use here
   ---
   0 is the default stream on the device.
*/
static int cutorch_setStream(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int stream = (int) luaL_checknumber(L, 1);
  THCState_setStreamForCurrentDevice(state, stream);

  return 0;
}

/*
   Usage:
   cutorch.setBlasHandle(n)
   For all devices, sets the current blasHandle in use to the index
   specified. e.g.,
   ---
   cutorch.setDevice(1)
   cutorch.setBlasHandle(3)
   -- device 1 blasHandle 3 in use here
   cutorch.setDevice(2)
   -- device 2 blasHandle 3 in use here
   ---
*/
static int cutorch_setBlasHandle(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int handle = (int) luaL_checknumber(L, 1);
  THCState_setBlasHandleForCurrentDevice(state, handle);

  return 0;
}

/*
   Usage:
   n = cutorch.getStream()
   Returns the current user stream for all devices in use (as previously
   set via cutorch.setStream(n). 0 is the default stream on the device
   and is its initial value.
*/
static int cutorch_getStream(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  lua_pushnumber(L, THCState_getCurrentStreamIndex(state));

  return 1;
}

/*
   Usage:
   n = cutorch.getBlasHandle()
   Returns the current blasHandle for all devices in use (as previously
   set via cutorch.setBlasHandle(n).
*/
static int cutorch_getBlasHandle(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  lua_pushnumber(L, THCState_getCurrentBlasHandleIndex(state));

  return 1;
}

/*
   Usage:
   cutorch.setDefaultStream()
   Equivalent to cutorch.setStream(0).
*/
static int cutorch_setDefaultStream(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCState_setStreamForCurrentDevice(state, 0);


  return 0;
}

/*
   Usage:
   cutorch.streamWaitFor(waiterStream, {waitForStream1, ..., waitForStreamN})
   for streams on the current device. Creates a one-way barrier where
   waiterStream waits for waitForStream1-N to reach the current point.
*/
static int cutorch_streamWaitFor(lua_State *L)
{
  THCState *state = cutorch_getstate(L);

  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));

  /* Check that the waiting stream is in bounds; this will error out if not */
  int waitingId = (int) luaL_checknumber(L, 1);
  cudaStream_t streamWaiting =
    THCState_getDeviceStream(state, curDev, waitingId);

  /* Validate the streams that we are waiting on */
  int streams = checkAndCountListOfStreams(L, state, 2, curDev);

  if (streams < 1) {
    /* nothing to synchronize */
    return 0;
  }
  /* One-way dependency; streamWaiting will wait for the list of streams to
     wait on to complete execution of pending scheduled kernels/events */
  cudaEvent_t * events = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * streams);
  createSingleDeviceEvents(L, state, 2, curDev, events);
  /* Then, wait on them */
  for (int i = 0; i < streams; i++) {
    THCudaCheck(cudaStreamWaitEvent(streamWaiting, events[i], 0));
    THCudaCheck(cudaEventDestroy(events[i]));
  }
  free(events);
  return 0;
}

/*
   Usage:
   cutorch.streamWaitForMultiDevice(gpuWaiter, streamWaiter,
                                    {[gpu1]={stream1_1, ..., stream1_N},
                                    [gpuK]={streamK_1, ..., streamK_M}})
   with a specified GPU per each list of streams.
   Stream (gpuWaiter, streamWaiter) will wait on all of the other streams
   (gpu1, stream1_1), ..., (gpu1, stream1_N), ...,
   (gpuK, streamK_1), ..., (gpuK, streamK_M) to complete fully, as a one-way
   barrier only (only streamWaiter is blocked).
   The streams to wait on are bucketed per device. Equivalent to
   streamWaitFor() if only one GPU's streams are listed.
*/
static int cutorch_streamWaitForMultiDevice(lua_State *L)
{
  THCState *state = cutorch_getstate(L);

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  /* Validate waiting (gpu, stream); this will error out if not */
  int gpuWaiter = (int) luaL_checknumber(L, 1) - 1;
  int streamWaiter = (int) luaL_checknumber(L, 2);
  cudaStream_t streamWaiting =
    THCState_getDeviceStream(state, gpuWaiter, streamWaiter);

  /* Validate and count set of {gpu={streams...}} we are waiting on */
  int gpus = 0;
  int streams = 0;
  checkAndCountListOfGPUStreamPairs(L, state, 3, &gpus, &streams);

  if (streams < 1) {
    /* nothing to synchronize together */
    return 0;
  }

  /*
     Events can only be recorded on the same device on which they are created.
     -For each GPU, create and record event per each stream given
     for that GPU.
     -For (gpuWaiter, streamWaiter), wait on all of the above events.
  */
  cudaEvent_t* events = (cudaEvent_t*) malloc(sizeof(cudaEvent_t) * streams);

  /* First, create an event per GPU and record events for the specified stream
     on that GPU */
  createMultiDeviceEvents(L, state, 3, events);

  /* Then, wait on the events */
  THCudaCheck(cudaSetDevice(gpuWaiter));
  for (int i = 0; i < streams; ++i) {
    THCudaCheck(cudaStreamWaitEvent(streamWaiting, events[i], 0));
  }

  /* Clean up events */
  for (int i = 0; i < streams; ++i) {
    THCudaCheck(cudaEventDestroy(events[i]));
  }
  free(events);
  THCudaCheck(cudaSetDevice(prevDev));

  return 0;
}

/*
   Usage:
   cutorch.streamBarrier({stream1, stream2, ..., streamN})
   applies to streams for the current device. Creates a N-way barrier
   to synchronize all of the streams given
*/
static int cutorch_streamBarrier(lua_State *L)
{
  THCState *state = cutorch_getstate(L);

  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));

  int streams = checkAndCountListOfStreams(L, state, 1, curDev);

  if (streams < 2) {
    /* nothing to synchronize together */
    return 0;
  }
  /* Multi-way dependency (barrier); all streams must complete execution
     of pending scheduled kernels/events */
  cudaEvent_t * events = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * streams);
  /* First, create an event and record them for all streams */
  int eventsCreated =  createSingleDeviceEvents(L, state, 1, curDev, events);

  /* Then, wait on the event. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  waitSingleDeviceEvents(L, state, 1, curDev, events, eventsCreated);
  for (int i = 0; i < eventsCreated; i++)
    THCudaCheck(cudaEventDestroy(events[i]));

  free(events);
  return 0;
}

/* usage:
   cutorch.streamBarrierMultiDevice({[gpu1]={stream1_1, ..., stream1_N},
                                     [gpuK]={streamK_1, ..., streamK_M}})
   with a specified GPU per each list of streams.
   Each stream (gpu1, stream1_1), ..., (gpu1, stream1_N), ...,
               (gpuK, streamK_1), ..., (gpuK, streamK_M) will wait
   for all others to complete fully.
   Streams are bucketed per device. Equivalent to streamBarrier() if only
   one GPU is specified.
 */
static int cutorch_streamBarrierMultiDevice(lua_State *L)
{
  THCState *state = cutorch_getstate(L);

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  /* Validate and count set of {gpu={streams...}} that are mutually waiting */
  int gpus = 0;
  int streams = 0;
  checkAndCountListOfGPUStreamPairs(L, state, 1, &gpus, &streams);

  if (streams < 2) {
    /* nothing to synchronize together */
    return 0;
  }

  /*
     Events can only be recorded on the same device on which they are created.
     -For each GPU, create an event, and record that event on each stream given
     for that GPU.
     -For each GPU, for each stream, wait on the event created by each other
     GPU.
  */
  cudaEvent_t* events = (cudaEvent_t*) malloc(sizeof(cudaEvent_t) * streams);

  /* First, create an event per GPU and record events for the specified stream
     on that GPU */
  createMultiDeviceEvents(L, state, 1, events);

  /* Then, wait on the events. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  waitMultiDeviceEvents(L, state, 1, events, streams);

  /* Clean up events */
  for (int i = 0; i < streams; ++i) {
    THCudaCheck(cudaEventDestroy(events[i]));
  }
  free(events);
  THCudaCheck(cudaSetDevice(prevDev));

  return 0;
}

/*
   Usage:
   cutorch.streamSynchronize(n)
   For the current device, synchronizes with the given stream only
   (cudaStreamSynchronize).
   0 is the default stream on the device.
*/
static int cutorch_streamSynchronize(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int streamId = (int) luaL_checknumber(L, 1);

  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));

  /* This also validates the stream */
  cudaStream_t stream = THCState_getDeviceStream(state, curDev, streamId);
  THCudaCheck(cudaStreamSynchronize(stream));

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
  printf("WARNING: cutorch.deviceReset has been depreceated."
	 " Just remove the call from your code.\n");
  return 0;
}

static int cutorch_getDeviceCount(lua_State *L)
{
  int ndevice;
  THCudaCheck(cudaGetDeviceCount(&ndevice));
  lua_pushnumber(L, ndevice);
  return 1;
}

static int cutorch_getPeerToPeerAccess(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int dev = (int) luaL_checknumber(L, 1) - 1;
  int devToAccess = (int) luaL_checknumber(L, 2) - 1;

  /* device bounds checking is performed within */
  int enabled = THCState_getPeerToPeerAccess(state, dev, devToAccess);
  lua_pushboolean(L, enabled);

  return 1;
}

static int cutorch_setPeerToPeerAccess(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int dev = (int) luaL_checknumber(L, 1) - 1;
  int devToAccess = (int) luaL_checknumber(L, 2) - 1;
  int enable = lua_toboolean(L, 3);

  /* device bounds checking is performed within */
  THCState_setPeerToPeerAccess(state, dev, devToAccess, enable);

  return 0;
}

static int cutorch_getKernelPeerToPeerAccess(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  lua_pushboolean(L, THCState_getKernelPeerToPeerAccessEnabled(state));

  return 1;
}

static int cutorch_setKernelPeerToPeerAccess(lua_State *L)
{
  THCState *state = cutorch_getstate(L);

  int val = lua_toboolean(L, -1);
  THCState_setKernelPeerToPeerAccessEnabled(state, val);

  return 0;
}

static int cutorch_getMemoryUsage(lua_State *L) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  int curDevice;
  THCudaCheck(cudaGetDevice(&curDevice));

  int device = luaL_optint(L, 1, -10);
  if (device == -10) { /* no argument passed, current device mem usage */
    THCudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes));
  } else { /* argument was given, particular device's memory usage */
    THCudaCheck(cudaSetDevice(device-1)); /* zero indexed */
    THCudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes));
    THCudaCheck(cudaSetDevice(curDevice));
  }
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

  /* The stream is per device, so update the stream as well */
  THCState_setStream(state, device, THCState_getCurrentStreamIndex(state));
  THCState_setBlasHandle(state, device, THCState_getCurrentBlasHandleIndex(state));

  return 0;
}

#define SET_DEVN_PROP(NAME) \
  lua_pushnumber(L, prop.NAME); \
  lua_setfield(L, -2, #NAME);

static int cutorch_getDeviceProperties(lua_State *L)
{
  int device = (int)luaL_checknumber(L, 1)-1;

  // switch context to given device so the call to cudaMemGetInfo is for the correct device
  int oldDevice;
  THCudaCheck(cudaGetDevice(&oldDevice));
  THCudaCheck(cudaSetDevice(device));

  struct cudaDeviceProp prop;
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

  // restore context
  THCudaCheck(cudaSetDevice(oldDevice));

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

static int cutorch_Event_new(lua_State *L)
{
  cudaEvent_t *event = luaT_alloc(L, sizeof(cudaEvent_t));
  THCudaCheck(cudaEventCreate(event));

  THCState *state = cutorch_getstate(L);
  THCudaCheck(cudaEventRecord(*event, THCState_getCurrentStream(state)));
  luaT_pushudata(L, event, "cutorch.Event");

  return 1;
}

static int cutorch_Event_free(lua_State *L)
{
  cudaEvent_t *event = luaT_checkudata(L, 1, "cutorch.Event");
  THCudaCheck(cudaEventDestroy(*event));
  luaT_free(L, event);

  return 0;
}

static int cutorch_Event_waitOn(lua_State *L)
{
  cudaEvent_t *event = luaT_checkudata(L, 1, "cutorch.Event");
  THCState *state = cutorch_getstate(L);
  THCudaCheck(cudaStreamWaitEvent(THCState_getCurrentStream(state), *event, 0));

  return 0;
}

static const struct luaL_Reg cutorch_Event__[] = {
  {"waitOn", cutorch_Event_waitOn},
  {NULL, NULL}
};

static void cutorch_Event_init(lua_State *L)
{
  luaT_newmetatable(L, "cutorch.Event", NULL, cutorch_Event_new, cutorch_Event_free, NULL);
  luaT_setfuncs(L, cutorch_Event__, 0);
  lua_pop(L, 1);
}

static void luaCutorchGCFunction(void *data)
{
  lua_State *L = data;
  lua_gc(L, LUA_GCCOLLECT, 0);
}

static int cutorch_setHeapTracking(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  int enabled = luaT_checkboolean(L,1);
  if(enabled) {
    THCSetGCHandler(state, luaCutorchGCFunction, L);
  } else {
    THCSetGCHandler(state, NULL, NULL);
  }
  return 0;
}

static int cutorch_shutdown(lua_State *L)
{
  THCState **state = (THCState **) lua_topointer(L, 1);
  THCudaShutdown(*state);
  free(*state);
  return 0;
}

static const struct luaL_Reg cutorch_stuff__ [] = {
  {"synchronize", cutorch_synchronize},
  {"synchronizeAll", cutorch_synchronizeAll},
  {"reserveBlasHandles", cutorch_reserveBlasHandles},
  {"getNumBlasHandles", cutorch_getNumBlasHandles},
  {"setBlasHandle", cutorch_setBlasHandle},
  {"getBlasHandle", cutorch_getBlasHandle},
  {"reserveStreams", cutorch_reserveStreams},
  {"getNumStreams", cutorch_getNumStreams},
  {"setStream", cutorch_setStream},
  {"getStream", cutorch_getStream},
  {"setDefaultStream", cutorch_setDefaultStream},
  {"streamWaitFor", cutorch_streamWaitFor},
  {"streamWaitForMultiDevice", cutorch_streamWaitForMultiDevice},
  {"streamBarrier", cutorch_streamBarrier},
  {"streamBarrierMultiDevice", cutorch_streamBarrierMultiDevice},
  {"streamSynchronize", cutorch_streamSynchronize},
  {"getDevice", cutorch_getDevice},
  {"deviceReset", cutorch_deviceReset},
  {"getDeviceCount", cutorch_getDeviceCount},
  {"getPeerToPeerAccess", cutorch_getPeerToPeerAccess},
  {"setPeerToPeerAccess", cutorch_setPeerToPeerAccess},
  {"setKernelPeerToPeerAccess", cutorch_setKernelPeerToPeerAccess},
  {"getKernelPeerToPeerAccess", cutorch_getKernelPeerToPeerAccess},
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
  {"setHeapTracking", cutorch_setHeapTracking},
  {NULL, NULL}
};

LUA_EXTERNC DLL_EXPORT int luaopen_libcutorch(lua_State *L);

int luaopen_libcutorch(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "cutorch");
  luaL_setfuncs(L, cutorch_stuff__, 0);

  THCState* state = (THCState*)malloc(sizeof(THCState));
  THCudaInit(state);

  /* Register torch.CudaHostAllocator. */
  luaT_pushudata(L, state->cudaHostAllocator, "torch.Allocator");
  lua_setfield(L, -2, "CudaHostAllocator");

#ifdef USE_MAGMA
  THCMagma_init(state);
  lua_pushboolean(L, 1);
  lua_setfield(L, -2, "magma");
#endif

  cutorch_CudaByteStorage_init(L);
  cutorch_CudaCharStorage_init(L);
  cutorch_CudaShortStorage_init(L);
  cutorch_CudaIntStorage_init(L);
  cutorch_CudaLongStorage_init(L);
  cutorch_CudaStorage_init(L);
  cutorch_CudaDoubleStorage_init(L);
#ifdef CUDA_HALF_TENSOR
  cutorch_CudaHalfStorage_init(L);
#endif

  cutorch_CudaByteTensor_init(L);
  cutorch_CudaCharTensor_init(L);
  cutorch_CudaShortTensor_init(L);
  cutorch_CudaIntTensor_init(L);
  cutorch_CudaLongTensor_init(L);
  cutorch_CudaTensor_init(L);
  cutorch_CudaDoubleTensor_init(L);
#ifdef CUDA_HALF_TENSOR
  cutorch_CudaHalfTensor_init(L);
#endif

  cutorch_CudaTensorOperator_init(L);

  cutorch_CudaByteTensorMath_init(L);
  cutorch_CudaCharTensorMath_init(L);
  cutorch_CudaShortTensorMath_init(L);
  cutorch_CudaIntTensorMath_init(L);
  cutorch_CudaLongTensorMath_init(L);
  cutorch_CudaTensorMath_init(L);
  cutorch_CudaDoubleTensorMath_init(L);
#ifdef CUDA_HALF_TENSOR
  cutorch_CudaHalfTensorMath_init(L);
#endif

  cutorch_Event_init(L);

  /* Store state in cutorch table. */
  lua_pushlightuserdata(L, state);
  lua_setfield(L, -2, "_state");

#ifdef CUDA_HALF_TENSOR
  lua_pushboolean(L, 1);
#else
  lua_pushboolean(L, 0);
#endif
  lua_setfield(L, -2, "hasHalf");

  /* when cutorch goes out of scope, we need to make sure THCState is properly
     shut down (so that memory doesn not leak. Since _state is a lightuserdata
     we cannot associate an __gc method with it. Hence, create a userdata, and
     associate a metatable with it, which has an __gc method which properly
     calls THCudaShutdown.
  */
  /* create a new userdata type which is a pointer to a pointer */
  THCState **thc_pointer = (THCState**)lua_newuserdata(L, sizeof(void*));
  /* set the state pointer */
  *thc_pointer = state;
  /* create a table that will be used as the metatable */
  lua_newtable(L);
  /* push the gc function onto the stack */
  lua_pushcfunction(L, &cutorch_shutdown);
  /* set the __gc field in the table to the function (function is popped) */
  lua_setfield(L, -2, "__gc");
  /* now the table is on the top of the stack, and the userdata below it,
     setmetatable on the userdata with the table. table is popped */
  lua_setmetatable(L, -2);
  /* now the userdata is on top, with the cutorch table below it,
     set the field cutorch.__stategc to this userdata.
     userdata is popped, leaving cutorch table on top of the stack */
  lua_setfield(L, -2, "_stategc");

  return 1;
}
