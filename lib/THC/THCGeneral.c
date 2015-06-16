#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"
#include "THCBlas.h"
#include "THCAllocator.h"

/* Size of scratch space available in global memory per each SM + stream */
#define GLOBAL_SCRATCH_SPACE_PER_SM_STREAM 4 * sizeof(float)

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

void THCudaInit(THCState* state)
{
  int count = 0;
  THCudaCheck(cudaGetDeviceCount(&count));

  int device = 0;
  THCudaCheck(cudaGetDevice(&device));

  state->rngState = (THCRNGState*)malloc(sizeof(THCRNGState));
  THCRandom_init(state, count, device);

  THCAllocator_init(state);

  state->numDevices = count;
  state->deviceProperties =
    (struct cudaDeviceProp*)malloc(count * sizeof(struct cudaDeviceProp));

  state->numUserStreams = 0;
  state->numUserBlasHandles = 0;

  /* Enable P2P access between all pairs, if possible */
  THCudaEnablePeerToPeerAccess(state);

  state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
    malloc(count * sizeof(THCCudaResourcesPerDevice));
  for (int i = 0; i < count; ++i) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);

    THCudaCheck(cudaSetDevice(i));
    THCudaCheck(cudaGetDeviceProperties(&state->deviceProperties[i], i));
    /* Stream index 0 will be the default stream for convenience; by
       default no user streams are reserved */
    res->streams = NULL;
    res->blasHandles = NULL;

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device */
    int numSM = state->deviceProperties[i].multiProcessorCount;
    size_t sizePerStream = numSM * GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;

    /* Allocate scratch space for each stream */
    res->devScratchSpacePerStream = malloc(sizeof(void*));
    THCudaCheck(cudaMalloc(&res->devScratchSpacePerStream[0],
                           sizePerStream));
  }

  /* Restore to previous device */
  THCudaCheck(cudaSetDevice(device));

  /* Start in the default stream on the current device */
  state->currentPerDeviceStream = 0;
  state->currentStream = NULL;

  /* There is no such thing as a default cublas handle.
     To maintain consistency with streams API, handle 0 is always NULL and we
     start counting at 1
   */
  THCState_reserveBlasHandles(state, 1);
  state->currentPerDeviceBlasHandle = 1;
  state->currentBlasHandle = THCState_getDeviceBlasHandle(state, device, 1);
}

void THCudaShutdown(THCState* state)
{
  THCRandom_shutdown(state);
  THCAllocator_shutdown(state);
  free(state->rngState);
  free(state->deviceProperties);

  int deviceCount = 0;
  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));
  THCudaCheck(cudaGetDeviceCount(&deviceCount));

  for (int dev = 0; dev < deviceCount; ++dev) {
    THCudaCheck(cudaSetDevice(dev));
    /* Free Torch-defined streams (0 is the default stream) */
    for (int stream = 1; stream <= state->numUserStreams; ++stream) {
      THCudaCheck(cudaStreamDestroy(
                    THCState_getDeviceStream(state, dev, stream)));
    }
    /* Free Torch-defined handles (0 is NULL for consistency with streams API) */
    for (int handle = 1; handle <= state->numUserBlasHandles; ++handle) {
      THCudaCheck(cublasDestroy(
                    THCState_getDeviceBlasHandle(state, dev, handle)));
    }
    /* Free per-stream scratch space; starts at 0 because there is space for
       the default stream as well*/
    for (int stream = 0; stream <= state->numUserStreams; ++stream) {
      THCudaCheck(cudaFree(THCState_getDeviceScratchSpace(state, dev, stream)));
    }

    free(state->resourcesPerDevice[dev].streams);
    free(state->resourcesPerDevice[dev].blasHandles);
    free(state->resourcesPerDevice[dev].devScratchSpacePerStream);
  }
  free(state->resourcesPerDevice);

  THCudaCheck(cudaSetDevice(prevDev));
}

void THCudaEnablePeerToPeerAccess(THCState* state)
{
  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  int numDevices = -1;
  THCudaCheck(cudaGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; ++i) {
    THCudaCheck(cudaSetDevice(i));

    for (int j = 0; j < numDevices; ++j) {
      if (i != j) {
        int can = 0;
        THCudaCheck(cudaDeviceCanAccessPeer(&can, i, j));

        if (can) {
          cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
          if (err == cudaErrorPeerAccessAlreadyEnabled) {
            // Any future call to cudaGetLastError will now return an error,
            // even though we've already dealt with this specific error here.
            // Call cudaGetLastError once to reset the last error state.
            cudaGetLastError();

            continue;
          }

          THCudaCheck(err);
        }
      }
    }
  }

  /* Restore previous device before continuing */
  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getNumDevices(THCState *state)
{
  return state->numDevices;
}

void THCState_reserveStreams(THCState* state, int numStreams)
{
  if (numStreams <= state->numUserStreams)
  {
    return;
  }

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  /* Otherwise, we have to allocate a new set of streams and stream data */
  for (int dev = 0; dev < state->numDevices; ++dev) {
    THCudaCheck(cudaSetDevice(dev));

    /* +1 for the default stream as well */
    cudaStream_t* newStreams =
      (cudaStream_t*) malloc((numStreams + 1) * sizeof(cudaStream_t));

    void** newScratchSpace =
      malloc((numStreams + 1) * sizeof(void*));

    /* Copy over old stream data
       (0 is default stream, 1 ... numUserStreams are rest) */
    for (int stream = 0; stream <= state->numUserStreams; ++stream) {
      newStreams[stream] =
        THCState_getDeviceStream(state, dev, stream);
      newScratchSpace[stream] =
        THCState_getDeviceScratchSpace(state, dev, stream);
    }

    /* Allocate new stream resources */
    size_t scratchSpaceSize = THCState_getDeviceScratchSpaceSize(state, dev);

    for (int stream = state->numUserStreams + 1; stream <= numStreams; ++stream) {
      newStreams[stream] = NULL;
      THCudaCheck(cudaStreamCreate(newStreams + stream));
      newScratchSpace[stream] = NULL;
      THCudaCheck(cudaMalloc(&newScratchSpace[stream], scratchSpaceSize));
    }

    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
    free(res->streams);
    res->streams = newStreams;
    free(res->devScratchSpacePerStream);
    res->devScratchSpacePerStream = newScratchSpace;
  }

  state->numUserStreams = numStreams;

  THCudaCheck(cudaSetDevice(prevDev));
}

void THCState_reserveBlasHandles(THCState* state, int numBlasHandles)
{
  if (numBlasHandles <= state->numUserBlasHandles)
  {
    return;
  }

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  /* Otherwise, we have to allocate a new set of blasHandles */
  for (int dev = 0; dev < state->numDevices; ++dev) {
    THCudaCheck(cudaSetDevice(dev));

    /* +1 to be consistent with stream API, blas handle 0 is NULL and unused */
    cublasHandle_t* newBlasHandles =
      (cublasHandle_t*) malloc((numBlasHandles + 1) * sizeof(cublasHandle_t));

    /* Copy over old blasHandles
       (0 is NULL, 1 ... numUserBlasHandles are rest) */
    newBlasHandles[0] = NULL;
    for (int hndl = 1; hndl <= state->numUserBlasHandles; ++hndl) {
      newBlasHandles[hndl] = THCState_getDeviceBlasHandle(state, dev, hndl);
    }

    /* Allocate new handles */
    for (int hndl = state->numUserBlasHandles + 1; hndl <= numBlasHandles; ++hndl) {
      newBlasHandles[hndl] = NULL;
      THCudaCheck(cublasCreate(newBlasHandles + hndl));
    }

    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
    free(res->blasHandles);
    res->blasHandles = newBlasHandles;
  }

  state->numUserBlasHandles = numBlasHandles;

  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getNumStreams(THCState* state)
{
  return state->numUserStreams;
}

int THCState_getNumBlasHandles(THCState* state)
{
  return state->numUserBlasHandles;
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  return &(state->resourcesPerDevice[device]);
}

cudaStream_t THCState_getDeviceStream(THCState *state, int device, int stream)
{
  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }
  return (THCState_getDeviceResourcePtr(state, device)->streams == NULL) ? 0
    : THCState_getDeviceResourcePtr(state, device)->streams[stream];
}

cublasHandle_t THCState_getDeviceBlasHandle(THCState *state, int device, int handle)
{
  if (handle <= 0 || handle > state->numUserBlasHandles)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  return THCState_getDeviceResourcePtr(state, device)->blasHandles[handle];
}

cudaStream_t THCState_getCurrentStream(THCState *state)
{
  /* This is called at the point of kernel execution.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    return state->currentStream;
  } else {
    /* assume default stream */
    return NULL;
  }
}

cublasHandle_t THCState_getCurrentBlasHandle(THCState *state)
{
  /* This is called at the point of kernel execution.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    if (state->currentBlasHandle <= 0) {
      THError("%d is not a valid handle, valid range is: (1, %d)",
              state->currentBlasHandle, state->numUserBlasHandles);
    }
    return state->currentBlasHandle;
  }
  THError("THCState and blasHandles must be set as there is no default blasHandle");
  return NULL;
}

int THCState_getCurrentStreamIndex(THCState *state)
{
  return state->currentPerDeviceStream;
}

int THCState_getCurrentBlasHandleIndex(THCState *state)
{
  if (state->currentPerDeviceBlasHandle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            state->currentPerDeviceBlasHandle, state->numUserBlasHandles);
  }
  return state->currentPerDeviceBlasHandle;
}

void THCState_setStream(THCState *state, int device, int stream)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }
  state->currentStream =
    THCState_getDeviceStream(state, device, stream);
  state->currentPerDeviceStream = stream;
  THCublasCheck(cublasSetStream(state->currentBlasHandle,
                                state->currentStream));
}

void THCState_setBlasHandle(THCState *state, int device, int handle)
{  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  if (handle > state->numUserBlasHandles || handle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  state->currentBlasHandle =
    THCState_getDeviceBlasHandle(state, device, handle);
  state->currentPerDeviceBlasHandle = handle;
  THCublasCheck(cublasSetStream(state->currentBlasHandle, state->currentStream));
}

void THCState_setStreamForCurrentDevice(THCState *state, int stream)
{
  if (state->currentPerDeviceStream != stream)
  {
    int device = -1;
    THCudaCheck(cudaGetDevice(&device));
    THCState_setStream(state, device, stream);
  }
}

void THCState_setBlasHandleForCurrentDevice(THCState *state, int handle)
{
  if (state->currentPerDeviceBlasHandle != handle)
  {
    int device = -1;
    THCudaCheck(cudaGetDevice(&device));
    THCState_setBlasHandle(state, device, handle);
  }
}

void* THCState_getCurrentDeviceScratchSpace(THCState* state)
{
  int device = -1;
  THCudaCheck(cudaGetDevice(&device));
  int stream = THCState_getCurrentStreamIndex(state);

  return THCState_getDeviceScratchSpace(state, device, stream);
}

void* THCState_getDeviceScratchSpace(THCState* state, int device, int stream)
{
  THCCudaResourcesPerDevice* res =
    THCState_getDeviceResourcePtr(state, device);

  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }

  return res->devScratchSpacePerStream[stream];
}

size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state)
{
  int device = -1;
  THCudaCheck(cudaGetDevice(&device));
  return THCState_getDeviceScratchSpaceSize(state, device);
}

size_t THCState_getDeviceScratchSpaceSize(THCState* state, int device)
{
  THCCudaResourcesPerDevice* res =
    THCState_getDeviceResourcePtr(state, device);

  return res->scratchSpacePerStream;
}

void __THCudaCheck(cudaError_t err, const char *file, const int line)
{
  if(err != cudaSuccess)
  {
    THError("%s(%i) : cuda runtime error (%d) : %s",
            file, line, err, cudaGetErrorString(err));
  }
}

void __THCublasCheck(cublasStatus_t status, const char *file, const int line)
{
  if(status != CUBLAS_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case CUBLAS_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case CUBLAS_STATUS_INVALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case CUBLAS_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

      case CUBLAS_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case CUBLAS_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

      case CUBLAS_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    THError("%s(%i) : cublas runtime error : %s",
            file, line, errmsg);
  }
}

#undef GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
