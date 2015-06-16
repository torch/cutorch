#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include "THGeneral.h"
#include "THAllocator.h"
#undef log1p

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#ifdef __cplusplus
# define THC_EXTERNC extern "C"
#else
# define THC_EXTERNC extern
#endif

#ifdef WIN32
# ifdef THC_EXPORTS
#  define THC_API THC_EXTERNC __declspec(dllexport)
# else
#  define THC_API THC_EXTERNC __declspec(dllimport)
# endif
#else
# define THC_API THC_EXTERNC
#endif

#ifndef THAssert
#define THAssert(exp)                                                   \
  do {                                                                  \
    if (!(exp)) {                                                       \
      THError("assert(%s) failed in file %s, line %d", #exp, __FILE__, __LINE__); \
    }                                                                   \
  } while(0)
#endif

struct THCRNGState;  /* Random number generator state. */

typedef struct _THCCudaResourcesPerDevice {
  cudaStream_t* streams;
  cublasHandle_t* blasHandles;
  /* Size of scratch space per each stream on this device available */
  size_t scratchSpacePerStream;
  /* Device-resident scratch space per stream, used for global memory
     reduction kernels. */
  void** devScratchSpacePerStream;
} THCCudaResourcesPerDevice;


/* Global state to be held in the cutorch table. */
typedef struct THCState
{
  struct THCRNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
  /* Convenience reference to the current stream/handle/plan in use */
  cudaStream_t currentStream;
  cublasHandle_t currentBlasHandle;
  /* Set of all allocated resources. resourcePerDevice[dev]->streams[0] is NULL,
     which specifies the per-device default stream. blasHandles do not have a
     default and must be explicitly initialized. We always initialize 1
     blasHandle but we can use more.
  */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;
  /* Number of Torch defined resources available, indices 1 ... numStreams */
  int numUserStreams;
  int numUserBlasHandles;
  /* Index of the current selected per-device resource. Actual CUDA resource
     changes based on the current device, since resources are per-device */
  int currentPerDeviceStream;
  int currentPerDeviceBlasHandle;
  /* Allocator using cudaMallocHost. */
  THAllocator* cudaHostAllocator;
} THCState;

THC_API void THCudaInit(THCState* state);
THC_API void THCudaShutdown(THCState* state);
THC_API void THCudaEnablePeerToPeerAccess(THCState* state);

/* State manipulators and accessors */
THC_API int THCState_getNumDevices(THCState* state);
THC_API void THCState_reserveStreams(THCState* state, int numStreams);
THC_API int THCState_getNumStreams(THCState* state);

THC_API cudaStream_t THCState_getDeviceStream(THCState *state, int device, int stream);
THC_API cudaStream_t THCState_getCurrentStream(THCState *state);
THC_API int THCState_getCurrentStreamIndex(THCState *state);
THC_API void THCState_setStream(THCState *state, int device, int stream);
THC_API void THCState_setStreamForCurrentDevice(THCState *state, int stream);

THC_API void THCState_reserveBlasHandles(THCState* state, int numHandles);
THC_API int THCState_getNumBlasHandles(THCState* state);

THC_API cublasHandle_t THCState_getDeviceBlasHandle(THCState *state, int device, int handle);
THC_API cublasHandle_t THCState_getCurrentBlasHandle(THCState *state);
THC_API int THCState_getCurrentBlasHandleIndex(THCState *state);
THC_API void THCState_setBlasHandle(THCState *state, int device, int handle);
THC_API void THCState_setBlasHandleForCurrentDevice(THCState *state, int handle);

/* For the current device and stream, returns the allocated scratch space */
THC_API void* THCState_getCurrentDeviceScratchSpace(THCState* state);
THC_API void* THCState_getDeviceScratchSpace(THCState* state, int device, int stream);
THC_API size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state);
THC_API size_t THCState_getDeviceScratchSpaceSize(THCState* state, int device);

#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCublasCheck(err)  __THCublasCheck(err,  __FILE__, __LINE__)

THC_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
THC_API void __THCublasCheck(cublasStatus_t status, const char *file, const int line);

#endif
