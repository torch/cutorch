#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"

/* We compile with CudaHalfTensor support if we have this: */
#if CUDA_VERSION >= 7050 || CUDA_HAS_FP16
#  define CUDA_HALF_TENSOR 1
#  ifndef TH_HALF_TYPE
#   define TH_HALF_TYPE __half
#  endif
#endif

#include "THHalf.h"

#include <stdint.h>

THC_API void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len);
THC_API void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len);

THC_API half THC_float2half(float a);
THC_API float THC_half2float(half a);


/* Check for native fp16 support on the current device (CC 5.3+) */
THC_API int THC_nativeHalfInstructions(THCState *state);

/* Check for performant native fp16 support on the current device */
THC_API int THC_fastHalfInstructions(THCState *state);

#endif
