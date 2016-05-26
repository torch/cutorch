#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "THCGeneral.h"

#if CUDA_VERSION >= 7050 || CUDA_HAS_FP16

#include <cuda_fp16.h>
#include <stdint.h>

THC_EXTERNC void THCFloat2Half(THCState *state, half *out, float *in, long len);
THC_EXTERNC void THCHalf2Float(THCState *state, float *out, half *in, long len);
THC_EXTERNC half THC_float2half(float a);
THC_EXTERNC float THC_half2float(half a);

#endif

#endif
