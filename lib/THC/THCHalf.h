#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "THCGeneral.h"

/* We compile with CudaHalfTensor support if we have this: */
#if CUDA_VERSION >= 7050 || CUDA_HAS_FP16
#define CUDA_HALF_TENSOR 1
#endif

/* This define forces use of 32-bit float math on 16-bit float type 'half' (a.k.a. "pseudo-fp16 mode")
   even if native harware support is available.
   This makes difference for Pascal (6.x) cards only: Maxwell (5.x) cards always run 'half' in pseudo mode.
   !!! Uncomment on your own risk !!!
   Native fp16 operations may in fact run slower than pseudo-fp16 on your system at the moment
  (especially if the bulk of your code is in CUDNN and not Cutorch).
*/

#define FORCE_PSEUDO_FP16 1

#ifndef FORCE_PSEUDO_FP16
/* Kernel side: Native fp16 ALU instructions are available if we have this: */
# if defined(CUDA_HALF_TENSOR) && (CUDA_VERSION >= 8000) && (__CUDA_ARCH__ >= 530)
#  define CUDA_HALF_INSTRUCTIONS 1
# endif
#endif

#ifdef CUDA_HALF_TENSOR

#include <cuda_fp16.h>
#include <stdint.h>

THC_EXTERNC void THCFloat2Half(THCState *state, half *out, float *in, long len);
THC_EXTERNC void THCHalf2Float(THCState *state, float *out, half *in, long len);
THC_EXTERNC half THC_float2half(float a);
THC_EXTERNC float THC_half2float(half a);

/* Check for native fp16 support on the current device (CC 5.3+) */
THC_EXTERNC int THC_nativeHalfInstructions(THCState *state);

#endif /* CUDA_HALF_TENSOR */

#endif
