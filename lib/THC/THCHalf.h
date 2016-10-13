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

/* CPU emulation */
THC_EXTERNC half THC_float2half(float a);
THC_EXTERNC float THC_half2float(half a);

#if defined (__CUDA_ARCH__)
# define THC_FLOAT_TO_HALF(x) __float2half((float)x)
# define THC_HALF_TO_FLOAT(x) __half2float((float)x)
#else
# define THC_FLOAT_TO_HALF(x) THC_float2half((float)x)
# define THC_HALF_TO_FLOAT(x) THC_half2float((float)x)
#endif

/* Check for native fp16 support on the current device (CC 5.3+) */
THC_EXTERNC int THC_nativeHalfInstructions(THCState *state);

__host__ __device__ __forceinline__ bool operator==(const half& a, const half& b) {
  return a.x == b.x;
}

__host__ __device__ __forceinline__ bool operator!=(const half& a, const half& b) {
  return a.x != b.x;
}

#endif /* CUDA_HALF_TENSOR */

#ifdef __CUDA_ARCH__
//
// host (CPU) routines
//
THC_EXTERNC void THCFloat2Half(THCState *state, half *out, float *in, long len);
THC_EXTERNC void THCHalf2Float(THCState *state, float *out, half *in, long len);

/// `half` has some type conversion issues associated with it, since it
/// is a struct without a constructor/implicit conversion constructor.
/// We use this to convert scalar values to the given type that the
/// tensor expects.

template <typename In, typename Out>
struct ScalarConvert {
  static inline __host__ __device__ Out to(const In& v) { return Out(v); }
};

template <typename Out>
struct ScalarConvert<half, Out> {
  static __host__ __device__ __forceinline__ Out to(const half& v) {
    return (Out) THC_HALF_TO_FLOAT(v);
  }
};

template <typename In>
struct ScalarConvert<In, half> {
  static __host__ __device__ __forceinline__ half to(const In& v) {
    return THC_FLOAT_TO_HALF(v);
  }
};

template <>
struct ScalarConvert<half, half> {
  static __host__ __device__ __forceinline__ half to(const half& v) {
    return v;
  }
};
#endif /* CUDA */

#endif
