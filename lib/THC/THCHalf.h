#ifndef THC_HALF_CONVERSION_INC
# define THC_HALF_CONVERSION_INC

# include "THCGeneral.h"

/* We compile with CudaHalfTensor support if we have this: */
# if CUDA_VERSION >= 7050 || CUDA_HAS_FP16
#  define CUDA_HALF_TENSOR 1
# endif

# ifdef CUDA_HALF_TENSOR

#  include <cuda_fp16.h>
#  include <stdint.h>

/* CPU conversion methods, scalar */
THC_EXTERNC half  THC_float2half(float a);
THC_EXTERNC float THC_half2float(half a);

//
// Vector conversion routines, using Thrust
//
THC_EXTERNC void THCFloat2Half(THCState *state, half *out, float *in, long len);
THC_EXTERNC void THCHalf2Float(THCState *state, float *out, half *in, long len);

#  if defined (__CUDA_ARCH__)
/* use instrintic functons defined for device only in cuda_fp16.h */
#   define THC_FLOAT_TO_HALF(x) __float2half((float)x)
#   define THC_HALF_TO_FLOAT(x) __half2float(x)
#  else
/* use host conversion functions */
#   define THC_FLOAT_TO_HALF(x) THC_float2half((float)x)
#   define THC_HALF_TO_FLOAT(x) THC_half2float(x)
#  endif

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
# define CUDA_FP16_INSTRINTICS 1
#endif


/* Check for native fp16 support on the current device (CC 5.3+) */
THC_API int THC_nativeHalfInstructions(THCState *state);

/* Check for performant native fp16 support on the current device */
THC_API int THC_fastHalfInstructions(THCState *state);

#if defined (__cplusplus__) || defined (__CUDACC__)

__host__ __device__ __forceinline__ bool operator==(const half& a, const half& b) {
  return a.x == b.x;
}

__host__ __device__ __forceinline__ bool operator!=(const half& a, const half& b) {
  return a.x != b.x;
}

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
  static __host__ __device__ __forceinline__ const half& to(const half& v) {
    return v;
  }
};
#  endif /* __cplusplus__ */
# endif /* CUDA_HALF_TENSOR */
#endif /* THC_HALF_CONVERSION_INC */
