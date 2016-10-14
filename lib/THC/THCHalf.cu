#include "THCHalf.h"
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

struct __half2floatOp {
  __device__ float operator()(half v) { return __half2float(v); }
};

struct __float2halfOp {
  __device__ half operator()(float v) { return __float2half(v); }
};

void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len) {
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __float2halfOp());
}

void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len) {
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __half2floatOp());
}
