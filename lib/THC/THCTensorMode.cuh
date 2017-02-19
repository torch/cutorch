#ifndef THC_TENSOR_MODE_CUH
#define THC_TENSOR_MODE_CUH

#include "THCNumerics.cuh"

struct ThrustHalfLess
{
  __host__ __device__ inline bool operator()(const half& lhs, const half& rhs) {
    return THCNumerics<half>::lt(lhs, rhs);
  }
};

struct ThrustHalfNotEqualTo
{
  __host__ __device__ inline bool operator()(const half& lhs, const half& rhs) {
    return THCNumerics<half>::ne(lhs, rhs);
  }
};

struct ThrustHalfEqualTo
{
  __host__ __device__ inline bool operator()(const half& lhs, const half& rhs) {
    return THCNumerics<half>::eq(lhs, rhs);
  }
};

struct ThrustHalfEqualToPredicate
{
  ThrustHalfEqualToPredicate(half val): val_(val) {}
  __host__ __device__ inline bool operator()(half x) {
    return THCNumerics<half>::eq(val_, x);
  }

  half val_;
};



#endif // THC_TENSOR_MODE_CUH
