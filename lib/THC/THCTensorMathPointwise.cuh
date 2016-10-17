#ifndef THC_TENSORMATH_POINTWISE_CUH
#define THC_TENSORMATH_POINTWISE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"

template <typename T>
struct TensorSigmoidOp {
  typedef THCNumerics<T> N_;
  typedef typename N_::traits traits;
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out = N_::div(traits::one(), N_::add(traits::one(), N_::neg(*in)));
  }
  __device__ __forceinline__ void operator()(T* v) const {
     this->operator()(v, v);
  }
};

template <typename T>
struct TensorSignOp {
  typedef THCNumerics<T> N_;
  typedef typename N_::traits traits;

  __device__ __forceinline__ void operator()(T* out, T* in) {
    const T& orig = *in;
    *out = (N_::gt(orig, traits::zero()) ? traits::one() :
            N_::lt(orig, traits::zero()) ? N_::neg(traits::one()) :
            traits::zero());
  }
  __device__ __forceinline__ void operator()(T* v) {
     this->operator()(v, v);
  }
};

template <typename T>
struct TensorAddOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::s_(N_::add(*in1, *in2));
  }
  __device__ __forceinline__ void operator()(T* out, T* in) {
    this->operator()(out, in, out);
  }

};


template <typename T>
struct TensorCAddOp {
  typedef THCNumerics<T> N_;
  TensorCAddOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    this->operator()(out, out, in);
  }
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::s_(N_::add(*in1, N_::mul(val, *in2)));
  }

 const typename N_::storage_type val;
};


template <typename T>
struct TensorSubOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::s_(N_::sub(*in1, *in2));
  }
  __device__ __forceinline__ void operator()(T* out, T* in) {
    this->operator()(out, out, in);
  }
};

template <typename T>
struct TensorMulOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::s_(N_::mul(*in1, *in2));
  }
  __device__ __forceinline__ void operator()(T* out, T* in) {
    this->operator()(out, out, in);
  }
};

template<typename T>
struct TensorPowOp {
  typedef THCNumerics<T> N_;
  TensorPowOp(T v) : val(N_::s_(v)) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::s_(N_::pow(*in, val));
  }

  __device__ __forceinline__ void operator()(T* v) {
    this->operator()(v, v);
  }
 const typename N_::storage_type val;
};

template <typename T>
struct TensorCPowOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::s_(N_::pow(*in1,*in2));
  }
  __device__ __forceinline__ void operator()(T* out, T* in) {
    this->operator()(out, out, in);
  }
};

template <typename T>
struct TensorDivOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::s_(N_::div(*in1,*in2));
  }
  __device__ __forceinline__ void operator()(T* out, T* in) {
    this->operator()(out, out, in);
  }
};

template <typename T>
struct TensorClampOp {
  typedef THCNumerics<T> N_;
  TensorClampOp(T min, T max) : minValue(min), maxValue(max) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    T val = N_::lt(*in, maxValue) ? *in : maxValue;
    *out = N_::gt(minValue, val) ? minValue : val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    T val = N_::lt(*v, maxValue) ? *v : maxValue;
    *v = N_::gt(minValue, val) ? minValue : val;
  }
  const typename N_::storage_type minValue;
  const typename N_::storage_type maxValue;
};

template <typename T>
struct TensorLerpOp {
  typedef THCNumerics<T> N_;
  TensorLerpOp(T w) : w(w) {}
  __device__ __forceinline__ void operator()(T *out, T *a, T *b) {
    *out = N_::add(*a, N_::mul(w, N_::sub(*b, *a)));
  }
  T w;
};

template <typename T>
struct TensorCrossOp {
  typedef THCNumerics<T> N_;
  TensorCrossOp(long sx, long sy, long so) : sx(sx), sy(sy), so(so) {}
  __device__ __forceinline__ void operator()(T* out, T* x, T*y) {
    out[0 * so] = N_::sub(
        N_::mul(x[1 * sx], y[2 * sy]),
        N_::mul(x[2 * sx], y[1 * sy])
    );

    out[1 * so] = N_::sub(
        N_::mul(x[2 * sx], y[0 * sy]),
        N_::mul(x[0 * sx], y[2 * sy])
    );

    out[2 * so] = N_::sub(
        N_::mul(x[0 * sx], y[1 * sy]),
        N_::mul(x[1 * sx], y[0 * sy])
    );
  }

  const long sx, sy, so;
};

template <typename T>
struct TensorMaxOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::gt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::gt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMinOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::lt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::lt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMaxValueOp {
  typedef THCNumerics<T> N_;
  TensorMaxValueOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out) {
    *out = N_::gt(*out, val) ? *out : val;
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::gt(*in, val) ? *in : val;
  }

  T val;
};

template <typename T>
struct TensorMinValueOp {
  typedef THCNumerics<T> N_;
  TensorMinValueOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out) {
    *out = N_::lt(*out, val) ? *out : val;
  }
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::lt(*in, val) ? *in : val;
  }
  T val;
};

template <typename T>
struct TensorAddCMulOp {
  typedef THCNumerics<T> N_;
  TensorAddCMulOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::add(*out,N_::mul(val,N_::mul(*in1, *in2)));
  }
  T val;
};

template <typename T>
struct TensorAddCDivOp {
  typedef THCNumerics<T> N_;
  TensorAddCDivOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::add( *out,
                    N_::mul(val,
                            N_::div(*in1, *in2)
                            )
                    );
  }
  typename N_::storage_type val;
};

#endif // THC_TENSORMATH_POINTWISE_CUH
