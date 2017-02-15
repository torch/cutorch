#ifndef THC_TENSORMATH_POINTWISE_CUH
#define THC_TENSORMATH_POINTWISE_CUH

#include "THCGeneral.h"
#include "THCTensorMath.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"

template <typename T>
struct TensorSigmoidOp {
  typedef THCNumerics<T> N_;
  typedef typename N_::Constants NC_;
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out = N_::div(NC_::one(), N_::add(NC_::one(), N_::exp(N_::neg(*in))));
  }
  __device__ __forceinline__ void operator()(T* v) const {
     this->operator()(v, v);
  }
};

template <typename T>
struct TensorSignOp {
  typedef THCNumerics<T> N_;
  typedef THCNumConstants<typename N_::storage_type> NC_;

  __device__ __forceinline__ void operator()(T* out, T* in) {
    const T& orig = *in;
    *out = (N_::gt(orig, NC_::zero()) ? NC_::one() :
            N_::lt(orig, NC_::zero()) ? N_::neg(NC_::one()) :
            NC_::zero());
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

template<typename T>
struct TensorTPowOp {
  TensorTPowOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::pow(val, *in);
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = THCNumerics<T>::pow(val, *v);
  }

  const T val;
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
struct TensorCRemainderOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in != 0 ? *out - *in * (*out / *in) : NAN;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in2 != 0 ? *in1 - *in2 * (*in1 / *in2) : NAN;
  }
};

template <>
struct TensorCRemainderOp<float> {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in != 0 ? *out - *in * floorf(*out / *in) : NAN;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in2 != 0 ? *in1 - *in2 * floorf(*in1 / *in2) : NAN;
  }
};

template <>
struct TensorCRemainderOp<double> {
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = *in != 0 ? *out - *in * floor(*out / *in) : NAN;
  }

  __device__ __forceinline__ void operator()(double* out, double* in1, double* in2) {
    *out = *in2 != 0 ? *in1 - *in2 * floor(*in1 / *in2) : NAN;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCRemainderOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*out, __hmul(*in, hfloor(hdiv(*out, *in))));
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    *out = fin != 0 ? __float2half(fout - fin * floorf(fout / fin)) : __float2half(NAN);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*in1, __hmul(*in2, hfloor(hdiv(*in1, *in2))));
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    *out = fin2 != 0 ? __float2half(fin1 - fin2 * floorf(fin1 / fin2)) : __float2half(NAN);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorCFmodOp {
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = N_::s_(N_::mod(*in1,*in2));
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
