#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#include <cuda.h>
#include <limits>
#include "THCHalf.h"


template <typename T>
struct CudaNumericLimits
{
};

template <typename T, bool is_int>
struct THCNumericsBase {
};

template <typename T>
struct THCMathTraitsBase
{
   typedef T storage_type;
   /* type value should be converted to before doing math on it.
      For most types except 16-bit floats, MathType==StorageType.
   */
   typedef T math_type;
   /* type of expression , like (a*b). Usually == MathType */
   typedef T expr_type;

   static const storage_type one();
   static const storage_type zero();
};

template <typename T>
const T THCMathTraitsBase<T>::zero() { return T(0); }

template <typename T>
const T THCMathTraitsBase<T>::one() { return T(1); }

template <typename T>
struct THCMathTraits: public THCMathTraitsBase<T>
{
};

/* default handling for bare half is pseudo */
template <>
struct THCMathTraits<half>: public THCMathTraitsBase<half>
{
   typedef float math_type;
   typedef half expr_type;
};

template <>
struct THCMathTraits<Half>: public THCMathTraitsBase<half>
{
   typedef half math_type;
   typedef half expr_type;
};

template <typename T>
struct THCNumericsCommonBase {
  typedef THCMathTraits<T> traits;
  typedef typename traits::storage_type storage_type;
  typedef typename traits::math_type math_type;
  typedef typename traits::expr_type expr_type;

  static __host__ __device__ __forceinline__ math_type m_(const storage_type& a) {
    return ScalarConvert<storage_type, math_type>::to(a);
  }
  static __host__ __device__ __forceinline__ expr_type e_(const math_type& a) {
    return ScalarConvert<math_type, expr_type>::to(a);
  }
  static __host__ __device__ __forceinline__ storage_type s_(const expr_type& a) {
    return ScalarConvert<expr_type, storage_type>::to(a);
  }
  static __host__ __device__ const T min();
  static __host__ __device__ const T max();

  static inline __host__ __device__ bool lt(const storage_type&  a, const storage_type&  b) { return m_(a) < m_(b);  }
  static inline __host__ __device__ bool le(const storage_type&  a, const storage_type&  b) { return m_(a) <= m_(b); }
  static inline __host__ __device__ bool gt(const storage_type&  a, const storage_type&  b) { return m_(a) > m_(b);  }
  static inline __host__ __device__ bool ge(const storage_type&  a, const storage_type&  b) { return m_(a) >= m_(b); }
  static inline __host__ __device__ bool eq(const storage_type&  a, const storage_type&  b) { return m_(a) == m_(b); }
  static inline __host__ __device__ bool ne(const storage_type&  a, const storage_type&  b) { return m_(a) != m_(b); }

  static inline __host__ __device__  expr_type  add(const storage_type&  a, const storage_type&  b) { return e_(m_(a) + m_(b)); }
  static inline __host__ __device__  expr_type  mul(const storage_type&  a, const storage_type&  b) { return e_(m_(a) * m_(b)); }
  static inline __host__ __device__  expr_type  sub(const storage_type&  a, const storage_type&  b) { return e_(m_(a) - m_(b)); }
  static inline __host__ __device__  expr_type  div(const storage_type&  a, const storage_type&  b) { return e_(m_(a) / m_(b)); }
  static inline __host__ __device__  expr_type  abs(const storage_type&  a) { return e_(abs(m_(a))); }
  static inline __host__ __device__  expr_type  neg(const storage_type& a) { return  e_(-m_(a)); }
  static inline __host__ __device__  expr_type pow (const storage_type& a, T b) { return e_(::pow((double)a, (double)b)); }

};

template <typename T>
__host__ __device__ const T THCNumericsCommonBase<T>::min() { return std::numeric_limits<T>::min(); }

template <typename T>
__host__ __device__ const T THCNumericsCommonBase<T>::max() { return std::numeric_limits<T>::max(); }

/* specialized versions */
template <>
const half THCNumericsCommonBase<half>::min();

template <>
const half THCNumericsCommonBase<half>::max();

/// Class for numeric limits of the particular data type, which
/// includes support for `half`.
template <typename T>
struct THCNumericsBase<T, true>  : public THCNumericsCommonBase<T> {
};

template <typename T>
struct THCNumericsBase<T, false> : public THCNumericsCommonBase<T> {
  typedef THCNumericsCommonBase<T> Base;
  using typename Base::traits;
  using typename Base::math_type;
  using typename Base::expr_type;
  using typename Base::storage_type;
  using Base::e_;
  using Base::m_;
  using Base::s_;


  static inline __host__ __device__  expr_type exp  (const storage_type& a) { return  e_(::exp(m_(a))); }
  static inline __host__ __device__  expr_type log  (const storage_type& a) { return  e_(::log(m_(a))); }
  static inline __host__ __device__  expr_type log1p(const storage_type& a) { return  e_(::log1p(m_(a))); }
  static inline __host__ __device__  expr_type cos  (const storage_type& a) { return  e_(::cos(m_(a))); }
  static inline __host__ __device__  expr_type sin  (const storage_type& a) { return  e_(::sin(m_(a))); }
  static inline __host__ __device__  expr_type sqrt (const storage_type& a) { return  e_(::sqrt(m_(a))); }
  static inline __host__ __device__  expr_type rsqrt(const storage_type& a) { return  e_(::rsqrt(m_(a))); }
  static inline __host__ __device__  expr_type ceil (const storage_type& a) { return  e_(::ceil(m_(a))); }
  static inline __host__ __device__  expr_type floor(const storage_type& a) { return  e_(::floor(m_(a))); }
  static inline __host__ __device__  expr_type trunc(const storage_type& a) { return  e_(::trunc(m_(a))); }
  static inline __host__ __device__  expr_type acos (const storage_type& a) { return  e_(::acos(m_(a))); }
  static inline __host__ __device__  expr_type cosh (const storage_type& a) { return  e_(::cosh(m_(a))); }
  static inline __host__ __device__  expr_type acosh(const storage_type& a) { return  e_(::acosh(m_(a))); }
  static inline __host__ __device__  expr_type asin (const storage_type& a) { return  e_(::asin(m_(a))); }
  static inline __host__ __device__  expr_type sinh (const storage_type& a) { return  e_(::sinh(m_(a))); }
  static inline __host__ __device__  expr_type asinh(const storage_type& a) { return  e_(::asinh(m_(a))); }
  static inline __host__ __device__  expr_type tan  (const storage_type& a) { return  e_(::tan(m_(a))); }
  static inline __host__ __device__  expr_type atan (const storage_type& a) { return  e_(::atan(m_(a))); }
  static inline __host__ __device__  expr_type tanh (const storage_type& a) { return  e_(::tanh(m_(a))); }
  static inline __host__ __device__  expr_type abs  (const storage_type& a) { return  e_(::abs(m_(a))); }
  static inline __host__ __device__  expr_type round(const storage_type& a) { return  e_(::round(m_(a))); }
  static inline __host__ __device__  expr_type frac (const storage_type& a) { return  e_(m_(a) - ::trunc(m_(a))); }
  static inline __host__ __device__  expr_type cinv (const storage_type& a) { return  Base::div(Base::traits::one(), a); }
  static inline __host__ __device__  expr_type pow  (const storage_type& a, T b) { return e_(::pow(m_(a), m_(b))); }
};

template <typename T>
struct THCNumerics: public THCNumericsBase<T, std::numeric_limits<T>::is_integer> {
};

#ifdef CUDA_HALF_TENSOR

template <>
struct THCNumerics<half>: public THCNumericsBase<half, false>  {
};

#if defined (__CUDA_ARCH__) && defined (CUDA_FP16_INSTRINTICS)
template <>
struct THCNumerics<Half>: public THCNumericsBase<Half, false>  {

  static inline __host__ __device__ bool lt(const half& a, const half& b) {
    return __hlt(a, b);
  }
  static inline __host__ __device__ bool le(const half& a, const half& b) {
    return __hle(a, b);
  }

  static inline __host__ __device__ bool gt(const half& a, const half& b) {
    return __hgt(a, b);
  }

  static inline __host__ __device__ bool ge(const half& a, const half& b) {
    return __hge(a, b);
  }

  static inline __host__ __device__ bool eq(const half& a, const half& b) {
    return __heq(a, b);
  }

  static inline __host__ __device__ bool ne(const half& a, const half& b) {
    return __hne(a, b);
  }
  static inline __host__ __device__ half exp(const half& a) {
    return hexp(a);
  }
  static inline __host__ __device__ half log(const half& a) {
    return hlog(a);
  }
  static inline __host__ __device__ half cos(const half& a) {
    return hcos(a);
  }
  static inline __host__ __device__ half sin(const half& a) {
    return hsin(a);
  }
  static inline __host__ __device__ half sqrt(const half& a) {
    return hsqrt(a);
  }
  static inline __host__ __device__ half rsqrt(const half& a) {
    return hrsqrt(a);
  }
  static inline __host__ __device__ half ceil(const half& a) {
    return hceil(a);
  }

  static inline __host__ __device__ half floor(const half& a) {
    return hfloor(a);
  }

  static inline __host__ __device__ half trunc(const half& a) {
    return htrunc(a);
  }

  static inline __host__ __device__ half neg(const half& a) {
    return __hneg(a);
  }

  static inline __host__ __device__ const half& add(const half& a, const half& b) {
    return __hadd(a, b);
  }
  static inline __host__ __device__ half mul(const half& a, const half& b) {
    return __hmul(a, b);
  }

  static inline __host__ __device__ half sub(const half& a, const half& b) {
    return __hsub(a, b);
  }
};
# endif
#endif

#endif // THC_NUMERICS_INC
