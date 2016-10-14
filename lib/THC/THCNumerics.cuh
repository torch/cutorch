#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#include <cuda.h>
#include <limits>
#include "THCHalf.h"

using std::numeric_limits;

template <typename T, bool is_int>
struct THCNumericsBase {
};

template <typename T>
struct THC_math_traits
{
   typedef T storage_type;
   /* type value should be converted to before doing math on it.
      For most types, MathType==StorageType.
   */
   typedef T math_type;
   /* type of expression , like (a*b). Usually == MathType */
   typedef T expr_type;
};

/* default handling for bare half is pseudo */
template <>
struct THC_math_traits<half>
{
   typedef half storage_type;
   typedef float math_type;
   typedef float expr_type;
}

template <>
struct THC_math_traits<PseudoHalf>
{
   typedef half storage_type;
   typedef float math_type;
   typedef float expr_type;
}

template <>
struct THC_math_traits<Half>
{
   typedef half storage_type;
   typedef half math_type;
   typedef half expr_type;
}

/// Class for numeric limits of the particular data type, which
/// includes support for `half`.
template <typename T>
struct THCNumericsBase<T, true> {

  typedef THC_math_traits<T> traits;
  typedef typename traits::storage_type storage_type;
  typedef typename traits::math_type math_type;
  typedef typename traits::expr_type expr_type;

  static const math_type one = 1;
  static const math_type zero = 0;

  static __host__ __device__ __forceinline__ const math_type& m_(const storage_type& a) {
    return ScalarConvert<storage_type, math_type>::to(a);
  }
  static __host__ __device__ __forceinline__ const expr_type& e_(const math_type& a) {
    return ScalarConvert<math_type, expr_type>::to(a);
  }

  static inline __host__ __device__ expr_type min() { return e_(numeric_limits<T>::min()); }
  static inline __host__ __device__ expt_type max() { return e_(numeric_limits<T>::max()); }

  static inline __host__ __device__ bool lt(const storage_type&  a, const storage_type&  b) { return m_(a) < m_(b);  }
  static inline __host__ __device__ bool le(const storage_type&  a, const storage_type&  b) { return m_(a) <= m_(b); }
  static inline __host__ __device__ bool gt(const storage_type&  a, const storage_type&  b) { return m_(a) > m_(b);  }
  static inline __host__ __device__ bool ge(const storage_type&  a, const storage_type&  b) { return m_(a) >= m_(b); }
  static inline __host__ __device__ bool eq(const storage_type&  a, const storage_type&  b) { return m_(a) == m_(b); }
  static inline __host__ __device__ bool ne(const storage_type&  a, const storage_type&  b) { return m_(a) != m_(b); }

  static inline __host__ __device__  const expr_type&  add(const storage_type&  a, const storage_type&  b) { return e_(m_(a) + m_(b)); }
  static inline __host__ __device__  const expr_type&  mul(const storage_type&  a, const storage_type&  b) { return e_(m_(a) * m_(b)); }
  static inline __host__ __device__  const expr_type&  sub(const storage_type&  a, const storage_type&  b) { return e_(m_(a) - m_(b)); }
  static inline __host__ __device__  const expr_type&  div(const storage_type&  a, const storage_type&  b) { return e_(m_(a) / m_(b)); }
  static inline __host__ __device__  const expr_type&  abs(const storage_type&  a) { return e_(abs(m_(a))); }
};

template <typename T>
struct THCNumericsBase<T, false>: public THCNumericsBase<T, true> {

  static const math_type one = 1.0;
  static const math_type zero = 0.;

  static inline __host__ __device__  expr_type exp  (const storage_type& a) { return  e_(::exp(m_(a)); }
  static inline __host__ __device__  expr_type log  (const storage_type& a) { return  e_(::log(m_(a))); }
  static inline __host__ __device__  expr_type log1p(const storage_type& a) { return  e_(::log1p(m_(a))); }
  static inline __host__ __device__  expr_type cos  (const storage_type& a) { return  e_(::cos(m_(a))); }
  static inline __host__ __device__  expr_type sin  (const storage_type& a) { return  e_(::sin(m_(a))); }
  static inline __host__ __device__  expr_type sqrt (const storage_type& a) { return  e_(::sqrt(m_(a))); }
  static inline __host__ __device__  expr_type rsqrt(const storage_type& a) { return  e_(::rsqrt(m_(a))); }
  static inline __host__ __device__  expr_type ceil (const storage_type& a) { return  e_(::ceil(m_(a))); }
  static inline __host__ __device__  expr_type floor(const storage_type& a) { return  e_(::floor(m_(a))); }
  static inline __host__ __device__  expr_type trunc(const storage_type& a) { return  e_(::trunc(m_(a))); }
  static inline __host__ __device__  expr_type neg  (const storage_type& a) { return  e_(-m_(a)); }
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
  static inline __host__ __device__  expr_type cinv (const storage_type& a) { return  e_(one / m_(m_(a)))); }
  static inline __host__ __device__  expr_type pow  (const storage_type& a, T b) { return e_(::pow(a, b)); }
};

template <typename T>
struct THCNumerics: public THCNumericsBase<T, numeric_limits<T>::is_integer> {
};

/* do we need this ? */
template <>
struct THCNumerics<long> : public THCNumericsBase<long, true> {
  static inline __host__ __device__  long abs(long a) { return labs(a); }
};


#ifdef CUDA_HALF_TENSOR

template <>
struct THCNumerics<PseudoHalf>: public THCNumericsBase<PseudoHalf, false>  {
};

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
#endif

#endif // THC_NUMERICS_INC
