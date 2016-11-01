#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC


#include <limits>
#include "THCGeneral.h"

template <typename T>
struct THCNumConstants
{
   static __host__ __device__ const T one()  { return T(1); }
   static __host__ __device__ const T zero() { return T(0); }
   static __host__ __device__ const T min()  { return std::numeric_limits<T>::min(); }
   static __host__ __device__ const T max()  { return std::numeric_limits<T>::max(); }

};

template <>
struct THCNumConstants<half>
{
  static __host__ __device__ const half one()  { half ret = THC_FLOAT_TO_HALF(1.f); return ret;} /* TODO: use literal */
  static __host__ __device__ const half zero() { half ret; ret.x = 0;  return ret;}
  static __host__ __device__ const half min() { half ret; ret.x = 0x0400;  return ret; }
  static __host__ __device__ const half max() { half ret; ret.x = 0x7BFF;  return ret; }
};

template <typename T, typename M>
struct THCNumCommonBase {
   typedef T storage_type;
   /* type value should be converted to before doing math on it.
      For most types except 16-bit floats, MathType==StorageType.
   */
   typedef M math_type;

   /* type of math operation result , like (a*b). Usually == StorageType */
   typedef T expr_type;

  static __host__ __device__ __forceinline__ math_type m_(const storage_type& a) {
    return ScalarConvert<storage_type, math_type>::to(a);
  }
  static __host__ __device__ __forceinline__ expr_type e_(const math_type& a) {
    return ScalarConvert<math_type, expr_type>::to(a);
  }
  static __host__ __device__ __forceinline__ storage_type s_(const expr_type& a) {
    return ScalarConvert<expr_type, storage_type>::to(a);
  }
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
  static inline __host__ __device__  expr_type  abs(const storage_type&  a) { bool isneg = (a<0); return e_(isneg ? -a  : a); }
  static inline __host__ __device__  expr_type  neg(const storage_type& a) { return  e_(-m_(a)); }
  static inline __host__ __device__  expr_type pow (const storage_type& a, T b) { return e_(::pow((double)a, (double)b)); }

};

template <typename T, typename M, bool is_int>
struct THCNumBase {};

template <typename T, typename M>
struct THCNumBase<T, M, true>  : public THCNumCommonBase<T, M> {
};

template <>
struct THCNumBase<long, long, true>  : public THCNumCommonBase<long, long> {
  static inline __host__ __device__  expr_type  abs(const storage_type&  a) { return labs(a); }
};

template <typename T, typename M>
struct THCNumBase<T, M, false> : public THCNumCommonBase<T, M> {
  typedef THCNumCommonBase<T, M> Base;
  using Base::e_;
  using Base::m_;
  using Base::s_;
  using typename Base::math_type;
  using typename Base::expr_type;
  using typename Base::storage_type;

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
  static inline __host__ __device__  expr_type cinv (const storage_type& a) { return  Base::div(THCNumConstants<T>::one(), a); }
  static inline __host__ __device__  expr_type pow  (const storage_type& a, T b) { return e_(::pow(m_(a), m_(b))); }
};

template <typename T>
struct THCNumerics: public THCNumBase<T, T, std::numeric_limits<T>::is_integer> {
  typedef THCNumCommonBase<T, T> Base;
  using typename Base::math_type;
  using typename Base::expr_type;
  using typename Base::storage_type;
  typedef THCNumConstants<T> Constants;
};

#ifdef CUDA_HALF_TENSOR

template <>
struct THCNumerics<half>: public THCNumBase<half, float, false>  {
  typedef THCNumCommonBase<half, float> Base;
  using typename Base::math_type;
  using typename Base::expr_type;
  using typename Base::storage_type;
  using Base::e_;
  using Base::m_;
  using Base::s_;
  typedef THCNumConstants<half> Constants;
};


#if TO_BE_CONTINUED_WITH_EXTRA_TEMPLATE_PARAM
template <>
struct THCNumerics<half,half>: public THCNumBase<half, half, false>  {
  typedef THCNumCommonBase<half, half> Base;
  using typename Base::math_type;
  using typename Base::expr_type;
  using typename Base::storage_type;
  using Base::e_;
  using Base::m_;
  using Base::s_;
  typedef THCNumConstants<half> Constants;

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
