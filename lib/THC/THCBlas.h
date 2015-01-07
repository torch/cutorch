#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"

#undef TH_API
#define TH_API THC_API
#define real float
#define Real Cuda
#define THBlas_(NAME) TH_CONCAT_4(TH,Real,Blas_,NAME)

#define TH_GENERIC_FILE "generic/THBlas.h"
#include "generic/THBlas.h"
#undef TH_GENERIC_FILE

TH_API void THCudaBlas_cgemm(char transa, char transb, long m, long n, long k, cuComplex alpha, cuComplex *a, long lda, cuComplex *b, long ldb, cuComplex beta, cuComplex *c, long ldc);

TH_API void THCudaBlas_cgemv(char trans, long m, long n, cuComplex alpha, cuComplex *a, long lda, cuComplex *x, long incx, cuComplex beta, cuComplex *y, long incy);

#undef THBlas_
#undef real
#undef Real
#undef TH_API

#ifdef WIN32
# define TH_API THC_EXTERNC __declspec(dllimport)
#else
# define TH_API THC_EXTERNC
#endif

#endif
