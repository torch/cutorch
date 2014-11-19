#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"

typedef struct THCudaBlasState {
  cublasHandle_t* handles;
  cublasHandle_t* current_handle;
  int n_devices;
} THCudaBlasState;

THC_API void THCudaBlas_swap(THCudaBlasState* state, long n, float *x, long incx, float *y, long incy);
THC_API void THCudaBlas_scal(THCudaBlasState* state, long n, float a, float *x, long incx);
THC_API void THCudaBlas_copy(THCudaBlasState* state, long n, float *x, long incx, float *y, long incy);
THC_API void THCudaBlas_axpy(THCudaBlasState* state, long n, float a, float *x, long incx, float *y, long incy);
THC_API float THCudaBlas_dot(THCudaBlasState* state, long n, float *x, long incx, float *y, long incy);
THC_API void THCudaBlas_gemv(THCudaBlasState* state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy);
THC_API void THCudaBlas_ger(THCudaBlasState* state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda);
THC_API void THCudaBlas_gemm(THCudaBlasState* state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);

#endif
