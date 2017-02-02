#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathScan.cu"
#else
# ifndef THC_GENERIC_NO_MATH
template<class BinaryOp>
__host__ void THCTensor_(scanOuterDim)(THCState *state, THCTensor *tgt,
                                       THCTensor *src, long dimension,
                                       real init, BinaryOp binary_op)
{
  unsigned ndim = THCTensor_(nDimension)(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (long dim = 0; dim < dimension; dim++) {
    num_orows *= THCTensor_(size)(state, src, dim);
  }
  unsigned row_size = THCTensor_(size)(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THCTensor_(size)(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  THCTensor_kernel_scanOuterDim<real><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
    THCTensor_(data)(state, tgt), THCTensor_(data)(state, src),
    num_orows, num_irows, row_size, init, binary_op);

  THCudaCheck(cudaGetLastError());
}

template<class BinaryFunction>
__host__ void THCTensor_(scanInnermostDim)(THCState *state, THCTensor *tgt,
                                           THCTensor *src, real init,
                                           BinaryFunction binary_op)
{
  unsigned ndim = THCTensor_(nDimension)(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCTensor_(size)(state, src, dim);
  }
  unsigned row_size = THCTensor_(size)(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

  THCTensor_kernel_scanInnermostDim<real, 16, 32><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
    THCTensor_(data)(state, tgt), THCTensor_(data)(state, src), num_rows, row_size, init, binary_op);

  THCudaCheck(cudaGetLastError());
}

template<class BinaryFunction>
void THCTensor_(scanDim)(THCState *state, THCTensor *self_, THCTensor *src,
                         long dimension, real init, BinaryFunction binary_op)
{
  THCTensor_(resizeAs)(state, self_, src);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  src = THCTensor_(newContiguous)(state, src);

  if (dimension == THCTensor_(nDimension)(state, src) - 1) {
    THCTensor_(scanInnermostDim)(state, self, src, init, binary_op);
  } else {
    THCTensor_(scanOuterDim)(state, self, src, dimension, init, binary_op);
  }

  THCTensor_(free)(state, src);
  THCTensor_(freeCopyTo)(state, self, self_);
}

void THCTensor_(cumsum)(THCState *state, THCTensor *self, THCTensor *src, long dimension)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self, src));
  return THCTensor_(scanDim)(state, self, src, dimension,
                             ScalarConvert<float, real>::to(0.0), AddOp<real>());
}

void THCTensor_(cumprod)(THCState *state, THCTensor *self, THCTensor *src, long dimension)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self, src));
  return THCTensor_(scanDim)(state, self, src, dimension,
                             ScalarConvert<float, real>::to(1.0), MulOp<real>());
}
# endif /* THC_GENERIC_NO_MATH */
#endif
