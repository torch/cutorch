#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMode.cu"
#else

THC_API void THCTensor_(calculateMode)(THCState* state,
                                        THCTensor* values,
                                        THCudaLongTensor* indices,
                                        THCTensor* input,
                                        int dimension,
                                        THLongStorage* position) {
  THAssert(THCTensor_(isContiguous)(state, input));

  // Because the input is contiguous, we want to get a reference to the
  // location of the buffer at the innermost dimension that we are going
  // to calculate the mode for --> we do this by manually doing the stride
  // calculations to get an offset
  real *data = THCTensor_(data)(state, input);
  for (int i = 0; i < THLongStorage_size(position); ++i) {
    data += THLongStorage_data(position)[i] * THCTensor_(stride)(state, input, i);
  }

  long nElement = THCTensor_(size)(state, input, THCTensor_(nDimension)(state, input) - 1);
  THCThrustAllocator thrustAlloc(state);

  thrust::device_ptr<real> vecPtr = thrust::device_pointer_cast(data);
  thrust::device_vector<real> iter(vecPtr, vecPtr + nElement);
  thrust::device_vector<long> seq(nElement);
  thrust::sequence(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    seq.begin(), seq.end());

  thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), seq.begin()
#if defined(THC_REAL_IS_HALF)
    , ThrustHalfLess()
#endif
  );

  // Count # of unique elements:
  int unique = 1 + thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end() - 1, iter.begin() + 1, 0, thrust::plus<int>(),
#if defined(THC_REAL_IS_HALF)
    ThrustHalfNotEqualTo()
#else
    thrust::not_equal_to<real>()
#endif
  );

  // Count frequency of each element
  thrust::device_vector<real> keys(unique);
  thrust::device_vector<int> counts(unique);
  thrust::reduce_by_key(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(),
    thrust::constant_iterator<int>(1), keys.begin(), counts.begin()
#if defined(THC_REAL_IS_HALF)
    , ThrustHalfEqualTo()
#endif
  );

  // Find index of maximum count
  thrust::device_vector<int>::iterator it = thrust::max_element(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    counts.begin(), counts.end());
  real mode = keys[it - counts.begin()];

  // Find first index within which it occurs
#if defined(THC_REAL_IS_HALF)
  thrust::device_vector<real>::iterator it2 = thrust::find_if(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), ThrustHalfEqualToPredicate(mode));
#else
  thrust::device_vector<real>::iterator it2 = thrust::find(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), mode);
#endif

  // assertion against iterator being end?
  long index = seq[it2 - iter.begin()];

  // Place mode, index in output
  ptrdiff_t valuesOffset = THCTensor_(storageOffset)(state, values);
  long indicesOffset = THCudaLongTensor_storageOffset(state, indices);

  for (int i = 0; i < THLongStorage_size(position); ++i) {
    long pos = THLongStorage_data(position)[i];
    valuesOffset += THCTensor_(stride)(state, values, i) * pos;
    indicesOffset += THCudaLongTensor_stride(state, indices, i) * pos;
  }
  THCStorage_(set)(state, THCTensor_(storage)(state, values), valuesOffset, mode);
  THCudaLongStorage_set(state, THCudaLongTensor_storage(state, indices), indicesOffset, index);
}

// TODO: this probably should be a loop, not a recursive algorithm
THC_API void THCTensor_(dimApplyMode)(THCState* state,
                               THCTensor* values,
                               THCudaLongTensor* indices,
                               THCTensor* input,
                               int dimension,
                               THLongStorage* position,
                               int curDim) {
  long ndim = THCTensor_(nDimension)(state, input);

  // Because we have transposed the Tensor, the data for the dimension we are mode'ing along
  // is always in the innermost dimension
  if (curDim == ndim - 1) {
    THCTensor_(calculateMode)(state, values, indices, input, dimension, position);
  } else {
    // Loop through the values and recurse
    for (int i = 0; i < THCTensor_(size)(state, input, curDim); ++i) {
      position->data[curDim] = i;
      THCTensor_(dimApplyMode)(state, values, indices, input, dimension, position, curDim + 1);
    }
  }
}

THC_API void THCTensor_(mode)(THCState *state,
                              THCTensor *values,
                              THCudaLongTensor *indices,
                              THCTensor *input,
                              int dimension) {
  THLongStorage *dim;
  THCTensor *transposed, *contiguous, *valuesTransposed;
  THLongStorage *position;
  THCudaLongTensor *indicesTransposed;
  long ndim;

  THAssert(THCTensor_(checkGPU)(state, 1, values));

  // Verify they are asking for a valid dimension
  ndim = THCTensor_(nDimension)(state, input);
  THArgCheck(dimension >= 0 && dimension < ndim, 4, "Dimension of out bounds");

  // Resize output value, index Tensors to appropriate sizes (i.e. the same as
  // the input Tensor, except at dim=dimension, the size is 1)
  dim = THCTensor_(newSizeOf)(state, input);
  THLongStorage_set(dim, dimension, 1);
  THCTensor_(resize)(state, values, dim, NULL);
  THCudaLongTensor_resize(state, indices, dim, NULL);
  THLongStorage_free(dim);

  // Beginning our naive implementation: We don't want to mutate the input Tensor, but
  // we need to be able to sort the inputs along the dimension in order to calculate the
  // mode. Additionally, its ideal if the data along the dimension is contiguous. So
  // we transpose the dimension with the innermost dimension and make a new contiguous
  // version that we can use.

  transposed = THCTensor_(newClone)(state, input);
  THCTensor_(transpose)(state, transposed, NULL, dimension, ndim - 1);
  contiguous = THCTensor_(newContiguous)(state, transposed);
  THCTensor_(free)(state, transposed);

  // We also need to view the values and indices Tensors as transposed in order to
  // properly determine the offset into the underlying storage in which to place the
  // mode and index for a particular set of dimension values
  valuesTransposed = THCTensor_(newTranspose)(state, values, dimension, ndim-1);
  indicesTransposed = THCudaLongTensor_newTranspose(state, indices, dimension, ndim-1);

  // Position is a Storage that will store the dimension values we are processing
  position = THLongStorage_newWithSize(ndim - 1);

  // Special Case: if its a 1D Tensor, call the mode code on that dimension directly
  if (ndim == 1) {
    THCTensor_(calculateMode)(state, valuesTransposed, indicesTransposed, contiguous, dimension, position);
  } else {
    THCTensor_(dimApplyMode)(state, valuesTransposed, indicesTransposed, contiguous, dimension, position, 0);
  }

  THCTensor_(free)(state, contiguous);
  THLongStorage_free(position);
  THCTensor_(free)(state, valuesTransposed);
  THCudaLongTensor_free(state, indicesTransposed);
}

#endif
