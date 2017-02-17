#include "THC.h"
#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

THC_API void THCudaTensor_calculateMode(THCState* state,
                                        THCudaTensor* values,
                                        THCudaLongTensor* indices,
                                        THCudaTensor* input,
                                        int dimension,
                                        THLongStorage* position) {
  THAssert(THCudaTensor_isContiguous(state, input));

  // Because the input is contiguous, we want to get a reference to the
  // location of the buffer at the innermost dimension that we are going
  // to calculate the mode for --> we do this by manually doing the stride
  // calculations to get an offset
  float *data = THCudaTensor_data(state, input);
  for (int i = 0; i < THLongStorage_size(position); ++i) {
    data += THLongStorage_data(position)[i] * THCudaTensor_stride(state, input, i);
  }

  long nElement = THCudaTensor_size(state, input, THCudaTensor_nDimension(state, input) - 1);
  THCThrustAllocator thrustAlloc(state);

  thrust::device_ptr<float> vecPtr = thrust::device_pointer_cast(data);
  thrust::device_vector<float> iter(vecPtr, vecPtr + nElement);
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
    iter.begin(), iter.end(), seq.begin());

  // Count # of unique elements:
  int unique = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end() - 1, iter.begin() + 1, 0, thrust::plus<int>(), thrust::not_equal_to<float>()) + 1;

  // Count frequency of each element
  thrust::device_vector<float> keys(unique);
  thrust::device_vector<int> counts(unique);
  thrust::reduce_by_key(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(),
    thrust::constant_iterator<int>(1), keys.begin(), counts.begin());

  // Find index of maximum count
  thrust::device_vector<int>::iterator it = thrust::max_element(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    counts.begin(), counts.end());
  float mode = keys[it - counts.begin()];

  // Find first index within which it occurs
  thrust::device_vector<float>::iterator it2 = thrust::find(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), mode);
  // assertion against iterator being end?
  long index = seq[it2 - iter.begin()];

  // Place mode, index in output
  ptrdiff_t valuesOffset = THCudaTensor_storageOffset(state, values);
  long indicesOffset = THCudaLongTensor_storageOffset(state, indices);

  for (int i = 0; i < THLongStorage_size(position); ++i) {
    long pos = THLongStorage_data(position)[i];
    valuesOffset += THCudaTensor_stride(state, values, i) * pos;
    indicesOffset += THCudaLongTensor_stride(state, indices, i) * pos;
  }
  THCudaStorage_set(state, THCudaTensor_storage(state, values), valuesOffset, mode);
  THCudaLongStorage_set(state, THCudaLongTensor_storage(state, indices), indicesOffset, index);
}

// TODO: this probably should be a loop, not a recursive algorithm
THC_API void THCudaTensor_dimApplyMode(THCState* state,
                               THCudaTensor* values,
                               THCudaLongTensor* indices,
                               THCudaTensor* input,
                               int dimension,
                               THLongStorage* position,
                               int curDim) {
  long ndim = THCudaTensor_nDimension(state, input);

  // Because we have transposed the Tensor, the data for the dimension we are mode'ing along
  // is always in the innermost dimension
  if (curDim == ndim - 1) {
    THCudaTensor_calculateMode(state, values, indices, input, dimension, position);
  } else {
    // Loop through the values and recurse
    for (int i = 0; i < THCudaTensor_size(state, input, curDim); ++i) {
      position->data[curDim] = i;
      THCudaTensor_dimApplyMode(state, values, indices, input, dimension, position, curDim + 1);
    }
  }
}

THC_API void THCudaTensor_mode(THCState *state,
                               THCudaTensor *values,
                               THCudaLongTensor *indices,
                               THCudaTensor *input,
                               int dimension) {
  THLongStorage *dim;
  THCudaTensor *transposed, *contiguous, *valuesTransposed;
  THLongStorage *position;
  THCudaLongTensor *indicesTransposed;
  long ndim;

  THAssert(THCudaTensor_checkGPU(state, 1, values));

  // Verify they are asking for a valid dimension
  ndim = THCudaTensor_nDimension(state, input);
  THArgCheck(dimension >= 0 && dimension < ndim, 4, "Dimension of out bounds");

  // Resize output value, index Tensors to appropriate sizes (i.e. the same as
  // the input Tensor, except at dim=dimension, the size is 1)
  dim = THCudaTensor_newSizeOf(state, input);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(state, values, dim, NULL);
  THCudaLongTensor_resize(state, indices, dim, NULL);
  THLongStorage_free(dim);

  // Beginning our naive implementation: We don't want to mutate the input Tensor, but
  // we need to be able to sort the inputs along the dimension in order to calculate the
  // mode. Additionally, its ideal if the data along the dimension is contiguous. So
  // we transpose the dimension with the innermost dimension and make a new contiguous
  // version that we can use.

  transposed = THCudaTensor_newClone(state, input);
  THCudaTensor_transpose(state, transposed, NULL, dimension, ndim - 1);
  contiguous = THCudaTensor_newContiguous(state, transposed);
  THCudaTensor_free(state, transposed);

  // We also need to view the values and indices Tensors as transposed in order to
  // properly determine the offset into the underlying storage in which to place the
  // mode and index for a particular set of dimension values
  valuesTransposed = THCudaTensor_newTranspose(state, values, dimension, ndim-1);
  indicesTransposed = THCudaLongTensor_newTranspose(state, indices, dimension, ndim-1);

  // Position is a Storage that will store the dimension values we are processing
  position = THLongStorage_newWithSize(ndim - 1);

  // Special Case: if its a 1D Tensor, call the mode code on that dimension directly
  if (ndim == 1) {
    THCudaTensor_calculateMode(state, valuesTransposed, indicesTransposed, contiguous, dimension, position);
  } else {
    THCudaTensor_dimApplyMode(state, valuesTransposed, indicesTransposed, contiguous, dimension, position, 0);
  }

  THCudaTensor_free(state, contiguous);
  THLongStorage_free(position);
  THCudaTensor_free(state, valuesTransposed);
  THCudaLongTensor_free(state, indicesTransposed);
}
