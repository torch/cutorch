#include "THC.h"
#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

THC_API void THCudaTensor_multiSelect(THCState* state, THCudaTensor* self, THCudaTensor* src, THLongStorage *position) {
  // TODO: memory tracking for sets
  THCudaTensor* temp = src;
  for (int i = 0; i < THLongStorage_size(position); ++i) {
    /* printf("Selecting on dimension %d with index %d\n", i, THLongStorage_data(position)[i]); */
    THCudaTensor_select(state, self, temp, 0, THLongStorage_data(position)[i]);
    temp = self;
  }
}

THC_API void THCudaTensor_calculateMode(THCState* state,
                                        THCudaTensor* values,
                                        THCudaLongTensor* indices,
                                        THCudaTensor* input,
                                        int dimension,
                                        THLongStorage* position) {
  // TODO: asserts, cuda check, gpu check etc.
  THCudaTensor *vec = THCudaTensor_new(state);
  if (THLongStorage_size(position) == 0) {

    THCudaTensor_set(state, vec, input);
  } else {
    THCudaTensor_multiSelect(state, vec, input, position);
  }
  THAssert(THCudaTensor_isContiguous(state, vec));
  long nElement = THCudaTensor_nElement(state, vec);

  THCThrustAllocator thrustAlloc(state);

  thrust::device_ptr<float> vecPtr(THCudaTensor_data(state, vec));
  thrust::device_vector<float> iter(vecPtr, vecPtr + nElement);
  thrust::device_vector<long> seq(nElement);
  thrust::sequence(seq.begin(), seq.end());

  thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), seq.begin());

  // Count # of unique elements:
  int unique = thrust::inner_product(iter.begin(), iter.end() - 1, iter.begin() + 1, 0, thrust::plus<int>(), thrust::not_equal_to<float>()) + 1;

  // Count frequency of each element
  thrust::device_vector<float> keys(unique);
  thrust::device_vector<int> counts(unique);
  thrust::reduce_by_key(
      iter.begin(), iter.end(),
      /* vecPtr, vecPtr + nElement, */
      thrust::constant_iterator<int>(1), keys.begin(), counts.begin());

  // Find index of maximum count
  thrust::device_vector<int>::iterator it = thrust::max_element(counts.begin(), counts.end());
  float mode = keys[it - counts.begin()];

  // Find first index within which it occurs
  thrust::device_vector<float>::iterator it2 = thrust::find(iter.begin(), iter.end(), mode);
  // assertion against iterator being end?
  long index = seq[thrust::distance(iter.begin(), it2)];

  // Place mode, index in output
  ptrdiff_t valuesOffset = THCudaTensor_storageOffset(state, values);
  long indicesOffset = THCudaLongTensor_storageOffset(state, indices);

  for (int i = 0; i < THLongStorage_size(position); ++i) {
    long pos = THLongStorage_data(position)[i];
    valuesOffset += THCudaTensor_stride(state, values, i) * pos;
    indicesOffset += THCudaLongTensor_stride(state, indices, i) * pos;
  }

  /* printf("Setting ValuesTensor at offset %ld\n", valuesOffset); */
  THCudaStorage_set(state, THCudaTensor_storage(state, values), valuesOffset, mode);
  /* printf("Setting IndicesTensor at offset %ld\n", indicesOffset); */
  THCudaLongStorage_set(state, THCudaLongTensor_storage(state, indices), indicesOffset, index);

  THCudaTensor_free(state, vec);
}

// TODO: this probably should be a loop, not a recursive algorithm
THC_API void THCudaTensor_dimApplyMode(THCState* state,
                               THCudaTensor* values,
                               THCudaLongTensor* indices,
                               THCudaTensor* input,
                               int dimension,
                               THLongStorage* position,
                               int curDim) {
  // TODO: asserts?

  // Because we have transposed the Tensor, the data for the dimension we are mode'ing along
  // is always in the innermost dimension
  long ndim = THCudaTensor_nDimension(state, input);
  if (curDim == ndim - 2) {
    // This is the innermost dimension where we are going to have to process each
    // size value, so we call the mode here
    for (int i = 0; i < THCudaTensor_size(state, input, curDim); ++i) {
      position->data[curDim] = i;
      THCudaTensor_calculateMode(state, values, indices, input, dimension, position);
    }
  } else {
    // Loop through the values and recurse
    for (int i = 0; i < THCudaTensor_size(state, input, curDim); ++i) {
      position->data[curDim] = i;
      THCudaTensor_dimApplyMode(state, values, indices, input, dimension, position, curDim + 1);
    }
  }
  position->data[curDim] = 0;
}

THC_API void THCudaTensor_mode(THCState* state,
                               THCudaTensor* values,
                               THCudaLongTensor* indices,
                               THCudaTensor* input,
                               int dimension) {
  // TODO: THCudaCheck. GPU Check
  THLongStorage *dim;
  THCudaTensor *transposed, *contiguous;
  THLongStorage * position;

  long ndim = THCudaTensor_nDimension(state, input);

  THArgCheck(dimension >= 0 && dimension < ndim, 4, "Dimension of out bounds");

  // empty tensor
  if (ndim == 0) {
    return;
  }

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

  // We also need to view the Tensors as transposed in order to properly determine
  // the output indices
  THCudaTensor *valuesTransposed = THCudaTensor_newTranspose(state, values, dimension, ndim-1);
  THCudaLongTensor *indicesTransposed = THCudaLongTensor_newTranspose(state, indices, dimension, ndim-1);

  // And also a Storage that will store the dimension values we are processing
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
