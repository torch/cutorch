#include "THC.h"
#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__global__ void printTensor(float *tensor, int count) {
  for (int i = 0; i < count; ++i) {
    // whateva u want bb
  }
}

THC_API void pt(THCState* s, THCudaTensor* t) {
  printTensor<<<1, 1>>>(THCudaTensor_data(s, t), THCudaTensor_nElement(s, t));
}

THC_API void ptm(THCState *s, THCudaTensor *t) {
  printf("Storage: %p, Size: %p, Stride: %p\n", t->storage, t->size, t->stride);
}

THC_API void THCudaTensor_multiSelect(THCState* state, THCudaTensor* self, THCudaTensor* src, THLongStorage *position) {
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
  THCudaTensor_multiSelect(state, vec, input, position);

  THCThrustAllocator thrustAlloc(state);

  thrust::device_ptr<float> iter(THCudaTensor_data(state, vec));
  /* thrust::fill(iter, iter + THCudaTensor_nElement(state, vec), 1); */

  thrust::sort(iter, iter + THCudaTensor_nElement(state, vec));

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

  // And also a Storage that will store the dimension values we are processing
  position = THLongStorage_newWithSize(ndim - 1);

  // Special Case: if its a 1D Tensor, call the mode code on that dimension directly
  if (ndim == 1) {
    THCudaTensor_calculateMode(state, values, indices, contiguous, dimension, position);
  } else {
    THCudaTensor_dimApplyMode(state, values, indices, contiguous, dimension, position, 0);
  }

  /* THCudaTensor_set(state, input, contiguous); */
  /* THCudaTensor_transpose(state, input, NULL, ndim - 1, dimension); */

  THCudaTensor_free(state, contiguous);
  THLongStorage_free(position);
}
