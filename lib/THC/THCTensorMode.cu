#include "THC.h"

THC_API void THCudaTensor_mode(THCState* state,
                               THCudaTensor* values,
                               THCudaLongTensor* indices,
                               THCudaTensor* input,
                               int dimension) {
  THLongStorage *dim;

  dim = THCudaTensor_newSizeOf(state, input);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(state, values, dim, NULL);
  THCudaLongTensor_resize(state, indices, dim, NULL);
  THLongStorage_free(dim);
}
