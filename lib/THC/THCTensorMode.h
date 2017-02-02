#ifndef TH_CUDA_TENSOR_MODE
#define TH_CUDA_TENSOR_MODE

/* Returns the mode, and index of the mode, for the set of values
 * along a given dimension in the input tensor. */
THC_API void THCudaTensor_mode(THCState* state,
                               THCudaTensor* values,
                               THCudaLongTensor* indices,
                               THCudaTensor* input,
                               int dimension);

#endif
