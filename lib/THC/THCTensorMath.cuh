#ifndef THC_TENSORMATH_CUH
#define THC_TENSORMATH_CUH

// Copy the kth diagonal of a matrix B to a vector A.
template <typename T>
__global__ void THCTensor_copyFromDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideA) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
template <typename T>
__global__ void THCTensor_copyToDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideB) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

// Limited by the maximum size of kernel arguments (4kb)
#define CAT_ARRAY_KERNEL_BATCH_SIZE 16

template <typename T>
struct CatArrayKernelParam {
  // Tensors to copy into the output parameter
  TensorInfo<T, unsigned int> inputs[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // The offsets along the dimension in the output tensor where we
  // should begin the copy, for each tensor
  int offsets[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // The of the dimension for each input tensor, used to specify
  // how to narrow the output tensor
  int dimSizes[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // Number of elements in each tensor, for the grid-stride loop
  // bound
  int nElements[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // Actual number of tensors in this param (may be less than the
  // batch size)
  int count;
};

template <typename T>
__global__ void catArrayBatchedCopy(TensorInfo<T, unsigned int> result, const CatArrayKernelParam<T> param, int dimension) {
  // A block is responsible for the ith tensor in the param if its blockDim.y = i, so let's narrow
  // the result TensorInfo according to the offset, dimSize for that tensor
  TensorInfo<T, unsigned int> nt = result.newNarrow(dimension, param.offsets[blockIdx.y], param.dimSizes[blockIdx.y]);

  // Now follow the normal pointwise op code, where the the linear index is determined by thread/block x values
  for (unsigned int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < param.nElements[blockIdx.y];
       linearIndex += gridDim.x * blockDim.x) {
    const unsigned int resultOffset = IndexToOffset<T, unsigned int, -1>::get(linearIndex, nt);
    const unsigned int srcOffset = IndexToOffset<T, unsigned int, -1>::get(linearIndex, param.inputs[blockIdx.y]);
    nt.data[resultOffset] = param.inputs[blockIdx.y].data[srcOffset];
  }
}

#endif
