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

// Limited by the 4kb argument limit for kernels. The larger KERNEL_MAX,
// the smaller the buffer can be. At this setting, can support concatenating
// 128 (1D Tensors), 128 (2D Tensors), 106 (3D Tensors), 80 (4D Tensors), etc.
#define CAT_ARRAY_KERNEL_MAX 128
#define CAT_ARRAY_STRIDE_BUFFER_SIZE 320

template <typename T>
struct CatArrayKernelParam {
  // Default consructor, does nothing
  __host__ __device__ CatArrayKernelParam();

  // Copy constructor -> for some reason, the compiler-generated copy constructor
  // does not properly copy data, so we need this one to successfully pass uncorrupted
  // data to the kernel
  __host__ __device__ CatArrayKernelParam(const CatArrayKernelParam& other);

  // This is the array of pointers to the Tensors we are concatenating into the result
  // Tensor.
  T *data[CAT_ARRAY_KERNEL_MAX];

  // Number of dimensions in input Tensors.
  int dims;

  // A buffer to store stride information for all the input Tensors. In particular, this
  // array stores count * dims entries: The first dims entries are the stride information
  // for the first tensor, arranged in the same format as a Tensor (i.e. outer dimension
  // first)
  int strides[CAT_ARRAY_STRIDE_BUFFER_SIZE];

  // The offsets along the dimension in the output Tensor where we should begin the
  // copy, for each Tensor.
  int offsets[CAT_ARRAY_KERNEL_MAX];

  // The size at the concatenation dimension for each input Tensor, used to specify
  // how to narrow the output tensor, as well as determining the offset for indices
  // into the input tensors.
  int dimSizes[CAT_ARRAY_KERNEL_MAX];

  // Number of elements in each tensor, for the grid-stride loop bound
  int nElements[CAT_ARRAY_KERNEL_MAX];

  // Actual number of tensors in this param (may be less than the  max size)
  int count;
};

template <typename T>
__host__ __device__ CatArrayKernelParam<T>::CatArrayKernelParam() {}

template <typename T>
__host__ __device__ CatArrayKernelParam<T>::CatArrayKernelParam(const CatArrayKernelParam& other) {
  for (int i = 0; i < other.count; ++i) {
    data[i] = other.data[i];
    offsets[i] = other.offsets[i];
    dimSizes[i] = other.dimSizes[i];
    nElements[i] = other.nElements[i];
    for (int j = 0; j < other.dims; ++j) {
      strides[(i*other.dims) + j] = other.strides[(i*other.dims) + j];
    }
  }
  dims = other.dims;
  count = other.count;
}

template <typename T, int Dims>
struct CatArrayKernelOffsetCalc {
  // Utility function to map and index to offset for a particular input tensor, and particular
  // index, in the kernel. This is the same as any other dynamic indexing approach (see e.g.
  // IndexToOffset) but we take advantage of the fact that for the catArray problem, the size
  // of the dimensions of every input Tensor must be the same for the dimensions we *are not*
  // concatenating along. Hence we can borrow them from the TensorInfo for the result.
  static inline __device__ unsigned int indexToOffset(
      const TensorInfo<T, unsigned int>& result,
      const CatArrayKernelParam<T>& param,
      const unsigned int concatDim,
      const unsigned int tensorIndex,
      unsigned int linearIndex) {
    assert(entry < param.count);

    unsigned int offset = 0;

    // Calculate offset into strides buffer -
    int bufOffset = result.dims * tensorIndex;

    // Static Dims - for unrolling this loop
    for (int i = Dims - 1; i >= 0; --i) {
      unsigned int curDimSize = i == concatDim ? param.dimSizes[tensorIndex] : result.sizes[i];
      unsigned int curDimIndex = linearIndex % curDimSize;
      unsigned int curDimOffset = curDimIndex * param.strides[i + bufOffset];
      offset += curDimOffset;

      linearIndex /= curDimSize;
    }

    return offset;
  }
};

// Dynamics Dims - cardinality only known at runtime
template <typename T>
struct CatArrayKernelOffsetCalc<T, -1> {
  static inline __device__ unsigned int indexToOffset(
      const TensorInfo<T, unsigned int>& result,
      const CatArrayKernelParam<T>& param,
      const unsigned int concatDim,
      const unsigned int tensorIndex,
      unsigned int linearIndex) {
    assert(entry < param.count);

    unsigned int offset = 0;

    // Calculate offset into strides buffer -
    int bufOffset = result.dims * tensorIndex;

    for (int i = param.dims - 1; i >= 0; --i) {
      unsigned int curDimSize = i == concatDim ? param.dimSizes[tensorIndex] : result.sizes[i];
      unsigned int curDimIndex = linearIndex % curDimSize;
      unsigned int curDimOffset = curDimIndex * param.strides[i + bufOffset];
      offset += curDimOffset;

      linearIndex /= curDimSize;
    }

    return offset;
  }
};

// Contiguous (linearIndex -> offset)
template <typename T>
struct CatArrayKernelOffsetCalc<T, -2> {
  static inline __device__ unsigned int indexToOffset(
      const TensorInfo<T, unsigned int>& result,
      const CatArrayKernelParam<T>& param,
      const unsigned int concatDim,
      const unsigned int tensorIndex,
      unsigned int linearIndex) {
    assert(entry < param.count);
    return linearIndex;
  }
};

template <typename T, int Dims>
__global__ void catArrayBatchedCopy(TensorInfo<T, unsigned int> result, const CatArrayKernelParam<T> param, int dimension) {
  // A block is responsible for the ith tensor in the param if its blockDim.y = i, so let's narrow
  // the result TensorInfo according to the offset, dimSize for that tensor
  TensorInfo<T, unsigned int> nt = result.newNarrow(dimension, param.offsets[blockIdx.y], param.dimSizes[blockIdx.y]);

  // Now follow the normal pointwise op code, where the the linear index is determined by thread/block x values
  for (unsigned int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < param.nElements[blockIdx.y];
       linearIndex += gridDim.x * blockDim.x) {
    const unsigned int resultOffset = IndexToOffset<T, unsigned int, -1>::get(linearIndex, nt);
    const unsigned int srcOffset = CatArrayKernelOffsetCalc<T, Dims>::indexToOffset(result, param, dimension, blockIdx.y, linearIndex);
    nt.data[resultOffset] = param.data[blockIdx.y][srcOffset];
  }
}

#endif
