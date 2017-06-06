#ifndef THC_REDUCE_INC
#define THC_REDUCE_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage.
//

#include "THCTensorTypeUtils.cuh"
#include "THCReduceApplyUtils.cuh"

#include <cstdio>

#define LOCAL_MAX_BLOCK_SIZE 512

template <typename ModifyOp,
          typename ReduceOp,
          typename T,
          typename IndexType,
          int ADims, int BDims>
__global__ void
kernelReduceNoncontigDim_shared(TensorInfo<T, IndexType> out,
                         TensorInfo<T, IndexType> in,
                         IndexType reductionStride,
                         IndexType reductionSize,
                         IndexType totalSlices,
                         T init,
                         ModifyOp modifyOp,
                         ReduceOp reduceOp) {
  IndexType threadLane  = threadIdx.x;
  IndexType groupID      = threadIdx.y;
  IndexType sliceIndex  = blockIdx.x * blockDim.x + threadLane;
  IndexType sliceStride = gridDim.x * blockDim.x;

  __shared__ T local_reduce[LOCAL_MAX_BLOCK_SIZE];
  //If multiple groups of threads work on the same reduction use shared memory
  //Otherwise use a register for reduction as only one thread is doing reduction
  T* shmem = &local_reduce[threadIdx.x + threadIdx.y * blockDim.x];

  for(;sliceIndex<totalSlices; sliceIndex+=sliceStride){
    //  while(sliceIndex < totalSlices){
    *shmem = init;

    IndexType outOffset = IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);
    IndexType inOffset = IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);
    IndexType stride = reductionStride * blockDim.y;

    for(IndexType i=groupID; i<reductionSize; i+=blockDim.y){
      (*shmem) = reduceOp(*shmem, modifyOp(in.data[inOffset]) );
      inOffset += stride;
    }

    __syncthreads();

    if(groupID == 0){
      for(IndexType i=1; i<min( (IndexType) blockDim.y, reductionSize); i++){
        *shmem = reduceOp(*shmem, *(shmem + blockDim.x * i) );
      }
      out.data[outOffset] = *shmem;
    }
  }
}


// Kernel that handles an entire reduction of a slice of a tensor per each thread
template <typename ModifyOp,
          typename ReduceOp,
          typename T,
          typename IndexType,
          int ADims, int BDims>
__global__ void
kernelReduceNoncontigDim(TensorInfo<T, IndexType> out,
                         TensorInfo<T, IndexType> in,
                         IndexType reductionStride,
                         IndexType reductionSize,
                         IndexType totalSlices,
                         T init,
                         ModifyOp modifyOp,
                         ReduceOp reduceOp) {
  IndexType sliceIndex  = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType sliceStride = gridDim.x * blockDim.x;

  for(;sliceIndex<totalSlices; sliceIndex+=sliceStride){
    IndexType outOffset = IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);
    IndexType inOffset = IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);
    T r = init;

    for(IndexType i = 0; i<reductionSize; i++){
      r = reduceOp(r, modifyOp(in.data[inOffset]) );
      inOffset += reductionStride;
    }

    __syncthreads();
    out.data[outOffset] = r;
  }
}

template <typename IndexType>
__device__ __forceinline__ IndexType getReduceContigDimSliceIndex() {
  // Each block handles one slice
  return getLinearBlockId<IndexType>();
}

// Kernel that handles an entire reduction of a slice of a tensor per
// each block
template <typename ModifyOp,
          typename ReduceOp,
          typename T,
          typename IndexType,
          int ADims, int BDims>
__global__ void
kernelReduceContigDim(TensorInfo<T, IndexType> out,
                      TensorInfo<T, IndexType> in,
                      IndexType reductionSize,
                      IndexType totalSlices,
                      T init,
                      ModifyOp modifyOp,
                      ReduceOp reduceOp) {
  const IndexType sliceIndex = getReduceContigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Get the offset in `out` for the reduction
  const IndexType outOffset =
    IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);

  // Get the base offset in `in` for this block's reduction
  const IndexType inBaseOffset =
    IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);

  // Each thread in the block will reduce some subset of elements in
  // the slice. The elements are guaranteed contiguous starting at
  // `inBaseOffset`.
  T r = init;
  for (IndexType i = threadIdx.x; i < reductionSize; i += blockDim.x) {
    r = reduceOp(r, modifyOp(in.data[inBaseOffset + i]));
  }

  // Reduce within the block
  // FIXME: extern name
  extern __shared__ char smemChar[];
  T* smem = (T*) smemChar;
  r = reduceBlock<T, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    out.data[outOffset] = r;
  }
}

inline dim3 getContigReduceBlock(ptrdiff_t numSlices, long reductionSize) {
  // If the number of slices is low but the reduction dimension size
  // is high, then we should increase block size for greater parallelism.
  // Aim for at least 32 warps per SM (assume 15 SMs; don't bother
  // inquiring the real number for now).
  int maxWarps = 4; // better occupancy if many blocks are around
  // For numSlices > 15 * 8, there are > 32 warps active per SM.
  if (numSlices < 15 * 8) {
    maxWarps = 8;
    if (numSlices < 15 * 4) {
      maxWarps = 16;
      if (numSlices < 15 * 2) {
        maxWarps = 32;
      }
    }
  }

  // Scale up block size based on the reduction dimension size
  long warpsInReductionSize = THCCeilDiv(reductionSize, 32L);
  int numWarps = warpsInReductionSize > (long) maxWarps ?
    maxWarps : (int) warpsInReductionSize;

  return dim3(numWarps * 32);
}

inline bool getContigReduceGrid(ptrdiff_t elements, dim3& grid) {
  // One output point per block
  return THC_getGridFromTiles(elements, grid);
}

// Performs a reduction out[..., 0, ...] = reduce_i(modify(in[..., i, ...])) for
// all in where i and the out's 0 are indexed at dimension `dim`
template <typename TensorType, typename ModifyOp, typename ReduceOp>
bool THC_reduceDim(THCState* state,
                   TensorType* out,
                   TensorType* in,
                   const ModifyOp& modifyOp,
                   const ReduceOp& reduceOp,
                   typename TensorUtils<TensorType>::DataType init,
                   int dim,
                   int keepdim) {
  ptrdiff_t inElements = TensorUtils<TensorType>::getNumElements(state, in);

  long reductionSize = TensorUtils<TensorType>::getSize(state, in, dim);
  long reductionStride = TensorUtils<TensorType>::getStride(state, in, dim);
  ptrdiff_t outElements = inElements / reductionSize;

  if (TensorUtils<TensorType>::getDims(state, out) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorType>::getDims(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorType>::getDims(state, in) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  // Is the reduction dimension contiguous? If so, then we can use a
  // shared memory reduction kernel to increase performance.
  bool contigReduction = (reductionStride == 1);

  dim3 block;
  dim3 grid;
  int smemSize = 0; // contiguous reduction uses smem
  if (contigReduction) {
    if (!getContigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getContigReduceBlock(outElements, reductionSize);
    smemSize = sizeof(typename TensorUtils<TensorType>::DataType) * block.x;
  } else {

    //If there are a large number of outputs to the reduction, avoid syncthreads
    //in kernel
    if(outElements > 512*16){
      block = dim3(512);
    }else{
      //x dim does different slices
      //y dim helps with a slice
      //If we only have 8 loops, don't bother sharing work across ydim
      unsigned long ydim = THCCeilDiv(reductionSize, 16L);

      ydim = min((unsigned long) 16, ydim);

      block = dim3(512, 1, 1);
      while(ydim > 1){
        block.x /= 2;
        block.y *= 2;
        ydim /= 2;
      }
    }
    long gridx  = THCCeilDiv( outElements, (long)block.x);
    if (gridx > 1024){
      long n_loops = THCCeilDiv(outElements, (long) (1024 * block.x) );
      gridx = outElements / (block.x*n_loops);
    }
    grid = dim3(gridx);

  }

  // Resize out to correspond to the reduced size
  THLongStorage* sizes = TensorUtils<TensorType>::newSizeOf(state, in);
  THLongStorage_set(sizes, dim, 1);
  TensorUtils<TensorType>::resize(state, out, sizes, NULL);
  THLongStorage_free(sizes);

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, OUT, IN)                                      \
  if (contigReduction) {                                                \
    kernelReduceContigDim<ModifyOp, ReduceOp,                           \
                          typename TensorUtils<TensorType>::DataType,   \
                          TYPE, OUT, IN>                                \
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>(    \
        outInfo, inInfo, reductionSize,                                 \
        (TYPE) outElements, init, modifyOp, reduceOp);                  \
  } else {                                                              \
    if(block.y > 1)                                                     \
      kernelReduceNoncontigDim_shared                                   \
        <ModifyOp, ReduceOp,                                            \
         typename TensorUtils<TensorType>::DataType,                    \
         TYPE, OUT, IN>                                                 \
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(         \
        outInfo, inInfo, reductionStride, reductionSize,                \
        (TYPE) outElements, init, modifyOp, reduceOp);                  \
    else                                                                \
      kernelReduceNoncontigDim                                          \
        <ModifyOp, ReduceOp,                                            \
         typename TensorUtils<TensorType>::DataType,                    \
         TYPE, OUT, IN>                                                 \
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(         \
        outInfo, inInfo, reductionStride, reductionSize,                \
        (TYPE) outElements, init, modifyOp, reduceOp);                  \
                                                                        \
  }                                                                     \


#define HANDLE_IN_CASE(TYPE, OUT, IN)                     \
  {                                                       \
    if (inInfo.isContiguous()) {                          \
      HANDLE_CASE(TYPE, OUT, -2);                         \
    } else {                                              \
      switch (IN) {                                       \
        case 1:                                           \
          HANDLE_CASE(TYPE, OUT, 1);                      \
          break;                                          \
        case 2:                                           \
          HANDLE_CASE(TYPE, OUT, 2);                      \
          break;                                          \
        default:                                          \
          HANDLE_CASE(TYPE, OUT, -1);                     \
          break;                                          \
      }                                                   \
    }                                                     \
  }

#define HANDLE_OUT_CASE(TYPE, OUT, IN)                 \
  {                                                    \
    if (outInfo.isContiguous()) {                      \
      HANDLE_IN_CASE(TYPE, -2, IN);                    \
    } else {                                           \
      switch (OUT) {                                   \
        case 1:                                        \
          HANDLE_IN_CASE(TYPE, 1, IN);                 \
          break;                                       \
        case 2:                                        \
          HANDLE_IN_CASE(TYPE, 2, IN);                 \
          break;                                       \
        default:                                       \
          HANDLE_IN_CASE(TYPE, -1, IN);                \
          break;                                       \
      }                                                \
    }                                                  \
  }

  if (TensorUtils<TensorType>::canUse32BitIndexMath(state, out) &&
      TensorUtils<TensorType>::canUse32BitIndexMath(state, in)) {
    TensorInfo<typename TensorUtils<TensorType>::DataType,
               unsigned int> outInfo =
      getTensorInfo<TensorType, unsigned int>(state, out);
    outInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorType>::DataType,
               unsigned int> inInfo =
      getTensorInfo<TensorType, unsigned int>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();

    HANDLE_OUT_CASE(unsigned int, outInfo.dims, inInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorType>::DataType,
               unsigned long> outInfo =
      getTensorInfo<TensorType, unsigned long>(state, out);
    outInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorType>::DataType,
               unsigned long> inInfo =
      getTensorInfo<TensorType, unsigned long>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (outInfo.isContiguous() && inInfo.isContiguous()) {
      HANDLE_CASE(unsigned long, -2, -2);
    } else {
      HANDLE_CASE(unsigned long, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE
#undef HANDLE_OUT_CASE


  if (!keepdim) {
    TensorUtils<TensorType>::squeeze1d(state, out, out, dim);
  }
  return true;
}

#undef LOCAL_MAX_BLOCK_SIZE

#endif // THC_REDUCE_INC
