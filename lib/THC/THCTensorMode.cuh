#ifndef THC_TENSOR_MODE_CUH
#define THC_TENSOR_MODE_CUH

#include "THCNumerics.cuh"

struct ThrustHalfLess
{
  __host__ __device__ inline bool operator()(const half& lhs, const half& rhs) {
    return THCNumerics<half>::lt(lhs, rhs);
  }
};

struct ThrustHalfNotEqualTo
{
  __host__ __device__ inline bool operator()(const half& lhs, const half& rhs) {
    return THCNumerics<half>::ne(lhs, rhs);
  }
};

struct ThrustHalfEqualTo
{
  __host__ __device__ inline bool operator()(const half& lhs, const half& rhs) {
    return THCNumerics<half>::eq(lhs, rhs);
  }
};

struct ThrustHalfEqualToPredicate
{
  ThrustHalfEqualToPredicate(half val): val_(val) {}
  __host__ __device__ inline bool operator()(half x) {
    return THCNumerics<half>::eq(val_, x);
  }

  half val_;
};

template <typename T>
struct BinaryAddOp {
  __host__ __device__ inline T operator()(const T a, const T b) {
    return THCNumerics<T>::add(a, b);
  }
};

template <typename T, class BinaryOp, int Power2ScanSize>
__device__ void segmentedInclusivePrefixScan(T *smem, bool *bmem, BinaryOp binop) {
  // Reduce step ("upsweep")
#pragma unroll
  for (int stride = 1; stride < Power2ScanSize; stride <<= 1) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < Power2ScanSize) {
      smem[index] = bmem[index] ? smem[index] : binop(smem[index], smem[index - stride]);
      bmem[index] = bmem[index] | bmem[index - stride];
    }
    __syncthreads();
  }

  // Post-reduce step ("downsweep")
#pragma unroll
  for (int stride = Power2ScanSize / 4; stride > 0; stride >>= 1) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < Power2ScanSize) {
      smem[index + stride] = bmem[index + stride] ? smem[index + stride] : binop(smem[index + stride], smem[index]);
      bmem[index + stride] = bmem[index + stride] | bmem[index];
    }
    __syncthreads();
  }
}

template <typename T, int Power2Size>
__global__ void computeMode(
    TensorInfo<T, unsigned int> sorted,
    TensorInfo<long, unsigned int> positions,
    TensorInfo<T, unsigned int> values,
    TensorInfo<long, unsigned int> indices,
    long sliceSize)
{
  int tidx = threadIdx.x;

  // First, we need to calculate the offset into the sorted Tensor that represents
  // the start of the slice for this block to calculate the mode for. This offset
  // is a combination of the gridIndices, and the number of elements in the slice
  unsigned int blockId = getLinearBlockId<unsigned int>();
  unsigned int linearOffset = blockId * sliceSize;

  __shared__ T smem[Power2Size];
  __shared__ bool bmem[Power2Size];
  __shared__ int  cmem[Power2Size];

  /* __shared__ unsigned int smem[1024]; */
  /* __shared__ bool      bmem[1024]; */

  T mode = ScalarConvert<int, T>::to(0);
  int maxCount = -1;
  long index = -1;

  // Each thread loads up to two elements from the Tensor into shared memory
  if (tidx * 2 < sliceSize) {
    smem[tidx * 2] = sorted.data[IndexToOffset<T, unsigned int, -2>::get(linearOffset + (tidx * 2), sorted)];
  }
  if (tidx * 2 + 1 < sliceSize) {
    smem[tidx * 2 + 1] = sorted.data[IndexToOffset<T, unsigned int, -2>::get(linearOffset + (tidx * 2) + 1, sorted)];
  }
  __syncthreads();

  // The first step of our algorithm is performing a block-wide comparison of
  // neighboring elements. In particular, given an input slice A, we produce
  // an output slice B, such that B[i] = 1 if A[i-i] == A[i], otherwise 0.
  //
  // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
  //                 B = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  //
  bmem[tidx] = true;  // for setting element 0
  bmem[tidx * 2 + 1] = THCNumerics<T>::ne(smem[tidx * 2], smem[tidx * 2 + 1]); // (0, 1), (1, 2), etc.
  if (((tidx + 1) * 2) < Power2Size) {
    bmem[(tidx + 1) * 2] = THCNumerics<T>::ne(smem[(tidx + 1) * 2 - 1], smem[(tidx + 1) * 2]);
  }
  __syncthreads();

  cmem[tidx * 2] = !bmem[tidx * 2];
  cmem[tidx * 2 + 1] = !bmem[tidx * 2 + 1];

  /* if (threadIdx.x == 0) { */
  /*   printf("smem:"); */
  /*   for (int i = 0; i < sliceSize; ++i) { */
  /*     printf(" %d", smem[i]); */
  /*   } */
  /*   printf("\n"); */
  /*   printf("bmem:"); */
  /*   for (int i = 0; i < sliceSize; ++i) { */
  /*     printf(" %d", bmem[i]); */
  /*   } */
  /*   printf("\n"); */
  /*   printf("cmem:"); */
  /*   for (int i = 0; i < sliceSize; ++i) { */
  /*     printf(" %d", cmem[i]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  __syncthreads();

  // Next, we perform a segmented prefix sum on the neighboring elements, where
  // the presence of a zero indicates the start of a segment.
  //
  // Input  = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  // Output = [0, 1, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0]
  //
  // Flag   = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  //           1  2  1  2  1  2  3  1  1  1  2  1  1]
  segmentedInclusivePrefixScan<int, BinaryAddOp<int>, Power2Size>(cmem, bmem, BinaryAddOp<int>());
  /* if (threadIdx.x == 0) { */
  /*   printf("bmem:"); */
  /*   for (int i = 0; i < sliceSize; ++i) { */
  /*     printf(" %d", bmem[i]); */
  /*   } */
  /*   printf("\n"); */
  /*   printf("cmem:"); */
  /*   for (int i = 0; i < sliceSize; ++i) { */
  /*     printf(" %d", cmem[i]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  // Now we reduce to find the maximum value, which represents the count of the mode
  if (threadIdx.x == 0) {
    for (int i = 0; i < sliceSize; ++i) {
      if (cmem[i] > maxCount) {
        maxCount = cmem[i];
        index = i;
      }
    }
    mode = smem[index];
    index -= maxCount;
  }
  __syncthreads();

  // Now that we have the mode, and corresponding index, we need to calculate the output
  // position in the values/indices array. TODO: sinking suspicion this is wrong
  if (threadIdx.x == 0) {
    unsigned int outputOffset = IndexToOffset<T, unsigned int, -1>::get(blockId, values);
    values.data[outputOffset] = mode;
    indices.data[outputOffset] = index + TH_INDEX_BASE;
  }
}

#endif // THC_TENSOR_MODE_CUH
