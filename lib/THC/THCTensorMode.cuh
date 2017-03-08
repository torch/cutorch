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

template <typename T, int Power2Size>
__global__ void computeMode(
    TensorInfo<T, unsigned int> sorted,
    TensorInfo<long, unsigned int> positions,
    TensorInfo<T, unsigned int> values,
    TensorInfo<long, unsigned int> indices,
    long sliceSize)
{
  int tid = threadIdx.x;

  // First, we need to calculate the offset into the sorted Tensor that represents
  // the start of the slice for this block to calculate the mode for. This offset
  // is a combination of the gridIndices, and the number of elements in the slice
  unsigned int blockId = getLinearBlockId<unsigned int>();
  unsigned int linearOffset = blockId * sliceSize;

  /* __shared__ T         vsmem[1024]; */
  /* __shared__ unsigned int smem[1024]; */
  /* __shared__ bool      bmem[1024]; */

  T mode = ScalarConvert<int, T>::to(0);
  long index = -1;

  // (FOR NOW, ASSUME #threads = sliceSize and blockDim.x >= sliceSize)
  /* for (IndexType i = 0; i < sliceSize; i += blockDim.x) { */

    // Next we need to calculate the offset into the slice for this thread's
    // element.
    /* IndexType threadOffset = linearOffset + i + tid; */

    // Assumes contiguity
    /* IndexType offset = IndexToOffset<T, IndexType, -2>::get(threadOffset, sorted); */

    // Load slice value into per-block shared memory
    /* vsmem[tid] = sorted.data[offset]; */

    // The first step of our algorithm is performing a block-wide comparison of
    // neighboring elements. In particular, given an input slice A, we produce
    // an output slice B, such that B[i] = 1 if A[i-i] == A[i], otherwise 0.
    //
    // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
    //                 B = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    //
    // TODO: think about a better way to avoid this divergence
    /* if (tid == 0) { */
    /*   smem[tid] = 0; */
    /* } else { */
    /*   smem[tid] = THCNumerics<T>::eq(vsmem[tid - 1], vsmem[tid]); */
    /* } */
    /* bmem[tid] = false; */



    // Next, we perform a segmented prefix sum on the neighboring elements, where
    // the presence of a zero indicates the start of a segment.
    //
    // Input  = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    // Output = [0, 1, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0]
    //
    // Flag   = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    //           1  2  1  2  1  2  3  1  1  1  2  1  1]
    //
    // Operator: If my flag is 1, then 0
    //           Otherwise, I am 1 + prev
    //
    // Upsweep:
    //                                                         5
    //                                     4                   1
    //                     2               2               1   0
    //             1       1       1       1       0       1   0
    // Input = 0   1   0   1   0   1   1   0   0   0   1   0   0
    // Flag =  1   0   1   0   1   0   0   1   1   1   0   1   1
    //
    // Downsweep:
    //
    //         0

    // We do so by defining
    // the following binary operator:
    //
    // (+) If I am 0, then this is the start of a new sequence, and the output is 0
    //     Otherwise, I am
  /* } */

  // Now that we have the mode, and corresponding index, we need to calculate the output
  // position in the values/indices array. TODO: sinking suspicion this is wrong
  unsigned int outputOffset = IndexToOffset<T, unsigned int, -1>::get(blockId, values);
  values.data[outputOffset] = mode;
  indices.data[outputOffset] = index;
}

#endif // THC_TENSOR_MODE_CUH
