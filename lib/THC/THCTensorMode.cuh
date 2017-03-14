#ifndef THC_TENSOR_MODE_CUH
#define THC_TENSOR_MODE_CUH

#include "THCNumerics.cuh"
#include "THCSortUtils.cuh"

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

template <>
struct BinaryAddOp<unsigned int> {
  __host__ __device__ inline unsigned int operator()(const unsigned int a, const unsigned int b) {
    return a + b;
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

// The mode kernel has the following characteristics: It uses internal shared memory
// buffers of Power2Size, which must be greater than the number of elements. Additionally,
// there is one block for every slice to calculate the mode for, and in each block there
// is one thread for every two elements.
//
// Both sorted and positions are assumed to be contiguous Tensors with the mode dimension
// as the innermost dim, such that we can get the particular slice for a Tensor via its
// linear block dimension * the slice size.
template <typename T, unsigned int Power2Size>
__global__ void computeMode(
    T *input,
    TensorInfo<T, unsigned int> values,
    TensorInfo<long, unsigned int> indices,
    long sliceSize)
{
  int tidx = threadIdx.x;
  int stidx = blockDim.x + threadIdx.x; // Second Index this thread responsible for

  // First, we need to calculate the offset into the sorted Tensor that represents
  // the start of the slice for this block to calculate the mode for. This offset
  // is a combination of the gridIndices, and the number of elements in the slice.
  unsigned int blockId = getLinearBlockId<unsigned int>();
  unsigned int linearOffset = blockId * sliceSize;

  // The smem buffer is used to store the elements from the slice
  __shared__ T    smem[Power2Size];

  // Each thread loads up to two elements from the Tensor into shared memory
  if (tidx < sliceSize) {
    smem[tidx] = input[linearOffset + tidx];
  }
  if (stidx < sliceSize) {
    smem[stidx] = input[linearOffset + stidx];
  }

  // The fmem buffer is used in multiple components of the kernel. First, it stores
  // whether bmem[i] = i < sliceSize to mark the valid components in the smem buffer
  // for sorting
  __shared__ bool bmem[Power2Size];

  bmem[tidx] = tidx < sliceSize;
  bmem[stidx] = stidx < sliceSize;
  __syncthreads(); // barrier for smem, bmem initialization

  if (threadIdx.x == 0) {
    /* printf("smem (before sort): "); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", smem[i]); */
    /* } */
    /* printf("\n"); */
  }

  // First, sort the input slice in ascending order. smem contains the input
  // elements, and bmem marks the valid indices
  bitonicSortKeys<LTComp<T>, T, unsigned int, Power2Size>(smem, bmem, LTComp<T>());
  __syncthreads(); // make no assumptions that the sort syncs at end

  if (threadIdx.x == 0) {
    /* printf("smem (after sort): "); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", smem[i]); */
    /* } */
    /* printf("\n"); */
  }

  // The next step of our algorithm is performing a block-wide comparison of
  // neighboring elements. In particular, given an sorted input slice A, we
  // produce an output slice B, such that B[i] = 1 if A[i-i] != A[i], otherwise 0.
  //
  // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
  //                 B = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  //
  // We re-use the bmem buffer for this computation. In particular, we can think of
  // B[i] = true indicating the start of a sequence of equal values in the sorted
  // list.

  if (tidx == 0) {
    bmem[tidx] = true;  // for setting element 0
  }

  // Compares elements (0, 1), (2, 3), ... and sets 1, 3, ...
  bmem[tidx * 2 + 1] = THCNumerics<T>::ne(smem[tidx * 2], smem[tidx * 2 + 1]); // (0, 1), (1, 2), etc.

  // Compares elements (1, 2), (3, 4), ... and sets 2, 4, ...
  if (((tidx + 1) * 2) < Power2Size) {
    bmem[(tidx + 1) * 2] = THCNumerics<T>::ne(smem[((tidx + 1) * 2) - 1], smem[(tidx + 1) * 2]);
  }
  __syncthreads(); // barrier for bmem initialization

  // Next, we initialize another shared memory buffer cmem which will be used in the
  // Segmented Prefix Sum
  __shared__ unsigned int  cmem[Power2Size];

  // We set cmem to be the negation of bmem. In particular, we can think of cmem[i] = true
  // iff A[i-1] == A[i] in our original sorted slice.
  cmem[tidx] = !bmem[tidx];
  cmem[stidx] = !bmem[stidx];
  __syncthreads(); // barrier for cmem initialization

  if (threadIdx.x == 0) {
    /* printf("(before scan): cmem:"); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", cmem[i]); */
    /* } */
    /* printf("\n"); */
    /* printf("(before scan): bmem:"); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", bmem[i]); */
    /* } */
    /* printf("\n"); */
  }

  // Next, we perform a segmented prefix sum on the neighboring elements, where
  // the presence of a one indicates the start of a segment. In this case bmem acts
  // as the segment start flags, and cmem is the buffer to be summed:
  //
  // Input  (cmem)  = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  // Flag   (bmem)  = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  // Output (cmem)  = [0, 1, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0]
  //
  // Afterwards, the cmem buffer contains the lengths of the segments (minus 1), i.e. the counts
  // of each element in the original input.
  segmentedInclusivePrefixScan<unsigned int, BinaryAddOp<unsigned int>, Power2Size>(cmem, bmem, BinaryAddOp<unsigned int>());

  // Our last shared memory buffer is used to track indices
  __shared__ unsigned int imem[Power2Size];

  // initialize the indices buffer such that imem[i] = i
  imem[tidx] = tidx;
  imem[stidx] = stidx;
  __syncthreads(); // barrier for both the scan and the imem initialization

  if (threadIdx.x == 0) {
    /* printf("(after scan): cmem:"); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", cmem[i]); */
    /* } */
    /* printf("\n"); */
    /* printf("(after scan): imem:"); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", imem[i]); */
    /* } */
    /* printf("\n"); */
  }

  // At this point, we need to find the maximum element in the cmem buffer.
  // This element will represent the count (-1) of the mode. Because of the
  // way we have set up the problem, the index where this mode occurs will
  // also be the location of the mode value in the sorted array, e.g.
  //
  // smem = [0, 0, 1, 1, 1, 2]
  // cmem = [0, 1, 0, 1, 2, 0]
  //                     ^
  //                     maximum value, also aligned with mode = 1
  //
  // We perform a block wide max-reduction of the cmem buffer, and bring imem
  // along with it.
  //
  // Loop 1 (Power2Size = offset = 4):
  //
  // (0, 4) --> cmem[4] = 2 > cmem[0] = 0, so update cmem[0], imem[0]
  // (1, 5) --> cmem[5] = 0 <= cmem[1] = 1, do nothing
  //
  // Now:         0  1  2  3  4  5
  //      cmem = [2, 1, 0, 1, 2, 0]
  //      imem = [4, 1, 2, 3, 4, 5]
  //
  // Loop 2 (offset = 2)
  //
  // (0, 2) --> cmem[2] == 0 <= cmem[0] = 2, do nothing
  // (1, 3) --> cmem[3] == 1 <= cmem[1] = 1, do nothing
  //
  // Now:         0  1  2  3  4  5
  //      cmem = [2, 1, 2, 1, 2, 0]
  //      imem = [4, 1, 4, 3, 4, 5]
  //
  // Loop 3 (offset = 1)
  //
  // (0, 1) --> cmem[1] == 1 <= cmem[0] = 2, do nothing
  //
  // So at the end at cmem[0] we have the maximum count = 2, and the
  // corresponding index = 4
#pragma unroll
  for (unsigned int offset = Power2Size / 2; offset > 0; offset >>= 1) {
    if (tidx < offset && tidx + offset < sliceSize) {
      // Note that we could do >= as well. We use >, so that we pick the
      // earliest maximum value in the sequence in case of ties. This will
      // result in picking the smallest value for the mode, i.e. if both
      // 3 and 4 occur the same number of times in the input, and their count
      // is the mode, then we return 3. This mimics the behavior of CPU-Torch
      if (cmem[tidx + offset] > cmem[tidx]) {
        cmem[tidx] = cmem[tidx + offset];
        imem[tidx] = imem[tidx + offset];
      }
    }
    __syncthreads();
  }

  /* if (threadIdx.x == 0) { */
  /*   printf("(after max reduction): cmem:"); */
  /*   for (int i = 0; i < sliceSize; ++i) { */
  /*     printf(" %d", cmem[i]); */
  /*   } */
  /*   printf("\n"); */
  /*   printf("(after max reduction): imem:"); */
  /*   for (int i = 0; i < sliceSize; ++i) { */
  /*     printf(" %d", imem[i]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  // Store the mode in shared memory for use in finding the mode in the input slice
  __shared__ T  mode;

  // Given the above constraints, the mode is the value at the maximum index in the segmented scan
  if (tidx == 0) {
    mode = smem[imem[0]];
    /* printf("imem[0] = %d, cmem[0] = %d, mode: %d, index: %ld\n", imem[0], cmem[0], mode, index); */
  }
  __syncthreads(); // broadcast mode

  // Finally, we need to find the "an" index of the mode in the input Tensor. The API does
  // not constrain which index we pick, so it can be any of the indices that contain the mode.
  // We will do a reduction to find the index. First, we mark indices that are equal to the mode,
  // i.e bmem[i] = true if input[i] == mode
  if (tidx < sliceSize) {
    bmem[tidx] = THCNumerics<T>::eq(input[linearOffset + tidx], mode);
    imem[tidx] = tidx;
  }
  if (stidx < sliceSize) {
    bmem[stidx] = THCNumerics<T>::eq(input[linearOffset + stidx], mode);
    imem[stidx] = stidx;
  }
  __syncthreads(); // barrier for initialization of bmem, imem

  if (tidx == 0) {
    /* printf("(in preparation for min index reduction): (imem, bmem):"); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" (%d, %d)", imem[i], bmem[i]); */
    /* } */
    /* printf("\n"); */
    /* printf("(in preparation for min index reduction): imem:"); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", imem[i]); */
    /* } */
    /* printf("\n"); */
  }

  // Then we perform a similar reduction to the one above, except this time we update
  // the element if the element at the base position is not equal to the mode and
  // the element at the offset position is. At the end, imem[0] will contain an index
  // with the mode.
  for (unsigned int offset = Power2Size / 2; offset > 0; offset >>= 1) {
    if (tidx < offset && tidx + offset < sliceSize) {
      /* printf("thread %d comparing (%d, %d)\n", tidx, tidx, tidx + offset); */
      /* if (!bmem[tidx] && bmem[tidx + offset]) { */

      // Just always update the base if the offset is true
      if (bmem[tidx + offset]) {
        imem[tidx] = imem[tidx + offset];
        bmem[tidx] = true;  // need to update match
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    /* printf("(after min index reduction): imem:"); */
    /* for (int i = 0; i < sliceSize; ++i) { */
    /*   printf(" %d", imem[i]); */
    /* } */
    /* printf("\n"); */
  }

  // Finally, we have the mode, and an index where it occurs. We use a single thread
  // to place this in the appropriate output position
  if (tidx == 0) {
    long index = TH_INDEX_BASE + imem[0];

    // Now that we have the mode, and corresponding index, we need to calculate the output
    // position in the values/indices array and store the data
    unsigned int outputOffset = IndexToOffset<T, unsigned int, -1>::get(blockId, values);
    /* printf("outputOffset: %d\n", outputOffset); */
    values.data[outputOffset] = mode;
    indices.data[outputOffset] = index;
  }
}

#endif // THC_TENSOR_MODE_CUH
