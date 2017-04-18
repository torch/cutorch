#ifndef THC_REDUCE_APPLY_UTILS_INC
#define THC_REDUCE_APPLY_UTILS_INC

#include <algorithm>
#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCDeviceUtils.cuh"
#include "THCTensorInfo.cuh"
#include "THCAsmUtils.cuh"

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };

template <typename IndexType>
__device__ __forceinline__ IndexType getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x +
    blockIdx.y * gridDim.x +
    blockIdx.x;
}

// Returns the minimum size of shared memory required of performing N reductions
// across a block, with each reduction having at most numVals individual elements
// to reduce. Note that internally, because reductions operate (at the shared memory
// level) with N elements per thread in the block, so we have to use min(numvals,
// max block size) to determine this count.
template <typename T, int N>
int reduceSmemSize(THCState *state, long numVals) {
  // check if we can use a warp shuffle
  cudaDeviceProp *props = THCState_getCurrentDeviceProperties(state);
  if (props->major >= 3) {
    return props->warpSize * N * sizeof(T);
  } else {
    return THCRoundUp(std::min(numVals, (long) props->maxThreadsPerBlock), (long) props->warpSize) * N * sizeof(T);
  }
}

template <typename T>
struct THCWarpUtils {
  static __device__ __forceinline__ T shflxor(T val, unsigned int mask) {
    return __shfl_xor(val, mask);
  }
};

template <>
struct THCWarpUtils<unsigned char> {
  static __device__ __forceinline__ unsigned char shflxor(unsigned char val, unsigned int mask) {
    return (unsigned char) __shfl_xor((int) val, mask);
  }
};

template <>
struct THCWarpUtils<char> {
  static __device__ __forceinline__ char shflxor(char val, unsigned int mask) {
    return (char) __shfl_xor((int) val, mask);
  }
};

template <>
struct THCWarpUtils<short> {
  static __device__ __forceinline__ short shflxor(short val, unsigned int mask) {
    return (short) __shfl_xor((int) val, mask);
  }
};

template <>
struct THCWarpUtils<double> {
  static __device__ __forceinline__ double shflxor(double val, unsigned int mask) {
    int2 a = *reinterpret_cast<int2*>(&val);
    a.x = __shfl_xor(a.x, mask);
    a.y = __shfl_xor(a.y, mask);
    return *reinterpret_cast<double*>(&a);
  }
};

template <>
struct THCWarpUtils<long> {
  static __device__ __forceinline__ long shflxor(long val, unsigned int mask) {
    int2 a = *reinterpret_cast<int2*>(&val);
    a.x = __shfl_xor(a.x, mask);
    a.y = __shfl_xor(a.y, mask);
    return *reinterpret_cast<long*>(&a);
  }
};

template <typename T, typename ReduceOp, int N>
__device__ void warpReduce(T threadVals[N], ReduceOp reduceOp) {
#pragma unroll
  for (int mask = 1; mask < warpSize; mask *= 2) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      T neighbor = THCWarpUtils<T>::shflxor(threadVals[i], mask);
      threadVals[i] = reduceOp(threadVals[i], neighbor);
    }
  }
}

template <typename T, typename ReduceOp, int N>
__device__ void warpReduceBlock(T *smem, T threadVals[N], int numVals, ReduceOp reduceOp, T init) {
  assert(blockDim.x % warpSize == 0);
  // First, warps cooperate to reduce values within the warp
  warpReduce<T, ReduceOp, N>(threadVals, reduceOp);
  int lane = getLaneId();
  int warp = threadIdx.x / warpSize;

  if (lane == 0) {

#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[warp + (i * warpSize)] = threadVals[i];
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < N; ++i) {
    threadVals[i] = (threadIdx.x < (blockDim.x / warpSize)) ? smem[lane + (i * warpSize)] : init;
  }

  if (warp == 0) {
    warpReduce<T, ReduceOp, N>(threadVals, reduceOp);
  }
}

// Reduce N values concurrently, i.e. suppose N = 2, and there are 4 threads:
// (1, 2), (3, 4), (5, 6), (7, 8), then the return in threadVals for thread 0
// is (1 + 3 + 5 + 7, 2 + 4 + 6 + 8) = (16, 20)
template <typename T, typename ReduceOp, int N>
__device__ void reduceNValuesInBlock(T *smem,
                             T threadVals[N],
                             int numVals,
                             ReduceOp reduceOp,
                             T init) {
  if (numVals == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = init;
    }
    return;
  }

#if __CUDA_ARCH__ >= 300
  warpReduceBlock<T, ReduceOp, N>(smem, threadVals, numVals, reduceOp, init);
#else
  // We store each of the N values contiguously, so if N = 2, all values for
  // the first threadVal for each thread in the block are stored followed by
  // all of the values for the second threadVal for each thread in the block
  if (threadIdx.x < numVals) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numVals + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  // Number of lanes in the final reduction --> this is used to determine
  // where to put the outputs of each of the n things we are reducing. If
  // nLP = 32, then we have the 32 outputs for the first threadVal,
  // followed by the 32 outputs for the second threadVal, etc.
  int numLanesParticipating = min(numVals, warpSize);

  if (numVals > warpSize && ((threadIdx.x / warpSize) == 0 )) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = threadIdx.x < numVals ? threadVals[i] : init;
    }

    for (int i = warpSize + threadIdx.x; i < numVals; i += warpSize) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        threadVals[j] = reduceOp(threadVals[j], smem[j * numVals + i]);
      }
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numLanesParticipating + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (numLanesParticipating == 32) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 1; j < 32; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * 32 + j]);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        for (int j = 1; j < numLanesParticipating; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * numVals + j]);
        }
      }
    }
  }
#endif
}

// Block-wide reduction in shared memory helper; only threadIdx.x == 0 will
// return the reduced value
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  reduceNValuesInBlock<T, ReduceOp, 1>(smem, &threadVal, numVals, reduceOp, init);
  return threadVal;
}

// Block-wide reduction where each thread locally reduces N
// values before letting a single warp take over - assumes
// threadVals is in registers, not shared memory. Note that
// numVals in this case is the number of values in the overall
// reduction, i.e. if there are 512 threads with N=2, and say
// there are 768 elements in the input block, then numVals is 768,
// not, say, 384 (i.e. 768 / N=2)
template <typename T, typename ReduceOp, int N>
__device__ T reduceBlockWithNThreadLocalReductions(T *smem,
                         T threadVals[N],
                         int numVals,
                         ReduceOp reduceOp,
                         T init) {
  int offset = threadIdx.x * N;
  T local = offset < numVals ? threadVals[0] : init;

#pragma unroll
  for (int i = 1; i < N; ++i) {
    ++offset;
    T next = offset < numVals ? threadVals[i] : init;
    local = reduceOp(local, next);
  }

  return reduceBlock<T, ReduceOp>(smem, THCCeilDiv(numVals, N), local, reduceOp, init);
}

// Make sure the given tensor doesn't have too many dimensions
void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg);

// Produces a grid with at least one point per tile
THC_API bool THC_getGridFromTiles(ptrdiff_t gridTiles, dim3& grid);

#endif // THC_REDUCE_APPLY_UTILS_INC
