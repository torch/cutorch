#ifndef THC_REDUCE_APPLY_UTILS_INC
#define THC_REDUCE_APPLY_UTILS_INC

#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCDeviceUtils.cuh"
#include "THCTensorInfo.cuh"

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };

template <typename IndexType>
__device__ __forceinline__ IndexType getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x +
    blockIdx.y * gridDim.x +
    blockIdx.x;
}

// Return value is in threadVals for threadIdx.x == 0
template <typename T, typename ReduceOp, int N>
__device__ void reduceNBlock(T *smem,
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

  if (threadIdx.x < numVals) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[threadIdx.x * N + i] = threadVals[i];
    }
  }
  __syncthreads();

  if ((threadIdx.x / warpSize) == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = threadIdx.x < numVals ? smem[threadIdx.x * N + i] : init;
    }

    for (int i = warpSize + threadIdx.x; i < numVals; i += warpSize) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        threadVals[j] = reduceOp(threadVals[j], smem[i * N + j]);
      }
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[threadIdx.x * N + i] = threadVals[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = smem[i];
    }

    int numLanesParticipating = min(numVals, warpSize);

    if (numLanesParticipating == 32) {
#pragma unroll
      for (int i = 1; i < 32; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
          threadVals[j] = reduceOp(threadVals[j], smem[i * N + j]);
        }
      }
    } else {
      for (int i = 1; i < numLanesParticipating; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
          threadVals[j] = reduceOp(threadVals[j], smem[i * N + j]);
        }
      }
    }
  }
}

// Block-wide reduction in shared memory helper; only threadIdx.x == 0 will
// return the reduced value
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  reduceNBlock<T, ReduceOp, 1>(smem, &threadVal, numVals, reduceOp, init);
  return threadVal;
}


// Block-wide reduction where each thread locally reduces N
// values before letting a single warp take over - assumes
// threadVals is in registers, not shared memory
template <typename T, typename ReduceOp, int N>
__device__ T reduceBlockN(T *smem,
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

  return reduceBlock<T, ReduceOp>(smem, blockDim.x < numVals ? blockDim.x : numVals, local, reduceOp, init);
}

// Make sure the given tensor doesn't have too many dimensions
void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg);

// Produces a grid with at least one point per tile
THC_API bool THC_getGridFromTiles(ptrdiff_t gridTiles, dim3& grid);

#endif // THC_REDUCE_APPLY_UTILS_INC
