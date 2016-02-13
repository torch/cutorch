#ifndef THC_APPLY_INC
#define THC_APPLY_INC

#include "THCTensorCopy.h"
#include "THCReduceApplyUtils.cuh"

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CUTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

// Threads per block for our apply kernel
#define THC_APPLY_THREADS_PER_BLOCK 32 * 16

#ifndef COMMA
#define COMMA ,
#endif

#define THC_APPLY_EXPAND(left, varname, right) \
  switch( varname ) { \
    case -2: \
      left -2 right; \
      break; \
    case 1: \
      left 1 right; \
      break; \
    case 2: \
      left 2 right; \
      break; \
    case 3: \
      left 3 right; \
      break; \
    default: \
      left -1 right; \
      break; \
  }

// Called when we are copying into an overlapping index `dst`, but
// we don't care which writer wins. Hacky but it works.
THC_API void THCudaTensor_copyIgnoringOverlaps(THCState* state,
                                       THCudaTensor* dst,
                                       THCudaTensor* src);

template <typename Op, typename IndexType, int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THCudaTensor_pointwiseApply1(TensorInfo<IndexType> a,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    op(&a.data[aOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THCudaTensor_pointwiseApply2(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    op(&a.data[aOffset], &b.data[bOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims, int CDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THCudaTensor_pointwiseApply3(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<IndexType, CDims>::get(linearIndex, c);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset]);
  }
}

inline dim3 getApplyBlock() {
  return dim3(THC_APPLY_THREADS_PER_BLOCK);
}

inline bool getApplyGrid(THCState* state, long totalElements, dim3& grid) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);

  if (curDevice == -1) {
    return false;
  }

  // Assume a reasonable number of SMs if no state is available
  int numSM =
    state ? THCState_getCurrentDeviceProperties(state)->multiProcessorCount : 15;

  // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
  // which seems to be a good sweetspot for latency hiding
  grid = dim3(min((long long) THCCeilDiv(totalElements,
                                         (long) THC_APPLY_THREADS_PER_BLOCK),
                  4LL * numSM));
  return true;
}

// Apply 1 ============

template <typename Op, typename IndexType, int A>
static void THCudaTensor_pointwiseApply1_launch(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  IndexType totalElements,
                                  const Op& op) {
  THCudaTensor_pointwiseApply1<Op, IndexType, A>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    aInfo, totalElements, op);
}

template <typename Op, typename IndexType>
static void THCudaTensor_pointwiseApply1_expanda(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  IndexType totalElements, const Op& op, int a) {
  THC_APPLY_EXPAND(
    THCudaTensor_pointwiseApply1_launch<Op COMMA IndexType COMMA,
    a,
    >(grid, block, state, aInfo, totalElements, op); )
}

template <typename Op>
bool THCudaTensor_pointwiseApply1(THCState* state,
                                  THCudaTensor* a,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite) {
  long totalElements = THCudaTensor_nElement(state, a);

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THCudaTensor* oldA = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (THC_canUse32BitIndexMath(state, a)) {
    TensorInfo<unsigned int> aInfo(state, a);
    aInfo.collapseDims();
    int a = aInfo.dims;
    if(aInfo.isContiguous()) a = -2;
    THCudaTensor_pointwiseApply1_expanda(
      grid, block, state, aInfo, (unsigned int)totalElements, op, a);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    aInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous()) {
      THCudaTensor_pointwiseApply1<Op, unsigned long, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);
    } else {
      THCudaTensor_pointwiseApply1<Op, unsigned long, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);
    }
  }

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  return true;
}

// Apply 2 ============

template <typename Op, typename IndexType, int A, int B>
static void THCudaTensor_pointwiseApply2_launch(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  TensorInfo<IndexType> bInfo,
                                  IndexType totalElements,
                                  const Op& op) {
  THCudaTensor_pointwiseApply2<Op, IndexType, A, B>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    aInfo, bInfo, totalElements, op);
}

template <typename Op, typename IndexType, int A>
static void THCudaTensor_pointwiseApply2_expandb(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  TensorInfo<IndexType> bInfo,
                                  IndexType totalElements, const Op& op, int b) {
  THC_APPLY_EXPAND(
    THCudaTensor_pointwiseApply2_launch<Op COMMA IndexType COMMA A COMMA,
    b,
    >(grid, block, state, aInfo, bInfo, totalElements, op); )
}

template <typename Op, typename IndexType>
static void THCudaTensor_pointwiseApply2_expanda(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  TensorInfo<IndexType> bInfo,
                                  IndexType totalElements, const Op& op, int a, int b) {
  THC_APPLY_EXPAND(
    THCudaTensor_pointwiseApply2_expandb<Op COMMA IndexType COMMA,
    a,
    >(grid, block, state, aInfo, bInfo, totalElements, op, b); )
}

template <typename Op>
bool THCudaTensor_pointwiseApply2(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly) {
  long totalElements = THCudaTensor_nElement(state, a);

  if (totalElements != THCudaTensor_nElement(state, b)) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, b) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THCudaTensor* oldA = NULL;
  THCudaTensor* oldB = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }
  if (bType == ReadWrite && THC_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THCudaTensor_newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

  if (THC_canUse32BitIndexMath(state, a) &&
      THC_canUse32BitIndexMath(state, b)) {
    TensorInfo<unsigned int> aInfo(state, a);
    aInfo.collapseDims();

    TensorInfo<unsigned int> bInfo(state, b);
    bInfo.collapseDims();

    int a = aInfo.dims;
    int b = bInfo.dims;

    if(aInfo.isContiguous()) a = -2;
    if(bInfo.isContiguous()) b = -2;

    THCudaTensor_pointwiseApply2_expanda(
      grid, block, state, aInfo, bInfo, (unsigned int)totalElements, op, a, b);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    aInfo.collapseDims();

    TensorInfo<unsigned long> bInfo(state, b);
    bInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      THCudaTensor_pointwiseApply2<Op, unsigned long, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, (unsigned long) totalElements, op);
    } else {
      THCudaTensor_pointwiseApply2<Op, unsigned long, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, (unsigned long) totalElements, op);
    }
  }

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldB, b);
    THCudaTensor_free(state, b);
    b = oldB;
  }

  return true;
}

// Apply 3 ============

template <typename Op, typename IndexType, int A, int B, int C>
static void THCudaTensor_pointwiseApply3_launch(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  TensorInfo<IndexType> bInfo,
                                  TensorInfo<IndexType> cInfo,
                                  IndexType totalElements,
                                  const Op& op) {
  THCudaTensor_pointwiseApply3<Op, IndexType, A, B, C>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    aInfo, bInfo, cInfo, totalElements, op);
}

template <typename Op, typename IndexType, int A, int B>
static void THCudaTensor_pointwiseApply3_expandc(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  TensorInfo<IndexType> bInfo,
                                  TensorInfo<IndexType> cInfo,
                                  IndexType totalElements, const Op& op, int c) {
  THC_APPLY_EXPAND(
    THCudaTensor_pointwiseApply3_launch<Op COMMA IndexType COMMA A COMMA B COMMA,
    c,
    >(grid, block, state, aInfo, bInfo, cInfo, totalElements, op); )
}

template <typename Op, typename IndexType, int A>
static void THCudaTensor_pointwiseApply3_expandb(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  TensorInfo<IndexType> bInfo,
                                  TensorInfo<IndexType> cInfo,
                                  IndexType totalElements, const Op& op, int b, int c) {
  THC_APPLY_EXPAND(
    THCudaTensor_pointwiseApply3_expandc<Op COMMA IndexType COMMA A COMMA,
    b,
    >(grid, block, state, aInfo, bInfo, cInfo, totalElements, op, c); )
}

template <typename Op, typename IndexType>
static void THCudaTensor_pointwiseApply3_expanda(const dim3 grid, const dim3 block, THCState *state,
                                  TensorInfo<IndexType> aInfo,
                                  TensorInfo<IndexType> bInfo,
                                  TensorInfo<IndexType> cInfo,
                                  IndexType totalElements, const Op& op, int a, int b, int c) {
  THC_APPLY_EXPAND(
    THCudaTensor_pointwiseApply3_expandb<Op COMMA IndexType COMMA,
    a,
    >(grid, block, state, aInfo, bInfo, cInfo, totalElements, op, b, c); )
}

template <typename Op>
bool THCudaTensor_pointwiseApply3(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  THCudaTensor* c,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly,
                                  TensorArgType cType = ReadOnly) {
  long totalElements = THCudaTensor_nElement(state, a);

  if (totalElements != THCudaTensor_nElement(state, b) ||
      totalElements != THCudaTensor_nElement(state, c)) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, b) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, c) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THCudaTensor* oldA = NULL;
  THCudaTensor* oldB = NULL;
  THCudaTensor* oldC = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }

  if (bType == ReadWrite && THC_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THCudaTensor_newContiguous(state, b);
  }

  if (cType == ReadWrite && THC_overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = THCudaTensor_newContiguous(state, c);
  }

  if (THC_canUse32BitIndexMath(state, a) &&
      THC_canUse32BitIndexMath(state, b) &&
      THC_canUse32BitIndexMath(state, c)) {
    TensorInfo<unsigned int> aInfo(state, a);
    aInfo.collapseDims();

    TensorInfo<unsigned int> bInfo(state, b);
    bInfo.collapseDims();

    TensorInfo<unsigned int> cInfo(state, c);
    cInfo.collapseDims();

    int a = aInfo.dims;
    int b = bInfo.dims;
    int c = cInfo.dims;

    if(aInfo.isContiguous()) a = -2;
    if(bInfo.isContiguous()) b = -2;
    if(cInfo.isContiguous()) c = -2;

    THCudaTensor_pointwiseApply3_expanda(
      grid, block, state, aInfo, bInfo, cInfo, (unsigned int)totalElements, op, a, b, c);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    aInfo.collapseDims();

    TensorInfo<unsigned long> bInfo(state, b);
    bInfo.collapseDims();

    TensorInfo<unsigned long> cInfo(state, c);
    cInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      THCudaTensor_pointwiseApply3<Op, unsigned long, -2, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    } else {
      THCudaTensor_pointwiseApply3<Op, unsigned long, -1, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    }
  }

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldB, b);
    THCudaTensor_free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldC, c);
    THCudaTensor_free(state, c);
    c = oldC;
  }

  return true;
}

#undef THC_APPLY_EXPAND
#undef THC_APPLY_THREADS_PER_BLOCK

#endif // THC_APPLY_INC
