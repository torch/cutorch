#include "THCTensorTypeUtils.cuh"
#include "THCTensor.h"
#include "THCTensorCopy.h"
#include "THCHalf.h"
#include <stdlib.h>

namespace {

struct SizeAndStride {
  long size;
  long stride;
};

int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;

  return aS->stride < bS->stride;
}

}

#define IMPL_TENSOR_UTILS(TENSOR_TYPE, DATA_TYPE)                       \
                                                                        \
TENSOR_TYPE*                                                            \
TensorUtils<TENSOR_TYPE>::newTensor(THCState* state) {                  \
  return TENSOR_TYPE##_new(state);                                      \
}                                                                       \
                                                                        \
TENSOR_TYPE*                                                            \
TensorUtils<TENSOR_TYPE>::newContiguous(THCState* state,                \
                                        TENSOR_TYPE* t) {               \
  return TENSOR_TYPE##_newContiguous(state, t);                         \
}                                                                       \
                                                                        \
THLongStorage*                                                          \
TensorUtils<TENSOR_TYPE>::newSizeOf(THCState* state,                    \
                                    TENSOR_TYPE* t) {                   \
  return TENSOR_TYPE##_newSizeOf(state, t);                             \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::retain(THCState* state,                       \
                                 TENSOR_TYPE* t) {                      \
  TENSOR_TYPE##_retain(state, t);                                       \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::free(THCState* state,                         \
                               TENSOR_TYPE* t) {                        \
  TENSOR_TYPE##_free(state, t);                                         \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::freeCopyTo(THCState* state,                   \
                                     TENSOR_TYPE* src,                  \
                                     TENSOR_TYPE* dst) {                \
  TENSOR_TYPE##_freeCopyTo(state, src, dst);                            \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::resize(THCState* state,                       \
                                 TENSOR_TYPE* out,                      \
                                 THLongStorage* sizes,                  \
                                 THLongStorage* strides) {              \
  TENSOR_TYPE##_resize(state, out, sizes, strides);                     \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::resizeAs(THCState* state,                     \
                                   TENSOR_TYPE* dst,                    \
                                   TENSOR_TYPE* src) {                  \
  TENSOR_TYPE##_resizeAs(state, dst, src);                              \
}                                                                       \
                                                                        \
DATA_TYPE*                                                              \
TensorUtils<TENSOR_TYPE>::getData(THCState* state,                      \
                                  TENSOR_TYPE* t) {                     \
  /* FIXME: no cast is required except for THCudaHalfTensor */          \
  return (DATA_TYPE*) TENSOR_TYPE##_data(state, t);                     \
}                                                                       \
                                                                        \
long                                                                    \
TensorUtils<TENSOR_TYPE>::getNumElements(THCState* state,               \
                                         TENSOR_TYPE* t) {              \
  return TENSOR_TYPE##_nElement(state, t);                              \
}                                                                       \
                                                                        \
long                                                                    \
TensorUtils<TENSOR_TYPE>::getSize(THCState* state,                      \
                                  TENSOR_TYPE* t,                       \
                                  int dim) {                            \
  return TENSOR_TYPE##_size(state, t, dim);                             \
}                                                                       \
                                                                        \
long                                                                    \
TensorUtils<TENSOR_TYPE>::getStride(THCState* state,                    \
                                    TENSOR_TYPE* t,                     \
                                    int dim) {                          \
  return TENSOR_TYPE##_stride(state, t, dim);                           \
}                                                                       \
                                                                        \
int                                                                     \
TensorUtils<TENSOR_TYPE>::getDims(THCState* state,                      \
                                  TENSOR_TYPE* t) {                     \
  return TENSOR_TYPE##_nDimension(state, t);                            \
}                                                                       \
                                                                        \
bool                                                                    \
TensorUtils<TENSOR_TYPE>::isContiguous(THCState* state,                 \
                                       TENSOR_TYPE* t) {                \
  return TENSOR_TYPE##_isContiguous(state, t);                          \
}                                                                       \
                                                                        \
int                                                                     \
TensorUtils<TENSOR_TYPE>::getDevice(THCState* state,                    \
                                    TENSOR_TYPE* t) {                   \
  return TENSOR_TYPE##_getDevice(state, t);                             \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::copyIgnoringOverlaps(THCState* state,         \
                                               TENSOR_TYPE* dst,        \
                                               TENSOR_TYPE* src) {      \
  return TENSOR_TYPE##_copyIgnoringOverlaps(state, dst, src);           \
}                                                                       \
                                                                        \
bool                                                                    \
TensorUtils<TENSOR_TYPE>::overlappingIndices(THCState* state,           \
                                             TENSOR_TYPE* t) {          \
  /* In this function, we don't care about permutations of the */       \
  /* size/stride arrays (transpositions). */                            \
  /* We order the size/stride arrays by stride, skipping dimensions of */ \
  /* size 1. Strides of dimensions of size 1 don't matter, since there */ \
  /* is only one addressing point in them. */                           \
  /* In this reordered view, the tensor is contiguous if */             \
  /* stride[dim] == size[dim + 1] * stride[dim + 1] for all `dim`. */   \
  /* The tensor has holes if */                                         \
  /* stride[dim] > size[dim + 1] * stride[dim + 1] for one or more */   \
  /* `dim`. */                                                          \
  /* The tensor has overlaps if */                                      \
  /* stride[dim] < size[dim + 1] * stride[dim + 1] for one or more */   \
  /* `dim`, or the innermost stride is 0. */                            \
                                                                        \
  /* Extract size/stride arrays; only consider size >1 dims. */         \
  SizeAndStride info[MAX_CUTORCH_DIMS];                                 \
                                                                        \
  int dims = TensorUtils<TENSOR_TYPE>::getDims(state, t);               \
  int nonSize1Dims = 0;                                                 \
  for (int i = 0; i < dims; ++i) {                                      \
    long size = TensorUtils<TENSOR_TYPE>::getSize(state, t, i);         \
    if (size > 1) {                                                     \
      info[nonSize1Dims].size = size;                                   \
      info[nonSize1Dims].stride =                                       \
        TensorUtils<TENSOR_TYPE>::getStride(state, t, i);               \
      ++nonSize1Dims;                                                   \
    }                                                                   \
  }                                                                     \
                                                                        \
  if (nonSize1Dims == 0) {                                              \
    /* no overlap */                                                    \
    return false;                                                       \
  }                                                                     \
                                                                        \
  /* Ascending order (innermost dimension in sorted view is at [0]) */  \
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride); \
                                                                        \
  /* Base case: innermost dimension must have stride >= 1 */            \
  if (info[nonSize1Dims - 1].stride < 1) {                              \
    return true;                                                        \
  }                                                                     \
                                                                        \
  /* Subsequent dimensions, if any */                                   \
  for (int i = nonSize1Dims - 2; i >= 0; --i) {                         \
    if (info[i].stride < info[i + 1].size * info[i + 1].stride) {       \
      /* There are overlaps */                                          \
      return true;                                                      \
    }                                                                   \
  }                                                                     \
                                                                        \
  /* Tensor has holes or is contiguous */                               \
  return false;                                                         \
}                                                                       \
                                                                        \
bool                                                                    \
TensorUtils<TENSOR_TYPE>::canUse32BitIndexMath(THCState* state,         \
                                               TENSOR_TYPE* t) {        \
  long elements = TensorUtils<TENSOR_TYPE>::getNumElements(state, t);   \
  if (elements >= UINT_MAX) {                                           \
    return false;                                                       \
  }                                                                     \
                                                                        \
  long offset = 0;                                                      \
  long linearId = elements - 1;                                         \
                                                                        \
  for (int i = TensorUtils<TENSOR_TYPE>::getDims(state, t) - 1; i >= 0; --i) { \
    long curDimIndex =                                                  \
      linearId % TensorUtils<TENSOR_TYPE>::getSize(state, t, i);        \
    long curDimOffset = curDimIndex *                                   \
      TensorUtils<TENSOR_TYPE>::getStride(state, t, i);                 \
    offset += curDimOffset;                                             \
    linearId /= TensorUtils<TENSOR_TYPE>::getSize(state, t, i);         \
  }                                                                     \
                                                                        \
  if (offset >= UINT_MAX) {                                             \
    return false;                                                       \
  }                                                                     \
                                                                        \
  return true;                                                          \
}

IMPL_TENSOR_UTILS(THCudaByteTensor, unsigned char)
IMPL_TENSOR_UTILS(THCudaCharTensor, char)
IMPL_TENSOR_UTILS(THCudaShortTensor, short)
IMPL_TENSOR_UTILS(THCudaIntTensor, int)
IMPL_TENSOR_UTILS(THCudaLongTensor, long)
IMPL_TENSOR_UTILS(THCudaTensor, float)
IMPL_TENSOR_UTILS(THCudaDoubleTensor, double)

#ifdef CUDA_HALF_TENSOR
IMPL_TENSOR_UTILS(THCudaHalfTensor, half);
#endif

#undef IMPL_TENSOR_UTILS
