#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorTopK.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_BYTE) || defined(THC_REAL_IS_CHAR) || defined(THC_REAL_IS_SHORT) || defined(THC_REAL_IS_INT)

/* Returns the set of all kth smallest (or largest) elements, depending */
/* on `dir` */
THC_API void THCTensor_(topk)(THCState* state,
                               THCTensor* topK,
                               THCudaLongTensor* indices,
                               THCTensor* input,
                               long k, int dim, int dir, int sorted);

#endif // THC_REAL_IS_FLOAT
#endif // THC_GENERIC_FILE
