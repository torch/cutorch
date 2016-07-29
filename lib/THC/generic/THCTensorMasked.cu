#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMasked.cu"
#else


THC_API void
THCTensor_(maskedFill)(THCState* state,
                       THCTensor *tensor, THCudaByteTensor *mask, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 2, tensor, mask));
  THArgCheck(THCTensor_(nElement)(state, tensor) ==
             THCudaByteTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THC_pointwiseApply2(state, tensor, mask,
                           TensorMaskedFillOp<real, unsigned char>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(maskedFillByte)(THCState* state,
                           THCTensor *tensor, THByteTensor *mask, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 1, tensor));
  THLongStorage* maskSizes = THByteTensor_newSizeOf(mask);
  THCudaByteTensor* maskCuda = THCudaByteTensor_newWithSize(state, maskSizes, NULL);
  THLongStorage_free(maskSizes);
  THCudaByteTensor_copyByte(state, maskCuda, mask);
  THCTensor_(maskedFill)(state, tensor, maskCuda, value);
  THCudaByteTensor_free(state, maskCuda);
}

THC_API void
THCTensor_(maskedCopy)(THCState* state,
                       THCTensor *tensor, THCudaByteTensor *mask, THCTensor *src)
{
  THAssert(THCTensor_(checkGPU)(state, 3, tensor, src, mask));
  long maskSize = THCudaByteTensor_nElement(state, mask);
  long tensorSize = THCTensor_(nElement)(state, tensor);
  long srcSize = THCTensor_(nElement)(state, src);

  // `mask` and `tensor` must have the same number of elements
  THArgCheck(maskSize == tensorSize, 2,
             "mask and tensor must have the same number of elements");

  // Determine our output size
  long totalElements = THCudaByteTensor_sumall(state, mask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (totalElements > srcSize) {
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  // FIXME: there appears to be a bug in Thrust (CUDA 7.0) for mixed
  // iterator prefix sums? Convert `mask` to the same datatype as what
  // we're accumulating the prefix sum in (long) to get around it
  THCudaLongTensor* maskLong = THCudaLongTensor_new(state);
  THLongStorage* maskSizes = THCudaByteTensor_newSizeOf(state, mask);
  THCudaLongTensor_resize(state, maskLong, maskSizes, NULL);
  THCudaLongTensor_copyCudaByte(state, maskLong, mask);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaLongTensor* maskPrefixSum = THCudaLongTensor_new(state);
  THCudaLongTensor_resize(state, maskPrefixSum, maskSizes, NULL);
  THLongStorage_free(maskSizes);

  thrust::device_ptr<long>
    maskData(THCudaLongTensor_data(state, maskLong));
  thrust::device_ptr<long>
    maskPrefixSumData(THCudaLongTensor_data(state, maskPrefixSum));

  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    maskData,
    maskData + THCudaLongTensor_nElement(state, maskLong),
    maskPrefixSumData);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THCTensor* contigSrc = THCTensor_(newContiguous)(state, src);

  // update `tensor` where `mask` == 1 but pull from `src` at
  // maskPrefixSum
  bool status = THC_pointwiseApply3(
    state, tensor, mask, maskPrefixSum,
    TensorMaskedCopyOp<real, unsigned char, long>(
      THCTensor_(data)(state, contigSrc)));

  THCTensor_(free)(state, contigSrc);
  THCudaLongTensor_free(state, maskLong);
  THCudaLongTensor_free(state, maskPrefixSum);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(maskedCopyByte)(THCState* state,
                           THCTensor *tensor, THByteTensor *mask, THCTensor *src) {
  THAssert(THCTensor_(checkGPU)(state, 2, tensor, src));
  THLongStorage* maskSizes = THByteTensor_newSizeOf(mask);
  THCudaByteTensor* maskCuda = THCudaByteTensor_newWithSize(state, maskSizes, NULL);
  THLongStorage_free(maskSizes);
  THCudaByteTensor_copyByte(state, maskCuda, mask);
  THCTensor_(maskedCopy)(state, tensor, maskCuda, src);
  THCudaByteTensor_free(state, maskCuda);
}

THC_API void
THCTensor_(maskedSelect)(THCState* state,
                         THCTensor* tensor, THCTensor* src, THCudaByteTensor* mask) {
  THAssert(THCTensor_(checkGPU)(state, 3, tensor, src, mask));
  THArgCheck(THCudaByteTensor_nElement(state, mask) ==
             THCTensor_(nElement)(state, src),
             2, "sizes do not match");

  // Determine our output size
  long totalElements = THCudaByteTensor_sumall(state, mask);
  THCTensor* tensorContig = THCTensor_(newContiguous)(state, tensor);

  THCTensor_(resize1d)(state, tensorContig, totalElements);
  if (tensor != tensorContig) {
    THCTensor_(resize1d)(state, tensor, totalElements);
  }

  // FIXME: there appears to be a bug in Thrust (CUDA 7.0) for mixed
  // iterator prefix sums? Convert `mask` to the same datatype as what
  // we're accumulating the prefix sum in (long) to get around it
  THCudaLongTensor* maskLong = THCudaLongTensor_new(state);
  THLongStorage* maskSizes = THCudaByteTensor_newSizeOf(state, mask);
  THCudaLongTensor_resize(state, maskLong, maskSizes, NULL);
  THCudaLongTensor_copyCudaByte(state, maskLong, mask);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaLongTensor* maskPrefixSum = THCudaLongTensor_new(state);
  THCudaLongTensor_resize(state, maskPrefixSum, maskSizes, NULL);
  THLongStorage_free(maskSizes);

  thrust::device_ptr<long>
    maskData(THCudaLongTensor_data(state, maskLong));
  thrust::device_ptr<long>
    maskPrefixSumData(THCudaLongTensor_data(state, maskPrefixSum));

  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    maskData,
    maskData + THCudaLongTensor_nElement(state, maskLong),
    maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THC_pointwiseApply3(
    state, mask, maskPrefixSum,
    src, TensorMaskedSelectOp<real, unsigned char, long>(
      THCTensor_(data)(state, tensor)));

  THCudaLongTensor_free(state, maskLong);
  THCudaLongTensor_free(state, maskPrefixSum);

  if (tensor != tensorContig) {
    THCTensor_(freeCopyTo)(state, tensorContig, tensor);
  } else {
    THCTensor_(free)(state, tensorContig);
  }

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

// FIXME: remove now that we have THCudaByteTensor?
THC_API void
THCTensor_(maskedSelectByte)(THCState* state,
                             THCTensor *tensor, THCTensor *src, THByteTensor *mask)
{
  THAssert(THCTensor_(checkGPU)(state, 2, tensor, src));
  THLongStorage* maskSizes = THByteTensor_newSizeOf(mask);
  THCudaByteTensor* maskCuda = THCudaByteTensor_newWithSize(state, maskSizes, NULL);
  THLongStorage_free(maskSizes);
  THCudaByteTensor_copyByte(state, maskCuda, mask);
  THCTensor_(maskedSelect)(state, tensor, src, maskCuda);
  THCudaByteTensor_free(state, maskCuda);
}

#endif
