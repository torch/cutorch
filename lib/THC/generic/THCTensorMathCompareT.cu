#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathCompareT.cu"
#else

THC_API void
THCTensor_(ltTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorLTOp<typename TensorUtils<THCTensor>::DataType,
                    unsigned char>());
}

THC_API void
THCTensor_(gtTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorGTOp<typename TensorUtils<THCTensor>::DataType,
                    unsigned char>());
}

THC_API void
THCTensor_(leTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorLEOp<typename TensorUtils<THCTensor>::DataType,
                    unsigned char>());
}

THC_API void
THCTensor_(geTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorGEOp<typename TensorUtils<THCTensor>::DataType,
                    unsigned char>());
}

THC_API void
THCTensor_(eqTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorEQOp<typename TensorUtils<THCTensor>::DataType,
                    unsigned char>());
}

THC_API void
THCTensor_(neTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorNEOp<typename TensorUtils<THCTensor>::DataType,
                    unsigned char>());
}

THC_API void
THCTensor_(ltTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorLTOp<typename TensorUtils<THCTensor>::DataType,
                    typename TensorUtils<THCTensor>::DataType>());
}

THC_API void
THCTensor_(gtTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorGTOp<typename TensorUtils<THCTensor>::DataType,
                    typename TensorUtils<THCTensor>::DataType>());
}

THC_API void
THCTensor_(leTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorLEOp<typename TensorUtils<THCTensor>::DataType,
                    typename TensorUtils<THCTensor>::DataType>());
}

THC_API void
THCTensor_(geTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorGEOp<typename TensorUtils<THCTensor>::DataType,
                    typename TensorUtils<THCTensor>::DataType>());
}

THC_API void
THCTensor_(eqTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorEQOp<typename TensorUtils<THCTensor>::DataType,
                    typename TensorUtils<THCTensor>::DataType>());
}

THC_API void
THCTensor_(neTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor(state, self_, src1, src2,
                    TensorNEOp<typename TensorUtils<THCTensor>::DataType,
                    typename TensorUtils<THCTensor>::DataType>());
}

#endif
