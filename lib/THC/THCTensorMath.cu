#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorRandom.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#define NB_THREADS_PER_BLOCK 256

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

void THCudaTensor_fill(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  thrust::fill(self_data, self_data+THCudaTensor_nElement(self), value);

  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_zero(THCudaTensor *self_)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaCheck(cudaMemset(THCudaTensor_data(self), 0, sizeof(float)*THCudaTensor_nElement(self)));
  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_zeros(THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_zero(r_);
}

void THCudaTensor_ones(THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_fill(r_, 1);
}

void THCudaTensor_reshape(THCudaTensor *r_, THCudaTensor *t, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_copy(r_, t);
}

long THCudaTensor_numel(THCudaTensor *t)
{
  return THCudaTensor_nElement(t);
}


struct addvalue_functor
{
  const float value;

  addvalue_functor(float value_) : value(value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x+value);
  }
};

void THCudaTensor_add(THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THCudaTensor_resizeAs(self_, src_);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaTensor *src = THCudaTensor_newContiguous(src_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, addvalue_functor(value));

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

struct mulvalue_functor
{
  const float value;
  mulvalue_functor(float value_) : value(value_) {}
    __host__ __device__ float operator()(const float& x) const
  {
    return (x*value);
  }
};

void THCudaTensor_mul(THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THCudaTensor_resizeAs(self_, src_);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaTensor *src = THCudaTensor_newContiguous(src_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, mulvalue_functor(value));

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

struct divvalue_functor
{
  const float value;
  divvalue_functor(float value_) : value(value_) {}
    __host__ __device__ float operator()(const float& x) const
  {
    return (x/value);
  }
};

void THCudaTensor_div(THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THCudaTensor_resizeAs(self_, src_);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaTensor *src = THCudaTensor_newContiguous(src_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, divvalue_functor(value));

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_cadd(struct THCudaBlasState* blas_state, THCudaTensor *self_, THCudaTensor* src1, float value, THCudaTensor *src2)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);

    if (self_ != src1) {
      src1 = THCudaTensor_newContiguous(src1);
      THCudaTensor_copy(self, src1);
      THCudaTensor_free(src1);
    }

    src2 = THCudaTensor_newContiguous(src2);

    THCudaBlas_axpy(blas_state, THCudaTensor_nElement(self), value, THCudaTensor_data(src2), 1, THCudaTensor_data(self), 1);

    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_cmul(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);
    thrust::device_ptr<float> self_data(THCudaTensor_data(self));
    thrust::device_ptr<float> src1_data(THCudaTensor_data(src1));
    thrust::device_ptr<float> src2_data(THCudaTensor_data(src2));

    thrust::transform(src2_data, src2_data+size, src1_data, self_data, thrust::multiplies<float>());

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_cdiv(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size does not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);
    thrust::device_ptr<float> self_data(THCudaTensor_data(self));
    thrust::device_ptr<float> src1_data(THCudaTensor_data(src1));
    thrust::device_ptr<float> src2_data(THCudaTensor_data(src2));

    thrust::transform(src1_data, src1_data+size, src2_data, self_data, thrust::divides<float>());

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_addcmul(float *data, float value, float *src1, float *src2, long size)
{
  long k = (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;

  if(k < size)
    data[k] += value*src1[k]*src2[k];
}


void THCudaTensor_addcmul(THCudaTensor *self_, THCudaTensor *t, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  if(self_ != t)
  {
    THCudaTensor_resizeAs(self_, t);
    THCudaTensor_copy(self_, t);
  }
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);

    int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
    THCudaGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);
    dim3 threads(nThreadPerBlock);
    dim3 grid(nBlockPerRow, nBlockPerColumn);

    THCudaTensor_kernel_addcmul<<<grid, threads>>>(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size);

    cudaError errcode = cudaGetLastError();
    if(errcode != cudaSuccess)
      THError(cudaGetErrorString(errcode));

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_addcdiv(float *data, float value, float *src1, float *src2, long size)
{
  long k = (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;

  if(k < size)
    data[k] += value*src1[k]/src2[k];
}


void THCudaTensor_addcdiv(THCudaTensor *self_, THCudaTensor *t, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  if(self_ != t)
  {
    THCudaTensor_resizeAs(self_, t);
    THCudaTensor_copy(self_, t);
  }

  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);

    int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
    THCudaGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);
    dim3 threads(nThreadPerBlock);
    dim3 grid(nBlockPerRow, nBlockPerColumn);

    THCudaTensor_kernel_addcdiv<<<grid, threads>>>(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size);

    cudaError errcode = cudaGetLastError();
    if(errcode != cudaSuccess)
      THError(cudaGetErrorString(errcode));

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

float THCudaTensor_dot(struct THCudaBlasState* blas_state, THCudaTensor *self, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self) == THCudaTensor_nElement(src), 2, "size do not match");

  {
    self = THCudaTensor_newContiguous(self);
    src = THCudaTensor_newContiguous(src);

    float result = THCudaBlas_dot(blas_state, THCudaTensor_nElement(self),
                                  THCudaTensor_data(self), 1,
                                  THCudaTensor_data(src), 1);
    THCudaTensor_free(src);
    THCudaTensor_free(self);

    return result;
  }
}

float THCudaTensor_minall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(self), (float)(THInf), thrust::minimum<float>());

  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_maxall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(self), (float)(-THInf), thrust::maximum<float>());

  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_sumall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(self), (float)(0), thrust::plus<float>());

  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_prodall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(self), (float)(1), thrust::multiplies<float>());

  THCudaTensor_free(self);
  return result;
}



struct dim4 {
    unsigned arr[4];

    __host__ dim4(unsigned init=0) {
        for(unsigned i=0; i<4; i++) { arr[i] = init; }
    }

    __host__ __device__ unsigned& operator[](const unsigned& idx) { return arr[idx]; }
};



/* Reduce one of the outer dimensions of a tensor
 *
 * For an n-d tensor (n <= 4) where the reduction is *not* along the innermost
 * dimension:
 *
 * - block.x and grid.x make up the innermost dimension;
 * - The reduced dimension is looped over inside a block; and
 * - grid.y and grid.z are the remaining two dimensions (if any).
 * - block.y and block.z are not used as we're limited to 512 or 1024 threads
 *   in the block.
 *
 * For sizes/strides, index 3 is the reduced dimension, while the remaining
 * indices are for the remaining dimensions with index 0 the innermost dimension.
 *
 * Reduction along the innermost dimension is handled in a separate kernel.
 */
template<class UnaryFunction, class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceOuterDim(float *tgt, float *src_,
        dim4 src_stride, dim4 tgt_stride, dim4 size,
        UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  const size_t reduce = 3;

  for(unsigned z = blockIdx.z; z < size[2] ; z += gridDim.z)
  for(unsigned y = blockIdx.y; y < size[1] ; y += gridDim.y)
  for(unsigned col = blockIdx.x * blockDim.x + threadIdx.x; col < size[0]; col += blockDim.x * gridDim.x) {
    float *src = src_ + z * src_stride[2] + y * src_stride[1] + col;
    float acc = init;
    for(unsigned i=0; i < size[reduce]; i++) {
      acc = binary_op(acc, unary_op(*src));
      src += src_stride[reduce];
    }
    tgt[z * tgt_stride[2] + y * tgt_stride[1] + col] = float(acc);
  }
}



template<class UnaryFunction, class BinaryFunction>
__host__ void THCudaTensor_transformReduceOuterDim(THCudaTensor *tgt, THCudaTensor *src,
        long rdim, UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  const size_t reduce = 3;
  dim4 src_stride(0);
  dim4 tgt_stride(0);
  dim4 size(1);

  unsigned ndim = THCudaTensor_nDimension(src);
  for(unsigned idim=0, o=ndim-2; idim < ndim; idim++) {
    unsigned odim = idim == rdim ? reduce : o--;
    src_stride[odim] = THCudaTensor_stride(src, idim);
    tgt_stride[odim] = THCudaTensor_stride(tgt, idim);
    size[odim]       = THCudaTensor_size(src, idim);
  }

  const unsigned nThreadPerBlock = 256;
  unsigned nBlockPerColumn = DIVUP(size[0], nThreadPerBlock);
  dim3 threads(nThreadPerBlock);
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  dim3 grid(min(maxGridDim, nBlockPerColumn), min(maxGridDim, size[1]), min(maxGridDim, size[2]));

  THCudaTensor_kernel_transformReduceOuterDim<<<grid, threads>>>(THCudaTensor_data(tgt),
          THCudaTensor_data(src), src_stride, tgt_stride, size, unary_op, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}



/* Reduce the innermost dimension of a tensor
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */
template<class UnaryFunction, class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceInnermostDim(float *tgt, float *src_,
        dim4 src_stride, dim4 tgt_stride, dim4 size, UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  __shared__ float sbuf[16][32]; // 8kB

  for(unsigned z = blockIdx.z; z < size[3] ; z += gridDim.z)
  for(unsigned x = blockIdx.x; x < size[2] ; x += gridDim.x)
  for(unsigned bRow = blockIdx.y * blockDim.y; bRow < size[1]; bRow += blockDim.y * gridDim.y) {

    float acc = init;
    unsigned row = bRow + threadIdx.y;
    float *src = src_ + z * src_stride[3] + x * src_stride[2] + row * src_stride[1];
    bool reducing = threadIdx.x < blockDim.y && bRow + threadIdx.x < size[1] && threadIdx.y == 0;

    for(unsigned bCol=0; bCol < size[0]; bCol += blockDim.x) {

      sbuf[threadIdx.y][threadIdx.x] = init;
      unsigned col = bCol + threadIdx.x;
      if(row < size[1] && col < size[0]) {
        sbuf[threadIdx.y][threadIdx.x] = unary_op(src[col]);
      }
      __syncthreads();

      float* line = &sbuf[threadIdx.y][0];
      for(unsigned s = 16; s > 1; s >>= 1) {
        if(row < size[1] && threadIdx.x < s) {
          line[threadIdx.x] = binary_op(line[threadIdx.x], line[threadIdx.x + s]);
        }
        __syncthreads();
      }
      if(reducing) {
        sbuf[threadIdx.x][0] = binary_op(sbuf[threadIdx.x][0], sbuf[threadIdx.x][1]);
        acc = binary_op(acc, sbuf[threadIdx.x][0]);
      }
      __syncthreads();
    }

    if(reducing) {
      unsigned row = bRow + threadIdx.x;
      unsigned tgt_offset = z * tgt_stride[3] + x * tgt_stride[2];
      tgt[tgt_offset + row] = acc;
    }
  }
}

template<class UnaryFunction, class BinaryFunction>
__host__ void THCudaTensor_transformReduceInnermostDim(THCudaTensor *tgt, THCudaTensor *src,
        UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  dim4 src_stride(0);
  dim4 tgt_stride(0);
  dim4 size(1);

  unsigned ndim = THCudaTensor_nDimension(src);
  for(unsigned dim=0; dim < ndim; dim++) {
    unsigned odim = ndim - 1 - dim;
    src_stride[odim] = THCudaTensor_stride(src, dim);
    tgt_stride[odim] = THCudaTensor_stride(tgt, dim);
    size[odim]       = THCudaTensor_size(src, dim);
  }

  dim3 threads(32, 16);
  unsigned nBlockPerRow = DIVUP(size[1], threads.y);
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  dim3 grid(min(maxGridDim, size[2]), min(maxGridDim, nBlockPerRow), min(maxGridDim, size[3]));

  THCudaTensor_kernel_transformReduceInnermostDim<<<grid, threads>>>(THCudaTensor_data(tgt),
          THCudaTensor_data(src), src_stride, tgt_stride, size, unary_op, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}


template<class UnaryFunction, class BinaryFunction>
void THCudaTensor_transformReduceDim(THCudaTensor *self_, THCudaTensor *src,
        long dimension, UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(src), 3, "dimension out of range");
  THArgCheck(THCudaTensor_nDimension(src) <= 4, 2, "too many dimensions (>4)");

  THLongStorage *dim = THCudaTensor_newSizeOf(src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(self_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);

  if(dimension == THCudaTensor_nDimension(src)-1) {
    THCudaTensor_transformReduceInnermostDim(self, src, unary_op, init, binary_op);
  } else {
    THCudaTensor_transformReduceOuterDim(self, src, dimension, unary_op, init, binary_op);
  }

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}


template<class BinaryFunction>
void THCudaTensor_reduceDim(THCudaTensor *self_, THCudaTensor *src, long dimension, float init, BinaryFunction binary_op)
{
  THCudaTensor_transformReduceDim(self_, src, dimension, thrust::identity<float>(), init, binary_op);
}


void THCudaTensor_sum(THCudaTensor *self, THCudaTensor *src, long dimension)
{
  return THCudaTensor_reduceDim(self, src, dimension, 0.0f, thrust::plus<float>());
}

void THCudaTensor_prod(THCudaTensor *self, THCudaTensor *src, long dimension)
{
  return THCudaTensor_reduceDim(self, src, dimension, 1.0f, thrust::multiplies<float>());
}

/* a set of reduction kernels that take in Binary ops on thrust pairs (of value, index)
   These are useful when you not only have to do a reduction, but you might have
   to preserve the location of contention (for example min/max operations)
 */
template<class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceOuterDimIndex(float *tgt1, float *tgt2,
                                                             float *src_,
                                                             dim4 src_stride,
                                                             dim4 tgt1_stride,
                                                             dim4 tgt2_stride,
                                                             dim4 size,
                                                             thrust::pair<float,float> init,
                                                             BinaryFunction binary_op)
{
  const size_t reduce = 3;

  for(unsigned z = blockIdx.z; z < size[2] ; z += gridDim.z)
  for(unsigned y = blockIdx.y; y < size[1] ; y += gridDim.y)
  for(unsigned col = blockIdx.x * blockDim.x + threadIdx.x; col < size[0]; col += blockDim.x * gridDim.x) {
    float *src = src_ + z * src_stride[2] + y * src_stride[1] + col;
    thrust::pair<float,float> acc = init;
    for(unsigned i=0; i < size[reduce]; i++) {
      acc = binary_op(thrust::make_pair(*src, i+1), acc); // i+1 for 1-indexing
      src += src_stride[reduce];
    }
    tgt1[z * tgt1_stride[2] + y * tgt1_stride[1] + col] = acc.first;
    tgt2[z * tgt2_stride[2] + y * tgt2_stride[1] + col] = acc.second;
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_transformReduceOuterDimIndex(THCudaTensor *tgt1, THCudaTensor *tgt2,
                                                   THCudaTensor *src,
                                                   long rdim, thrust::pair<float,float> init,
                                                   BinaryFunction binary_op)
{
  const size_t reduce = 3;
  dim4 src_stride(0);
  dim4 tgt1_stride(0);
  dim4 tgt2_stride(0);
  dim4 size(1);

  unsigned ndim = THCudaTensor_nDimension(src);
  for(unsigned idim=0, o=ndim-2; idim < ndim; idim++) {
    unsigned odim = idim == rdim ? reduce : o--;
    src_stride[odim] = THCudaTensor_stride(src, idim);
    tgt1_stride[odim] = THCudaTensor_stride(tgt1, idim);
    tgt2_stride[odim] = THCudaTensor_stride(tgt2, idim);
    size[odim]       = THCudaTensor_size(src, idim);
  }

  const unsigned nThreadPerBlock = 256;
  unsigned nBlockPerColumn = DIVUP(size[0], nThreadPerBlock);
  dim3 threads(nThreadPerBlock);
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  dim3 grid(min(maxGridDim, nBlockPerColumn), min(maxGridDim, size[1]), min(maxGridDim, size[2]));

  THCudaTensor_kernel_transformReduceOuterDimIndex<<<grid, threads>>>(
    THCudaTensor_data(tgt1), THCudaTensor_data(tgt2),
    THCudaTensor_data(src), src_stride, tgt1_stride, tgt2_stride, size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

/* Reduce the innermost dimension of a tensor (on thrust::pair functors which are (value, index))
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */
template<class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceInnermostDimIndex(
  float *tgt1, float* tgt2, float *src_,
  dim4 src_stride, dim4 tgt1_stride, dim4 tgt2_stride,
  dim4 size, thrust::pair<float,float> init, BinaryFunction binary_op)
{
  __shared__ float sbuf[16][32]; // 8kB
  __shared__ float ibuf[16][32]; // 8kB

  for(unsigned z = blockIdx.z; z < size[3] ; z += gridDim.z)
  for(unsigned x = blockIdx.x; x < size[2] ; x += gridDim.x)
  for(unsigned bRow = blockIdx.y * blockDim.y; bRow < size[1]; bRow += blockDim.y * gridDim.y) {

    thrust::pair<float,float> acc = init;
    unsigned row = bRow + threadIdx.y;
    float *src = src_ + z * src_stride[3] + x * src_stride[2] + row * src_stride[1];
    bool reducing = threadIdx.x < blockDim.y && bRow + threadIdx.x < size[1] && threadIdx.y == 0;

    for(unsigned bCol=0; bCol < size[0]; bCol += blockDim.x) {

      sbuf[threadIdx.y][threadIdx.x] = init.first;
      ibuf[threadIdx.y][threadIdx.x] = init.second;
      unsigned col = bCol + threadIdx.x;
      if(row < size[1] && col < size[0]) {
        sbuf[threadIdx.y][threadIdx.x] = src[col];
        ibuf[threadIdx.y][threadIdx.x] = col+1; // +1 for 1-indexing
      }
      __syncthreads();

      float* sline = &sbuf[threadIdx.y][0];
      float* iline = &ibuf[threadIdx.y][0];
      for(unsigned s = 16; s > 1; s >>= 1) {
        if(row < size[1] && threadIdx.x < s) {
          thrust::pair<float,float> arg1 = thrust::make_pair<float,float>(sline[threadIdx.x], iline[threadIdx.x]);
          thrust::pair<float,float> arg2 = thrust::make_pair<float,float>(sline[threadIdx.x + s], iline[threadIdx.x + s]);
          thrust::pair<float,float> res = binary_op(arg1, arg2);
          sline[threadIdx.x] = res.first;
          iline[threadIdx.x] = res.second;
        }
        __syncthreads();
      }
      if(reducing) {
        thrust::pair<float,float> res = binary_op(thrust::make_pair<float,float>(sbuf[threadIdx.x][0], ibuf[threadIdx.x][0]),
                                            thrust::make_pair<float,float>(sbuf[threadIdx.x][1], ibuf[threadIdx.x][1]));
        sbuf[threadIdx.x][0] = res.first;
        ibuf[threadIdx.x][0] = res.second;
        acc = binary_op(acc, res);
      }
      __syncthreads();
    }

    if(reducing) {
      unsigned row = bRow + threadIdx.x;
      unsigned tgt1_offset = z * tgt1_stride[3] + x * tgt1_stride[2];
      unsigned tgt2_offset = z * tgt2_stride[3] + x * tgt2_stride[2];
      tgt1[tgt1_offset + row] = acc.first;
      tgt2[tgt2_offset + row] = acc.second;
    }
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_transformReduceInnermostDimIndex(
  THCudaTensor *tgt1, THCudaTensor *tgt2, THCudaTensor *src,
  thrust::pair<float,float> init, BinaryFunction binary_op)
{
  dim4 src_stride(0);
  dim4 tgt1_stride(0);
  dim4 tgt2_stride(0);
  dim4 size(1);

  unsigned ndim = THCudaTensor_nDimension(src);
  for(unsigned dim=0; dim < ndim; dim++) {
    unsigned odim = ndim - 1 - dim;
    src_stride[odim] = THCudaTensor_stride(src, dim);
    tgt1_stride[odim] = THCudaTensor_stride(tgt1, dim);
    tgt2_stride[odim] = THCudaTensor_stride(tgt2, dim);
    size[odim]       = THCudaTensor_size(src, dim);
  }

  dim3 threads(32, 16);
  unsigned nBlockPerRow = DIVUP(size[1], threads.y);
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  dim3 grid(min(maxGridDim, size[2]), min(maxGridDim, nBlockPerRow), min(maxGridDim, size[3]));

  THCudaTensor_kernel_transformReduceInnermostDimIndex<<<grid, threads>>>(
    THCudaTensor_data(tgt1), THCudaTensor_data(tgt2),
    THCudaTensor_data(src), src_stride, tgt1_stride, tgt2_stride, size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

template<class BinaryFunction>
void THCudaTensor_reduceDimIndex(THCudaTensor *tgt1_, THCudaTensor *tgt2_, THCudaTensor *src,
                             long dimension, thrust::pair<float,float> init,
                                     BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(src), 3, "dimension out of range");
  THArgCheck(THCudaTensor_nDimension(src) <= 4, 2, "too many dimensions (>4)");

  THLongStorage *dim = THCudaTensor_newSizeOf(src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(tgt1_, dim, NULL);
  THCudaTensor_resize(tgt2_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *tgt1 = THCudaTensor_newContiguous(tgt1_);
  THCudaTensor *tgt2 = THCudaTensor_newContiguous(tgt2_);
  src = THCudaTensor_newContiguous(src);

  if(dimension == THCudaTensor_nDimension(src)-1) {
    THCudaTensor_transformReduceInnermostDimIndex(tgt1, tgt2, src, init, binary_op);
  } else {
    THCudaTensor_transformReduceOuterDimIndex(tgt1, tgt2, src, dimension, init, binary_op);
  }

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(tgt1, tgt1_);
  THCudaTensor_freeCopyTo(tgt2, tgt2_);
}

struct maxvalue_functor
{
  __host__ __device__ thrust::pair<float,float> operator()(const thrust::pair<float,float> &a,
                                                            const thrust::pair<float,float> &b)
  {
    if (a.first > b.first) return a;
    else return b;
  }
};

void THCudaTensor_max(THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dimension)
{
  const float minfloat32 = -3.402823466e+38f;
  thrust::pair<float,float> init = thrust::make_pair<float,float>(minfloat32, -1);
  return THCudaTensor_reduceDimIndex(values, indices, src, dimension, init,
                                 maxvalue_functor());
}

struct minvalue_functor
{
  __host__ __device__ thrust::pair<float,float> operator()(const thrust::pair<float,float> &a,
                                                            const thrust::pair<float,float> &b)
  {
    if (a.first < b.first) return a;
    else return b;
  }
};

void THCudaTensor_min(THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dimension)
{
  const float maxfloat32 = 3.402823466e+38f;
  thrust::pair<float,float> init = thrust::make_pair<float,float>(maxfloat32, -1);
  return THCudaTensor_reduceDimIndex(values, indices, src, dimension, init,
                                     minvalue_functor());
}


void THCudaTensor_addmv(struct THCudaBlasState* blas_state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(r_, t);
    THCudaTensor_copy(r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THCudaBlas_gemv(blas_state, 'n', mat->size[0], mat->size[1],
                    alpha, THCudaTensor_data(mat), mat->stride[1],
                    THCudaTensor_data(vec), vec->stride[0],
                    beta, THCudaTensor_data(r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THCudaBlas_gemv(blas_state, 't',  mat->size[1], mat->size[0],
                    alpha, THCudaTensor_data(mat), mat->stride[0],
                    THCudaTensor_data(vec), vec->stride[0],
                    beta, THCudaTensor_data(r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cmat = THCudaTensor_newContiguous(mat);

    THCudaBlas_gemv(blas_state, 't',  mat->size[1], mat->size[0],
                    alpha, THCudaTensor_data(cmat), cmat->stride[0],
                    THCudaTensor_data(vec), vec->stride[0],
                    beta, THCudaTensor_data(r_), r_->stride[0]);

    THCudaTensor_free(cmat);
  }
}

void THCudaTensor_addmm(struct THCudaBlasState* blas_state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *m1, THCudaTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THCudaTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if(t != r_)
  {
    THCudaTensor_resizeAs(r_, t);
    THCudaTensor_copy(r_, t);
  }

  /* r_ */
  if(r_->stride[0] == 1)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1)
  {
    THCudaTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    r__ = THCudaTensor_newWithSize2d(r_->size[1], r_->size[0]);
    THCudaTensor_copy(r__, r_);
    THCudaTensor_transpose(r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THCudaTensor_newContiguous(m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THCudaTensor_newContiguous(m2);
  }

  /* do the operation */
  THCudaBlas_gemm(blas_state, transpose_m1,
                  transpose_m2,
                  r__->size[(transpose_r == 'n' ? 0 : 1)],
                  r__->size[(transpose_r == 'n' ? 1 : 0)],
                  m1_->size[(transpose_r == 'n' ? 1 : 0)],
                  alpha,
                  THCudaTensor_data(m1_),
                  (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  THCudaTensor_data(m2_),
                  (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  beta,
                  THCudaTensor_data(r__),
                  r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THCudaTensor_free(m1_);

  if(m2_ != m2)
    THCudaTensor_free(m2_);

  if(r__ != r_)
    THCudaTensor_freeCopyTo(r__, r_);
}

void THCudaTensor_addr(struct THCudaBlasState* blas_state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(r_, t);
    THCudaTensor_copy(r_, t);
  }

  if(beta != 1)
    THCudaTensor_mul(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THCudaBlas_ger(blas_state, vec1->size[0], vec2->size[0],
                   alpha, THCudaTensor_data(vec1), vec1->stride[0],
                   THCudaTensor_data(vec2), vec2->stride[0],
                   THCudaTensor_data(r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THCudaBlas_ger(blas_state, vec2->size[0], vec1->size[0],
                   alpha, THCudaTensor_data(vec2), vec2->stride[0],
                   THCudaTensor_data(vec1), vec1->stride[0],
                   THCudaTensor_data(r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cr = THCudaTensor_newClone(r_);

    THCudaBlas_ger(blas_state, vec2->size[0], vec1->size[0],
                   alpha, THCudaTensor_data(vec2), vec2->stride[0],
                   THCudaTensor_data(vec1), vec1->stride[0],
                   THCudaTensor_data(cr), cr->stride[0]);

    THCudaTensor_freeCopyTo(cr, r_);
  }
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                        \
  struct NAME##_functor                                                      \
  {                                                                          \
    __host__ __device__ float operator()(const float& x) const               \
    {                                                                        \
      return CFUNC(x);                                                       \
    }                                                                        \
  };                                                                         \
                                                                             \
  void THCudaTensor_##NAME(THCudaTensor *self_, THCudaTensor *src)           \
  {                                                                          \
    THCudaTensor_resizeAs(self_, src);                                       \
    THCudaTensor *self = THCudaTensor_newContiguous(self_);                  \
    src = THCudaTensor_newContiguous(src);                                   \
    long size = THCudaTensor_nElement(self);                                 \
    thrust::device_ptr<float> self_data(THCudaTensor_data(self));            \
    thrust::device_ptr<float> src_data(THCudaTensor_data(src));              \
                                                                             \
    thrust::transform(src_data, src_data+size, self_data, NAME##_functor()); \
                                                                             \
    THCudaTensor_free(src);                                                  \
    THCudaTensor_freeCopyTo(self, self_);                                    \
  }

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, fabs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, roundf)

struct pow_functor
{
  const float value;

  pow_functor(float value_) : value(value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return pow(x, value);
  }
};

void THCudaTensor_pow(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_resizeAs(self_, src);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, pow_functor(value));

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

struct atan2_functor
{
  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      return atan2f(x, y);
  }
};

void THCudaTensor_atan2(THCudaTensor *self_, THCudaTensor *tx, THCudaTensor *ty)
{
  THCudaTensor_resizeAs(self_, tx);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  tx = THCudaTensor_newContiguous(tx);
  ty = THCudaTensor_newContiguous(ty);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> tx_data(THCudaTensor_data(tx));
  thrust::device_ptr<float> ty_data(THCudaTensor_data(ty));

  thrust::transform(tx_data, tx_data+size, ty_data, self_data, atan2_functor());

  THCudaTensor_free(tx);
  THCudaTensor_free(ty);
  THCudaTensor_freeCopyTo(self, self_);
}


struct clamp_functor
{
  const float min_value;
  const float max_value;

  clamp_functor(float min_value_, float max_value_) : min_value(min_value_), max_value(max_value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    if (x < min_value) {
      return min_value;
    }
    if (x > max_value) {
      return max_value;
    }
    return x;
  }
};

void THCudaTensor_clamp(THCudaTensor *self_, THCudaTensor *src, float min_value,
  float max_value)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 2, "sizes do not match");
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, clamp_functor(min_value,
    max_value));

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}


struct sign_functor
{
  __device__ float operator()(const float &v) const {
    return (v > 0) - (v < 0);
  }
};


void THCudaTensor_sign(THCudaTensor *self_, THCudaTensor *src)
{
  THCudaTensor_resizeAs(self_, src);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  src = THCudaTensor_newContiguous(src);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, sign_functor());

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

float THCudaTensor_meanall(THCudaTensor *self)
{
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THCudaTensor_sumall(self)/THCudaTensor_nElement(self);
}

void
THCudaTensor_mean(THCudaTensor *self, THCudaTensor *src, long dim)
{
  THCudaTensor_sum(self, src, dim);
  THCudaTensor_div(self, self, THCudaTensor_size(src, dim));
}

struct square_functor
{
  const float mean;

  square_functor(float mean_) : mean(mean_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x-mean)*(x-mean);
  }
};

float THCudaTensor_varall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float mean = THCudaTensor_meanall(self);
  float result = thrust::transform_reduce(self_data, self_data+size, square_functor(mean), (float)0, thrust::plus<float>());

  result = result/(THCudaTensor_nElement(self)-1);

  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_stdall(THCudaTensor *self)
{
  return sqrt(THCudaTensor_varall(self));
}

// Given the sum of values and the sum of squares, compute the variance or standard deviation.
template<bool flag, bool apply_sqrt>
__forceinline__ __device__ float THCudaTensor_computeVar(float sum, float sum2, unsigned row_size) {
  if (flag) {
    sum /= row_size;
    sum2 /= row_size;
    sum2 -= sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  else {
    sum /= row_size;
    sum2 /= row_size - 1;
    sum2 -= ((float)row_size) / ((float)(row_size - 1)) * sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  if (apply_sqrt)
    return sqrt(sum2);
  else
    return sum2;
}

/* Compute the variance (or standard deviation) along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<bool flag, bool apply_sqrt>
__global__ void THCudaTensor_kernel_varOuterDim(float *tgt, float *src_, unsigned num_orows, unsigned num_irows, unsigned row_size)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      float sum = 0, sum2 = 0;

      for (unsigned col = 0; col < row_size; ++col) {
        float val = *src;
        sum += val;
        sum2 += val * val;

        src += num_irows;
      }

      tgt[orow * num_irows + irow] = THCudaTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
  }
}

template<bool apply_sqrt>
__host__ void THCudaTensor_varOuterDim(THCudaTensor *tgt, THCudaTensor *src, long dimension, int flag)
{
  unsigned ndim = THCudaTensor_nDimension(src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < dimension; dim++) {
    num_orows *= THCudaTensor_size(src, dim);
  }
  unsigned row_size = THCudaTensor_size(src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THCudaTensor_size(src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, DIVUP(num_irows, threads.x)));

  if (flag) {
    THCudaTensor_kernel_varOuterDim<true, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(tgt), THCudaTensor_data(src), num_orows, num_irows, row_size);
  } else {
    THCudaTensor_kernel_varOuterDim<false, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(tgt), THCudaTensor_data(src), num_orows, num_irows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}


/* Compute the variance (or standard deviation) of the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<bool flag, bool apply_sqrt>
__global__ void THCudaTensor_kernel_varInnermostDim(float *tgt, float *src_, unsigned num_rows, unsigned row_size)
{
  __shared__ float ssum[32][16];
  __shared__ float ssum2[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    float sum = 0, sum2 = 0;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        float val = src[col];
        sum += val;
        sum2 += val * val;
      }
    }
    ssum[threadIdx.y][threadIdx.x] = sum;
    ssum2[threadIdx.y][threadIdx.x] = sum2;
    __syncthreads();

    // Reduce intermediate values to single value.
    for (unsigned s = 8; s > 1; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        ssum[threadIdx.y][threadIdx.x] += ssum[threadIdx.y][threadIdx.x + s];
        ssum2[threadIdx.y][threadIdx.x] += ssum2[threadIdx.y][threadIdx.x + s];
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      sum = ssum[threadIdx.y][0] + ssum[threadIdx.y][1];
      sum2 = ssum2[threadIdx.y][0] + ssum2[threadIdx.y][1];
      tgt[row] = THCudaTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
    __syncthreads();
  }
}

template<bool apply_sqrt>
__host__ void THCudaTensor_varInnermostDim(THCudaTensor *tgt, THCudaTensor *src, int flag)
{
  unsigned ndim = THCudaTensor_nDimension(src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCudaTensor_size(src, dim);
  }
  unsigned row_size = THCudaTensor_size(src, ndim - 1);

  // From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
  dim3 threads(16, 32);
  dim3 grid(min(1024, DIVUP(num_rows, threads.y)));

  if (flag) {
    THCudaTensor_kernel_varInnermostDim<true, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(tgt), THCudaTensor_data(src), num_rows, row_size);
  } else {
    THCudaTensor_kernel_varInnermostDim<false, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(tgt), THCudaTensor_data(src), num_rows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

void THCudaTensor_var(THCudaTensor *self_, THCudaTensor *src, long dimension, int flag)
{
  THLongStorage *dim = THCudaTensor_newSizeOf(src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(self_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);

  if (dimension == THCudaTensor_nDimension(src) - 1) {
    THCudaTensor_varInnermostDim<false>(self, src, flag);
  } else {
    THCudaTensor_varOuterDim<false>(self, src, dimension, flag);
  }

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_std(THCudaTensor *self_, THCudaTensor *src, long dimension, int flag)
{
  THLongStorage *dim = THCudaTensor_newSizeOf(src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(self_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);

  if (dimension == THCudaTensor_nDimension(src) - 1) {
    THCudaTensor_varInnermostDim<true>(self, src, flag);
  } else {
    THCudaTensor_varOuterDim<true>(self, src, dimension, flag);
  }

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}


template<class Op>
void THCudaTensor_logicalValue(THCudaTensor *self_, THCudaTensor *src, Op op)
{
  THCudaTensor_resizeAs(self_, src);

  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  src = THCudaTensor_newContiguous(src);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, op);

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}


struct partial_less_functor
{
  const float rhs;
  partial_less_functor(float rhs) : rhs(rhs) {}
  __host__ __device__ bool operator()(const float &lhs) const {return lhs < rhs;}
};


void THCudaTensor_ltValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_less_functor(value));
}


struct partial_greater_functor
{
  const float rhs;
  partial_greater_functor(float rhs) : rhs(rhs) {}
  __host__ __device__ bool operator()(const float &lhs) const {return lhs > rhs;}
};


void THCudaTensor_gtValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_greater_functor(value));
}


struct partial_less_equal_functor
{
  const float rhs;
  partial_less_equal_functor(float rhs) : rhs(rhs) {}
  __host__ __device__ bool operator()(const float &lhs) const {return lhs <= rhs;}
};


void THCudaTensor_leValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_less_equal_functor(value));
}


struct partial_greater_equal_functor
{
  const float rhs;
  partial_greater_equal_functor(float rhs) : rhs(rhs) {}
  __host__ __device__ bool operator()(const float &lhs) const {return lhs >= rhs;}
};


void THCudaTensor_geValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_greater_equal_functor(value));
}


struct partial_equal_functor
{
  const float rhs;
  partial_equal_functor(float rhs) : rhs(rhs) {}
  __host__ __device__ bool operator()(const float &lhs) const {return lhs == rhs;}
};


void THCudaTensor_eqValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_equal_functor(value));
}


struct partial_not_equal_functor
{
  const float rhs;
  partial_not_equal_functor(float rhs) : rhs(rhs) {}
  __host__ __device__ bool operator()(const float &lhs) const {return lhs != rhs;}
};


void THCudaTensor_neValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_not_equal_functor(value));
}


template<class Op>
void THCudaTensor_logicalTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2, Op op)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");

  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  src1 = THCudaTensor_newContiguous(src1);
  src2 = THCudaTensor_newContiguous(src2);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src1_data(THCudaTensor_data(src1));
  thrust::device_ptr<float> src2_data(THCudaTensor_data(src2));

  thrust::transform(src1_data, src1_data+size, src2_data, self_data, op);

  THCudaTensor_free(src1);
  THCudaTensor_free(src2);
  THCudaTensor_freeCopyTo(self, self_);
}


void THCudaTensor_ltTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, thrust::less<float>());
}


void THCudaTensor_gtTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, thrust::greater<float>());
}


void THCudaTensor_leTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, thrust::less_equal<float>());
}


void THCudaTensor_geTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, thrust::greater_equal<float>());
}


void THCudaTensor_eqTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, thrust::equal_to<float>());
}


void THCudaTensor_neTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, thrust::not_equal_to<float>());
}


struct norm_functor
{
  const float exponent;

  norm_functor(float exponent_) : exponent(exponent_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return pow(fabs(x), exponent);
  }
};


float THCudaTensor_normall(THCudaTensor *self, float value)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result;
  if(value == 0.0f) {
    result = thrust::transform_reduce(self_data, self_data+size, partial_not_equal_functor(0.0f), (float)0, thrust::plus<float>());
  } else {
    result = thrust::transform_reduce(self_data, self_data+size, norm_functor(value), (float)0, thrust::plus<float>());
    result = pow(result, (float)1.0/value);
  }

  THCudaTensor_free(self);
  return result;
}

void THCudaTensor_norm(THCudaTensor* self, THCudaTensor* src, float value, long dimension)
{
  if (value == 0.0f) {
    THCudaTensor_transformReduceDim(self, src, dimension, partial_not_equal_functor(0.0f), (float)0, thrust::plus<float>());
  } else {
    THCudaTensor_transformReduceDim(self, src, dimension, norm_functor(value), (float)0, thrust::plus<float>());
    THCudaTensor_pow(self, self, 1/value);
  }
}

__global__ void THCudaTensor_kernel_renorm(float *data, const float value, const long size, const float maxnorm)
{
  __shared__ float buffer[32];
  long tx = threadIdx.x;
  long bx = blockIdx.x;
  long step = blockDim.x;
  float *row = data + size*bx;

  buffer[tx] = 0;

  // get norm of axis
  for (long i=tx; i<size; i+=step)
  {
    buffer[tx] += pow(fabs(row[i]), value);
  }
  // add (reduce)
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  // clip norms
  __syncthreads();
  float norm = pow(buffer[0], 1/value);
  if (norm > maxnorm)
  {
    norm = maxnorm / (norm + 1e-7);
    // renormalize
    for (long i=tx; i<size; i+=step)
    {
      row[i] *= norm;
    }
  }
}

void THCudaTensor_renorm(THCudaTensor* self, THCudaTensor* src, float value, long dimension, float maxnorm)
{
  THCudaTensor *self_;
  THCudaTensor *src_ = THCudaTensor_newTranspose(src, dimension, 0);
  THCudaTensor *data = THCudaTensor_newClone(src_);
  long size = THCudaTensor_nElement(data)/data->size[0];

  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(src), 3, "invalid dimension");
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THCudaTensor_nDimension(src) > 1, 1, "need at least 2 dimensions");

  dim3 grid(data->size[0]);
  dim3 threads(32);

  THCudaTensor_kernel_renorm<<<grid, threads>>>(THCudaTensor_data(data), value, size, maxnorm);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(src_);
  self_ = THCudaTensor_newTranspose(data, dimension, 0);
  THCudaTensor_resizeAs(self, self_);
  THCudaTensor_freeCopyTo(self_, self);
  THCudaTensor_free(data);
}

struct dist_functor
{
  const float exponent;

  dist_functor(float exponent_) : exponent(exponent_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return pow(fabs(x-y), exponent);
  }
};

float THCudaTensor_dist(THCudaTensor *self, THCudaTensor *src, float value)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  src = THCudaTensor_newContiguous(src);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  float result = thrust::inner_product(self_data, self_data+size, src_data, (float) 0,thrust::plus<float>(), dist_functor(value));

  THCudaTensor_free(src);
  THCudaTensor_free(self);

  return pow(result, (float)1.0/value);
}

void THCudaTensor_rand(THCudaRNGState* rng_state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_uniform(rng_state, r_, 0, 1);
}

void THCudaTensor_randn(THCudaRNGState* rng_state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_normal(rng_state, r_, 0, 1);
}

__global__ void THCudaTensor_kernel_indexFill(
   float *tensor, long* stride, float *index, long src_nDim,
   int dim, long idx_size, long tensor_size, long size_dim, float val
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = tensor_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          coeff = leftover / (stride[d] / size_dim);
          leftover -= coeff * (stride[d] / size_dim);
          srcIdx += coeff * stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / stride[d];
          leftover -= coeff * stride[d];
          srcIdx += coeff * stride[d];
        }
      }
        tensor[srcIdx + (int)((index[i])-1)*stride[dim]] = val;
    }
  }
}

__global__ void THCudaTensor_kernel_indexCopy(
   float *res, float *src, long* res_stride, float *index,
   long res_nDim, int dim, long idx_size, long src_size, long size_dim
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = src_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int resIdx = 0;
      for (int d=0; d<res_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = res_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          resIdx += coeff * res_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / res_stride[d];
          leftover -= coeff * res_stride[d];
          targetIdx += coeff * res_stride[d];
          resIdx += coeff * res_stride[d];
        }
      }
      res[resIdx + ((int)(index[i])-1)*res_stride[dim]] = src[targetIdx + i*res_stride[dim]];
    }
  }
}

void THCudaTensor_indexCopy(THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src)
{
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THCudaTensor_newContiguous(src);
  indices_ = THCudaTensor_newWithSize1d(nIndex);
  THCudaTensor_copyLong(indices_, indices);

  nRes = THCudaTensor_nElement(res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexCopy<<<nblocks, nthreads>>>(
    THCudaTensor_data(res_), THCudaTensor_data(src),
    stride_, THCudaTensor_data(indices_),
    res_->nDimension, dim, nIndex,
    THCudaTensor_nElement(src), res_->size[dim]
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(indices_);
  THCudaTensor_free(src);
}


void THCudaTensor_indexFill(THCudaTensor *res_, int dim, THLongTensor *indices, float val)
{
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < res_->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(res_->nDimension > 0, 2, "Source tensor is empty");

  indices_ = THCudaTensor_newWithSize1d(nIndex);
  THCudaTensor_copyLong(indices_, indices);

  nRes = THCudaTensor_nElement(res_) / res_->size[dim] * nIndex;


  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexFill<<<nblocks, nthreads>>>(
    THCudaTensor_data(res_), stride_, THCudaTensor_data(indices_),
    res_->nDimension, dim, nIndex, nRes, res_->size[dim], val
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(indices_);
}

__global__ void THCudaTensor_kernel_indexSelect(
   float *tensor, float *src, long* src_stride, float *index,
   long src_nDim, int dim, long idx_size, long tensor_size, long size_dim
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = tensor_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = src_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          srcIdx += coeff * src_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / src_stride[d];
          leftover -= coeff * src_stride[d];
          targetIdx += coeff * src_stride[d];
          srcIdx += coeff * src_stride[d];
        }
      }
      tensor[targetIdx + i*src_stride[dim]] = src[srcIdx + ((int)(index[i])-1)*src_stride[dim]];
    }
  }
}


void THCudaTensor_indexSelect(THCudaTensor *res_, THCudaTensor *src, int dim, THLongTensor *indices)
{
  THLongStorage *newSize;
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize, src->size);
  newSize->data[dim] = nIndex;
  THCudaTensor_resize(res_, newSize, NULL);
  THLongStorage_free(newSize);

  indices_ = THCudaTensor_newWithSize1d(nIndex);
  THCudaTensor_copyLong(indices_, indices);

  nRes = THCudaTensor_nElement(res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, src->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, src->stride, src->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexSelect<<<nblocks, nthreads>>>(
    THCudaTensor_data(res_), THCudaTensor_data(src),
    stride_, THCudaTensor_data(indices_),
    src->nDimension, dim, indices->size[0], nRes, src->size[dim]
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(indices_);
}
