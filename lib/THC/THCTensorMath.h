#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"

struct THCudaBlasState;
struct THCudaRNGState;

THC_API void THCudaTensor_fill(THCudaTensor *self, float value);
THC_API void THCudaTensor_zero(THCudaTensor *self);

THC_API void THCudaTensor_zeros(THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_ones(THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_reshape(THCudaTensor *r_, THCudaTensor *t, THLongStorage *size);
THC_API long THCudaTensor_numel(THCudaTensor *t);

THC_API void THCudaTensor_add(THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_mul(THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_div(THCudaTensor *self, THCudaTensor *src, float value);


THC_API void THCudaTensor_cadd(struct THCudaBlasState* blas_state, THCudaTensor *self, THCudaTensor *src1, float value, THCudaTensor *src2);
THC_API void THCudaTensor_cmul(THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cdiv(THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);

THC_API void THCudaTensor_addcmul(THCudaTensor *self, THCudaTensor* t, float value, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_addcdiv(THCudaTensor *self, THCudaTensor* t, float value, THCudaTensor *src1, THCudaTensor *src2);

THC_API float THCudaTensor_dot(struct THCudaBlasState* blas_state, THCudaTensor *self, THCudaTensor *src);

THC_API float THCudaTensor_minall(THCudaTensor *self);
THC_API float THCudaTensor_maxall(THCudaTensor *self);
THC_API float THCudaTensor_sumall(THCudaTensor *self);
THC_API float THCudaTensor_prodall(THCudaTensor *self);
THC_API void THCudaTensor_min(THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dim);
THC_API void THCudaTensor_max(THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dim);
THC_API void THCudaTensor_sum(THCudaTensor *self, THCudaTensor *src, long dim);
THC_API void THCudaTensor_prod(THCudaTensor *self, THCudaTensor *src, long dim);

THC_API void THCudaTensor_addmv(struct THCudaBlasState* blas_state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec);
THC_API void THCudaTensor_addmm(struct THCudaBlasState* blas_state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat1, THCudaTensor *mat2);
THC_API void THCudaTensor_addr(struct THCudaBlasState* blas_state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2);

THC_API void THCudaTensor_log(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_log1p(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_exp(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_cos(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_acos(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_cosh(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sin(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_asin(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sinh(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_tan(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_atan(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_tanh(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_pow(THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_clamp(THCudaTensor *self, THCudaTensor *src, float min_value, float max_value);
THC_API void THCudaTensor_sqrt(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_ceil(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_floor(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_abs(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sign(THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_round(THCudaTensor *self, THCudaTensor *src);
TH_API void THCudaTensor_atan2(THCudaTensor *r_, THCudaTensor *tx, THCudaTensor *ty);

THC_API void THCudaTensor_ltValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_gtValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_leValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_geValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_eqValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_neValue(THCudaTensor *self_, THCudaTensor *src, float value);

THC_API void THCudaTensor_ltTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_gtTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_leTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_geTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_eqTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_neTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);

THC_API float THCudaTensor_meanall(THCudaTensor *self);
THC_API void  THCudaTensor_mean(THCudaTensor *self, THCudaTensor *src, long dim);
THC_API float THCudaTensor_varall(THCudaTensor *self);
THC_API void  THCudaTensor_var(THCudaTensor *self, THCudaTensor *src, long dim, int flag);
THC_API float THCudaTensor_stdall(THCudaTensor *self);
THC_API void  THCudaTensor_std(THCudaTensor *self, THCudaTensor *src, long dim, int flag);
THC_API float THCudaTensor_normall(THCudaTensor *self, float value);
THC_API void  THCudaTensor_norm(THCudaTensor* self, THCudaTensor* src, float value, long dimension);
THC_API void  THCudaTensor_renorm(THCudaTensor* self, THCudaTensor* src, float value, long dimension, float max_norm);
THC_API float THCudaTensor_dist(THCudaTensor *self, THCudaTensor *src, float value);

THC_API void THCudaTensor_rand(struct THCudaRNGState* rng_state, THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_randn(struct THCudaRNGState* rng_state, THCudaTensor *r_, THLongStorage *size);

THC_API void THCudaTensor_indexCopy(THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src);
THC_API void THCudaTensor_indexFill(THCudaTensor *tensor, int dim, THLongTensor *index, float val);
THC_API void THCudaTensor_indexSelect(THCudaTensor *tensor, THCudaTensor *src, int dim, THLongTensor *index);


#endif
