#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorRandom.cu"
#else

#define NUM_BLOCKS min((int)THCCeilDiv(size, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(uniform)(THCState* state, THCTensor *self_, double a, double b)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_uniform<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, a, b);

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(normal)(THCState* state, THCTensor *self_, double mean, double stdv)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_normal<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, mean, stdv);

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(normal_means)(THCState *state, THCTensor *self, THCTensor *means, double stddev) {
  THCTensor_(resizeAs)(state, self, means);
  THCTensor_(normal)(state, self, 0, stddev);
  THCTensor_(cadd)(state, self, self, ScalarConvert<int, real>::to(1), means);
}

THC_API void THCTensor_(normal_stddevs)(THCState *state, THCTensor *self, double mean, THCTensor *stddevs)
{
  THCTensor_(resizeAs)(state, self, stddevs);
  THCTensor_(normal)(state, self, 0, 1);
  THCTensor_(cmul)(state, self, self, stddevs);
  THCTensor_(add)(state, self, self, ScalarConvert<double, real>::to(mean));
}

THC_API void THCTensor_(normal_means_stddevs)(THCState *state, THCTensor *self, THCTensor *means, THCTensor *stddevs)
{
  THCTensor_(resizeAs)(state, self, means);
  THCTensor_(normal)(state, self, 0, 1);
  THCTensor_(cmul)(state, self, self, stddevs);
  THCTensor_(cadd)(state, self, self, ScalarConvert<int, real>::to(1), means);
}

THC_API void THCTensor_(logNormal)(THCState* state, THCTensor *self_, double mean, double stdv)
{

  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generateLogNormal<real><<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, mean, stdv);

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(exponential)(THCState* state, THCTensor *self_, double lambda)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_exponential<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, lambda);

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(cauchy)(THCState* state, THCTensor *self_, double median, double sigma)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_cauchy<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, median, sigma);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(renormRows)(struct THCState* state,
                             THCTensor* t) {
  THAssert(THCTensor_(nDimension)(state, t) == 2);
  long rows = THCTensor_(size)(state, t, 0);
  long cols = THCTensor_(size)(state, t, 1);

  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
  THAssert(props != NULL);

  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;

  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(cols < maxThreads ? cols : maxThreads);

  renormRowsL1<real>
    <<<grid, block, block.x * sizeof(real),
    THCState_getCurrentStream(state)>>>(THCTensor_(data)(state, t),
                                        rows, cols);
}

THC_API void THCTensor_(multinomial)(struct THCState *state,
                                      THCudaLongTensor *self,
                                      THCTensor *prob_dist,
                                      int n_sample,
                                      int with_replacement)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, prob_dist));
  Generator* gen = THCRandom_getGenerator(state);

  int inputSize = THCTensor_(nDimension)(state, prob_dist);
  THArgCheck(inputSize > 0 && inputSize <= 2, 2,
             "prob_dist must be 1 or 2 dim");

  // Categories are in the innermost dimension
  long numDist =
    inputSize == 1 ? 1 : THCTensor_(size)(state, prob_dist, 0);
  long numCategoriesLong =
    inputSize == 1 ? THCTensor_(size)(state, prob_dist, 0) :
    THCTensor_(size)(state, prob_dist, 1);

  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  THArgCheck(numCategoriesLong <= FLOAT32_MAX_CONSECUTIVE_INT, 2,
             "number of categories cannot exceed 2^24");
  int numCategories = (int) numCategoriesLong;

  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");

  if (!with_replacement) {
    THArgCheck(n_sample <= numCategories, 2,
               "cannot sample n_sample > prob_dist:size(1) samples without "
               "replacement");
  }

  // It is possible that prob_dist is non-contiguous
  THCTensor* probDistContig =
    THCTensor_(newContiguous)(state, prob_dist);

  // Restructure data for 2d
  if (inputSize == 1) {
    THCTensor_(resize2d)(state, probDistContig, 1, numCategories);
  }

  THCudaLongTensor_resize2d(state, self, numDist, n_sample);

  // get current device properties
  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
  THAssert(props != NULL);
  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;
  int maxShared = props->sharedMemPerBlock;
  int requiredShared = (numCategories < maxThreads ? numCategories : maxThreads)
                                * (sizeof(real) * sizeof(accreal));
                                
  if (n_sample == 1 && maxShared >= requiredShared) {
    // Optimized allocation-free implementation
    // To exploit greater parallelism for the sampling, generate the
    // Uniform random samples in a separate kernel launch, into
    // temporarily allocated memory. The device RNG is thread-limited
    THCTensor *sampled = THCTensor_(newWithSize2d)(state, numDist, n_sample);
    THCTensor_(uniform)(state, sampled, 0.0, 1.0);

    dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
    dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);

    sampleMultinomialOnce<real, accreal>
      <<<grid, block,
         requiredShared,
         THCState_getCurrentStream(state)>>>(
      THCudaLongTensor_data(state, self),
      numDist,
      numCategories,
      THCTensor_(data)(state, sampled),
      THCTensor_(data)(state, probDistContig));
    THCTensor_(free)(state, sampled);
  } else {
    // Generic, slow implementation with memory allocations

    // For sampling without replacement, we modify the distribution
    // for subsequent samples in this space
    THCTensor* origDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, origDist, probDistContig);
    THCTensor_(copy)(state, origDist, probDistContig);

    THCTensor* normDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, normDist, probDistContig);

    THCTensor* prefixSum = THCTensor_(new)(state);

    // Renorm along rows
    THCTensor_(copy)(state, normDist, origDist);
    THCTensor_(renormRows)(state, normDist);

    // Prefix sum along rows
    THCTensor_(cumsum)(state, prefixSum, normDist, 1);

    if (with_replacement) {
      // Sample with replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from one
      // distribution concurrently.
      dim3 grid(numDist < MAX_NUM_BLOCKS ? numDist : MAX_NUM_BLOCKS);

      
      //Create the matrix of uniformly sampled numbers
      THCTensor *uniform_idx = THCTensor_(newWithSize1d)(state, n_sample);
      THCTensor_(uniform)(state, uniform_idx, 0, 1);
      
      
      sampleMultinomialWithReplacement
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
	   THCTensor_(data)(state, uniform_idx),
	   n_sample,
	   THCudaLongTensor_data(state, self),
	   numDist, numCategories,
	   THCTensor_(data)(state, prefixSum));
      
      THCTensor_(free)(state, uniform_idx);
    } else {
      // Sample without replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from a different
      // distribution concurrently.
      ptrdiff_t numBlocks = THCCeilDiv(numDist, 4L);
      dim3 grid(numBlocks < MAX_NUM_BLOCKS ? numBlocks : MAX_NUM_BLOCKS);

      for (int sample = 0; sample < n_sample; ++sample) {
        if (sample > 0) {
          // Update probabilities
          // Renorm along rows
          THCTensor_(copy)(state, normDist, origDist);
          THCTensor_(renormRows)(state, normDist);

          // Prefix sum along rows
          THCTensor_(cumsum)(state, prefixSum, normDist, 1);
        }

        // The kernel can only draw one sample before we have to
        // recalculate our distribution
        sampleMultinomialWithoutReplacement
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
	     gen->gen_states,
	     n_sample,	     
	     sample,	     
	     THCudaLongTensor_data(state, self),
	     numDist, numCategories,
	     THCTensor_(data)(state, origDist),
	     THCTensor_(data)(state, prefixSum));
      }
    }

    THCTensor_(free)(state, prefixSum);
    THCTensor_(free)(state, normDist);
    THCTensor_(free)(state, origDist);
  }

  // Revert data restructuring based on input sizes
  if (inputSize == 1) {
    THCudaLongTensor_resize1d(state, self, n_sample);

    // Unfortunately, if prob_dist is contiguous already,
    // newContiguous is not a private copy, so we have to restructure
    // this too, so as to not affect prob_dist
    THCTensor_(resize1d)(state, probDistContig, numCategories);
  }

  THCTensor_(free)(state, probDistContig);
}

THC_API void THCTensor_(multinomialAliasSetup)(THCState *state, THCTensor *_probs, THCudaLongTensor *_J, THCTensor *_q){
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  THAssert(THCTensor_(isContiguous)(state, _probs));
  long inputsize = THCTensor_(nElement)(state, _probs);
  THCudaLongTensor *smaller = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *smaller_short = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger_short = THCudaLongTensor_newWithSize1d(state, inputsize);
  
  THCudaLongTensor_resize1d(state, _J, inputsize);
  THCTensor_(resize1d)(state, _q, inputsize);
  
  real one = ScalarConvert<long, real>::to(1);
  int inputBlockDim = THCCeilDiv((int)inputsize + BLOCK_SIZE - 1, BLOCK_SIZE);
  aliasMultinomialFilter
    <<<inputBlockDim, BLOCK_SIZE, 0, THCState_getCurrentStream(state) >>>(
								     THCTensor_(data)(state, _q),
								     THCTensor_(data)(state, _probs),
								     THCudaLongTensor_data(state, smaller),
								     THCudaLongTensor_data(state, larger),
								     THCudaLongTensor_data(state, _J),
								     THCudaLongTensor_data(state, smaller_short),
								     THCudaLongTensor_data(state, larger_short),
								     one, inputsize
								     );
  
  THCudaLongTensor_nonzero(state, smaller_short, smaller);
  THCudaLongTensor_nonzero(state, larger_short, larger);
  int h_large_c = THCudaLongTensor_nElement(state, larger_short);
  THCudaLongTensor_resize1d(state, smaller_short, inputsize);
  THCudaLongTensor_resize1d(state, larger_short, inputsize);
  aliasMultinomialSetup
    <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
						    THCudaLongTensor_data(state, _J),
						    THCTensor_(data)(state, _q),
						    inputsize,
						    THCudaLongTensor_data(state, smaller_short),
						    THCudaLongTensor_data(state, larger_short),
						    inputsize - h_large_c, h_large_c
						    );
  real q_max = THCTensor_(maxall)(state, _q);
  condDiv<<<
    inputBlockDim, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
								      THCTensor_(data)(state, _q),
								      THCudaLongTensor_data(state, _J),
								      inputsize, q_max
								      );
  
  THCudaLongTensor_free(state, smaller);
  THCudaLongTensor_free(state, larger);
  THCudaLongTensor_free(state, smaller_short);
  THCudaLongTensor_free(state, larger_short);
}

THC_API void THCTensor_(multinomialAliasDraw)(THCState *state, THCudaLongTensor *self, THCudaLongTensor *_J, THCTensor *_q){
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  Generator* gen = THCRandom_getGenerator(state);
  long K = THCudaLongTensor_nElement(state, _J);
  long output_nelem = THCudaLongTensor_nElement(state, self);
  ptrdiff_t size = THCudaLongTensor_nElement(state, self);

  THCTensor *uniform = THCTensor_(newWithSize1d)(state, output_nelem);
  THCTensor *bernoulli = THCTensor_(newWithSize1d)(state, output_nelem);

  THCTensor_(uniform)(state, uniform, 0, K);
  THCTensor_(uniform)(state, bernoulli, 0, 1);

  multinomialAliasDrawKernel
    <<<THCCeilDiv((int)output_nelem+BLOCK_SIZE-1, BLOCK_SIZE), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
				  size,
				  THCudaLongTensor_data(state, self),
				  THCudaLongTensor_data(state, _J),
				  THCTensor_(data)(state, _q),
				  K,
				  THCTensor_(data)(state, uniform),
				  THCTensor_(data)(state, bernoulli)
				  );
}

THC_API void THCTensor_(rand)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(uniform)(state, r_, 0, 1);
}

void THCTensor_(randn)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(normal)(state, r_, 0, 1);
}

#endif

#if defined(THC_REAL_IS_DOUBLE)
GENERATE_KERNEL1(generate_bernoulli, double, double p, double, curand_uniform_double, x <= p)
#else
GENERATE_KERNEL1(generate_bernoulli, real, double p, float, curand_uniform, (ScalarConvert<bool, real>::to(x <= p)))
#endif

THC_API void THCTensor_(bernoulli)(THCState* state, THCTensor *self_, double p)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_bernoulli<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, p);

  THCTensor_(freeCopyTo)(state, self, self_);
};

#define DEFINE_BERNOULLI_TENSOR(NAME, PROB_TYPE, PROB_DATA_TYPE)               \
THC_API void THCTensor_(NAME)(THCState* state,                                 \
        THCTensor *self_, PROB_TYPE *probs_)                                   \
{                                                                              \
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, probs_));                     \
  Generator* gen = THCRandom_getGenerator(state);                              \
  THCTensor *self = THCTensor_(newContiguous)(state, self_);                   \
  PROB_TYPE *probs = PROB_TYPE##_newContiguous(state, probs_);                 \
  ptrdiff_t size = THCTensor_(nElement)(state, self);                          \
  ptrdiff_t prob_size = PROB_TYPE##_nElement(state, probs);                    \
  real *result_data = THCTensor_(data)(state, self);                           \
  PROB_DATA_TYPE *probs_data = PROB_TYPE##_data(state, probs);                 \
                                                                               \
  THArgCheck(size == prob_size, 3, "inconsistent tensor size");                \
                                                                               \
  generate_bernoulli_tensor<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>( \
      gen->gen_states, size, result_data, probs_data);                         \
                                                                               \
  PROB_TYPE##_free(state, probs);                                              \
  THCTensor_(freeCopyTo)(state, self, self_);                                  \
}

DEFINE_BERNOULLI_TENSOR(bernoulli_FloatTensor, THCudaTensor, float)
DEFINE_BERNOULLI_TENSOR(bernoulli_DoubleTensor, THCudaDoubleTensor, double)

#if defined(THC_REAL_IS_DOUBLE)

GENERATE_KERNEL1(generate_geometric, double, double p, double, curand_uniform_double, ceil(log(x) / log(1-p)))
#else
GENERATE_KERNEL1(generate_geometric, real, double p, float, curand_uniform, (ScalarConvert<float, real>::to(ceilf(logf(x) / log(1-p)))))
#endif

THC_API void THCTensor_(geometric)(THCState* state, THCTensor *self_, double p)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_geometric<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, p);

  THCTensor_(freeCopyTo)(state, self, self_);
};
#undef NUM_BLOCKS

#endif
