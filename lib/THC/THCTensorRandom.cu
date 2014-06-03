#include "THCTensorRandom.h"
#include "THCGeneral.h"

#include <thrust/functional.h>
#include <curand.h>

/* Generator */
static curandGenerator_t gen;

/* Initial seed */
static int initf = 0;
static unsigned long initial_seed = 0;

/* Random seed (this must be called once) */
__host__ unsigned long THCRandom_seed()
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeed(s);
  return s;
}

/* Manually set the seed */
__host__ void THCRandom_manualSeed(unsigned long seed)
{
  initial_seed = seed;
  if (initf == 1) curandDestroyGenerator(gen);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(gen, initial_seed);
  initf = 1;
}

/* Get the initial seed */
__host__ unsigned long THCRandom_initialSeed()
{
  return initial_seed;
}

/* The following functors are use to modify uniform distributions  */
struct bernoulli_functor
{
  const double p;
  bernoulli_functor(double p_) : p(p_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)(x <= p);
  }
};

struct geometric_functor
{
  const double p;
  geometric_functor(double p_) : p(p_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)((log(1-x) / log(p)) + 1);
  }
};

struct exponential_functor
{
  const double lambda;
  exponential_functor(double lambda_) : lambda(lambda_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)(-1. / lambda * log(1-x));
  }
};

struct cauchy_functor
{
  const double median,sigma;
  cauchy_functor(double median_, double sigma_) : median(median_),sigma(sigma_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)(median + sigma * tan(M_PI*(x-0.5)));
  }
};

THC_API void THCudaTensor_uniform(THCudaTensor *self_, double a, double b) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);

  curandGenerateUniform(gen, data, size);

  if ((a != 0) || (b != 1)) {
      THCudaTensor_mul(self, b-a);
      THCudaTensor_add(self, a);
  }

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_bernoulli(THCudaTensor *self_, double p) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, bernoulli_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_normal(THCudaTensor *self_, double mean, double stdv) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);

  curandGenerateNormal(gen, data, size, mean, stdv);

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_logNormal(THCudaTensor *self_, double mean, double stdv) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  
  curandGenerateLogNormal(gen, data, size, mean, stdv);

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_geometric(THCudaTensor *self_, double p) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, geometric_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_exponential(THCudaTensor *self_, double lambda) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, exponential_functor(lambda));

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_cauchy(THCudaTensor *self_, double median, double sigma) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, cauchy_functor(median, sigma));

  THCudaTensor_freeCopyTo(self, self_);
};
<<<<<<< HEAD

__global__ void find_sampleidx(float* cum_dist, float *sample_idx, float uniform_sample, long n_categories)
{
  int k = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  if (k < n_categories-1)
  {
    if (uniform_sample <= cum_dist[0])
    {
      *sample_idx = 1;
    }
  
    else if (cum_dist[k] < uniform_sample && uniform_sample <= cum_dist[k+1])
    {
      *sample_idx = k + 2;
    }   
  }
}

THC_API void THCudaTensor_multinomial(THCudaTensor *self, THCudaTensor *prob_dist, int n_sample, int with_replacement)
{

  int start_dim = THCudaTensor_nDimension(prob_dist);
  long n_dist;
  long n_categories;
  THCudaTensor* cum_dist;
  int i,j,k;
  
  if (start_dim == 1)
  {
    THCudaTensor_resize2d(prob_dist, 1, THCudaTensor_size(prob_dist, 0));
  }
  
  n_dist = THCudaTensor_size(prob_dist, 0);
  n_categories = THCudaTensor_size(prob_dist, 1);
  
  THArgCheck(n_sample > 0, 2, "cannot sample n_sample < 0 samples");
  
  if (!with_replacement)
  {
    THArgCheck((!with_replacement) && (n_sample <= n_categories), 2, \
    "cannot sample n_sample > prob_dist:size(1) samples without replacement");
  }
  
  // cumulative probability distribution vector
  cum_dist = THCudaTensor_newWithSize1d(n_categories);
    
  /* will contain multinomial samples (category indices to be returned) */
  THCudaTensor_resize2d(self, n_dist , n_sample);
  
  for (i=0; i<n_dist; i++)
  {
    /* Get normalized cumulative distribution from prob distribution */
    float sum = 0;
    for (j=0; j<n_categories; j++)
    {
      sum += THCudaStorage_get(prob_dist->storage, 
              prob_dist->storageOffset + i * prob_dist->stride[0] + j * prob_dist->stride[1]);
        
      THCudaStorage_set(cum_dist->storage, 
                        cum_dist->storageOffset + j * cum_dist->stride[0], 
                        sum);
    }
    
    THArgCheck((sum > 0), 2, "invalid multinomial distribution (sum of probabilities <= 0)");
    
    /* normalize cumulative probability distribution so that last val is 1 
    i.e. dosen't assume original prob_dist row sums to one */
    if ( (sum > 0) || ( ( sum < 1.00001) && (sum > 0.99999) ) )  
    {
      for (j=0; j<n_categories; j++)
      {     
        long idx = cum_dist->storageOffset + j * cum_dist->stride[0];
        float val = THCudaStorage_get(cum_dist->storage, idx);
        THCudaStorage_set(cum_dist->storage, idx, val/sum);
      }
    }
    
    for (j=0; j<n_sample; j++)
    {
      /* sample a probability mass from a uniform distribution */
      THCudaTensor *uniform_tensor = THCudaTensor_newWithSize1d(1);
      THCudaTensor_uniform(uniform_tensor, 0, 1);      
      float uniform_sample = THCudaStorage_get(uniform_tensor->storage,
                                               uniform_tensor->storageOffset);

      float *sample_idx;
      THCudaCheck(cudaMalloc((void**)&sample_idx, sizeof(float)));
      int nblocks = ceil((float)n_categories / (16 * 16));
        
      dim3 threads(16,16);
      dim3 grid(nblocks);
      find_sampleidx<<<grid, threads>>>(THCudaTensor_data(cum_dist), sample_idx, 
                                        uniform_sample, n_categories);

      long index = self->storageOffset + i * self->stride[0] + j * self->stride[1];
        
      THCudaCheck(cudaMemcpy(self->storage->data + index, sample_idx, sizeof(float), cudaMemcpyDeviceToDevice));
      THCudaCheck(cudaMemcpy(sample_idx, sample_idx, sizeof(float), cudaMemcpyDeviceToHost));

      float *h_sample_idx; 
      h_sample_idx = (float*)THAlloc(sizeof(float));
      THCudaCheck(cudaMemcpy(h_sample_idx, sample_idx, sizeof(float), cudaMemcpyDeviceToHost));
      
      // down to array indexing
      *h_sample_idx -= 1;
      
      // Once a sample is drawn, it cannot be drawn again. ie sample without replacement
      if (!with_replacement)
      {
        /* update cumulative distribution so that sample cannot be drawn again */
        float diff;
        float new_val = 0;
        float sum;
        
        if (*h_sample_idx != 0)
        {
          new_val = THCudaStorage_get(cum_dist->storage,
                                      cum_dist->storageOffset+(*h_sample_idx-1)*cum_dist->stride[0]);
        }
        /* marginal cumulative mass (i.e. original probability) of sample */
        diff = THCudaStorage_get( \
          cum_dist->storage, \
          cum_dist->storageOffset+*h_sample_idx*cum_dist->stride[0] \
        ) - new_val;
        /* new sum of marginals is not one anymore... */
        sum = 1.0 - diff;
        for (k=0; k<n_categories; k++)
        {
          new_val = THCudaStorage_get( \
            cum_dist->storage, \
            cum_dist->storageOffset+k*cum_dist->stride[0] \
          );
          if (k >= *h_sample_idx) 
          {
            /* remove sampled probability mass from later cumulative probabilities */
            new_val -= diff;
          }
          /* make total marginals sum to one */
          new_val /= sum;
          THCudaStorage_set( \
            cum_dist->storage, \
            cum_dist->storageOffset+k*cum_dist->stride[0], \
            new_val \
          );
        }
      }                                
    }
  }
  
  THCudaTensor_free(cum_dist);
  
  if (start_dim == 1)
  {
    THCudaTensor_resize1d(self, n_sample);
    THCudaTensor_resize1d(prob_dist, n_categories);
  }
}
=======
>>>>>>> parent of ec0f992... added multinomial cuda
