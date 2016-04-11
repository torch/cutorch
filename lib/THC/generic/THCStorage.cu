#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.cu"
#else

void THCStorage_(fill)(THCState *state, THCStorage *self, hostreal _value)
{
  thrust::device_ptr<real> self_data(self->data);
  real value = hostrealToReal(_value);
  thrust::fill(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+self->size, value);
}

void THCStorage_(resize)(THCState *state, THCStorage *self, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      if(state->showAllocations) printf("Free CUDA storage %li bytes\n",
        self->size * sizeof(real));
      THCudaCheck(THCudaFree(state, self->data));
      THCHeapUpdate(state, -self->size * sizeof(real));
    }
    self->data = NULL;
    self->size = 0;
  }
  else
  {
    real *data = NULL;
    // update heap *before* attempting malloc, to free space for the malloc
    THCHeapUpdate(state, size * sizeof(real));
    if(state->showAllocations) printf("Alloc CUDA storage %li bytes\n",
      size * sizeof(real));
    cudaError_t err = THCudaMalloc(state, (void**)(&data), size * sizeof(real));
    if(err != cudaSuccess) {
      THCHeapUpdate(state, -size * sizeof(real));
    }
    THCudaCheck(err);

    if (self->data) {
      THCudaCheck(cudaMemcpyAsync(data,
                                  self->data,
                                  THMin(self->size, size) * sizeof(real),
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      if(state->showAllocations) printf("Free CUDA storage %li bytes\n",
        self->size * sizeof(real));
      THCudaCheck(THCudaFree(state, self->data));
      THCHeapUpdate(state, -self->size * sizeof(real));
    }

    self->data = data;
    self->size = size;
  }
}
#endif
