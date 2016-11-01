#include "THCAllocator.h"

static void *THCudaHostAllocator_malloc(void* ctx, ptrdiff_t size) {
  void* ptr;

  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  THCudaCheck(cudaMallocHost(&ptr, size));

  return ptr;
}

static void THCudaHostAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;

  THCudaCheck(cudaFreeHost(ptr));
}

void THCAllocator_init(THCState *state) {
  state->cudaHostAllocator->malloc = &THCudaHostAllocator_malloc;
  state->cudaHostAllocator->realloc = NULL;
  state->cudaHostAllocator->free = &THCudaHostAllocator_free;
}

static void *THCUVAHostAllocator_alloc(void* ctx, ptrdiff_t size) {
  void* ptr;

  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  THCudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachHost));
  return ptr;
}

static void THCUVAHostAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;

  THCudaCheck(cudaFree(ptr));
}

static void *THCUVAHostAllocator_realloc(void* ctx, void* ptr, ptrdiff_t size) {
  if (size < 0) THError("Invalid memory size: %ld", size);

  THCUVAHostAllocator_free(ctx, ptr);

  if (size == 0) return NULL;

  THCudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachHost));

  return ptr;
}

void THCUVAHostAllocator_init(THAllocator *cudaUVAHostAllocator) {
  cudaUVAHostAllocator->malloc = &THCUVAHostAllocator_alloc;
  cudaUVAHostAllocator->realloc = &THCUVAHostAllocator_realloc;
  cudaUVAHostAllocator->free = &THCUVAHostAllocator_free;
}
