#include "THC.h"
#include "THCThrustAllocator.cuh"
#include "THCTensorTypeUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "THCTensorMode.cuh"

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
unsigned long nnextHighestPowerOf2(unsigned long n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;
  return n;
}

#include "generic/THCTensorMode.cu"
#include "THCGenerateAllTypes.h"
