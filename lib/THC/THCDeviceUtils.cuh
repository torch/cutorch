#ifndef THC_DEVICE_UTILS_INC
#define THC_DEVICE_UTILS_INC

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

#endif // THC_DEVICE_UTILS_INC