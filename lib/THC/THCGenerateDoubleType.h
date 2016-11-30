#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateDoubleType.h"
#endif
#ifndef THC_GENERIC_NO_DOUBLE
#define real double
#define accreal double
#define Real Double
#define CReal CudaDouble
#define THC_REAL_IS_DOUBLE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_DOUBLE
#endif

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
