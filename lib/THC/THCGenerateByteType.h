#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define real unsigned char
#define accreal long
#define Real Byte
#define CReal CudaByte
#define THC_REAL_IS_BYTE
# ifdef THC_MIN_MATH
#  define THC_GENERIC_NO_MATH 1
# endif
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_BYTE
#undef THC_GENERIC_NO_MATH 

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
