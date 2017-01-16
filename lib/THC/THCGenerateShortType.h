#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateShortType.h"
#endif

#define real short
#define accreal long
#define Real Short
#define CReal CudaShort
#define THC_REAL_IS_SHORT
# ifdef THC_MIN_MATH
#  define THC_GENERIC_NO_MATH 1
# endif
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_SHORT
#undef THC_GENERIC_NO_MATH
#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
