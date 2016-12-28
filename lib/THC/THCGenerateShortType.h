#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateShortType.h"
#endif
#ifndef THC_GENERIC_NO_SHORT
#define real short
#define accreal long
#define Real Short
#define CReal CudaShort
#define THC_REAL_IS_SHORT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_SHORT
#endif
#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
