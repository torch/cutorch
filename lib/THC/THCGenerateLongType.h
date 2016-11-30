#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateLongType.h"
#endif
#ifndef THC_GENERIC_NO_LONG
#define real long
#define accreal long
#define Real Long
#define CReal CudaLong
#define THC_REAL_IS_LONG
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_LONG
#endif
#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
