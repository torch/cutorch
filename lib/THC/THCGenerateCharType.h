#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateCharType.h"
#endif
#ifndef THC_GENERIC_NO_CHAR
#define real char
#define accreal long
#define Real Char
#define CReal CudaChar
#define THC_REAL_IS_CHAR
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_CHAR
#endif

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
