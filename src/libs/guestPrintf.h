#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_GUEST_PRINTF_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_GUEST_PRINTF_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs {

struct VaContext;
struct VaList;

using guest_printf_std_func_t = KYTY_FORMAT_PRINTF(1, 2) KYTY_SYSV_ABI
    int (*)(const char* str, ...);
using guest_printf_ctx_func_t   = int (*)(VaContext* c);
using guest_snprintf_ctx_func_t = int (*)(VaContext* c);
using guest_vprintf_func_t      = int (*)(const char* str, VaList* c);

guest_printf_std_func_t   GetGuestPrintfStdFunc();
guest_printf_ctx_func_t   GetGuestPrintfCtxFunc();
guest_snprintf_ctx_func_t GetGuestSnprintfCtxFunc();
guest_vprintf_func_t      GetGuestVprintfFunc();

// Full formatter used by snprintf. Truncates to size-1 and returns the
// length that would have been written, excluding the terminating NUL.
int GuestVsnprintf(char* buffer, size_t size, const char* format, VaList* va_list);

} // namespace Libs

#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_GUEST_PRINTF_H_ */
