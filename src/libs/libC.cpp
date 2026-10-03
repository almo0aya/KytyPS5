#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/hostMemory.h"
#include "kernel/fileSystem.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "libs/guestPrintf.h"
#include "libs/libs.h"
#include "libs/vaContext.h"
#include "loader/runtimeLinker.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <getopt.h>
#include <libgen.h>
#endif
#include <cstring>
#include <ctime>
#include <fmt/format.h>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>


#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The RetroArch HLE in this file calls POSIX libc that MSVC/clang-cl does not ship.
// Linux and macOS keep the host implementations. These shims exist so the same
// calls compile on Windows; getopt_long follows POSIX (it does not permute argv).

static char* basename(char* path) {
	if (path == nullptr || path[0] == '\0') {
		return const_cast<char*>(".");
	}
	size_t len = std::strlen(path);
	while (len > 1 && path[len - 1] == '/') {
		path[--len] = '\0';
	}
	char* slash = std::strrchr(path, '/');
	if (slash == nullptr) {
		return path;
	}
	return slash[1] != '\0' ? slash + 1 : const_cast<char*>(".");
}

static char* dirname(char* path) {
	if (path == nullptr || path[0] == '\0') {
		return const_cast<char*>(".");
	}
	size_t len = std::strlen(path);
	while (len > 1 && path[len - 1] == '/') {
		path[--len] = '\0';
	}
	char* slash = std::strrchr(path, '/');
	if (slash == nullptr) {
		return const_cast<char*>(".");
	}
	if (slash == path) {
		slash[1] = '\0';
		return path;
	}
	*slash = '\0';
	return path;
}

static int strcasecmp(const char* a, const char* b) {
	return _stricmp(a, b);
}

static int strncasecmp(const char* a, const char* b, size_t n) {
	return _strnicmp(a, b, n);
}

static char* strtok_r(char* str, const char* delim, char** saveptr) {
	return strtok_s(str, delim, saveptr);
}

static char* strcasestr(const char* haystack, const char* needle) {
	if (haystack == nullptr || needle == nullptr) {
		return nullptr;
	}
	if (needle[0] == '\0') {
		return const_cast<char*>(haystack);
	}
	for (const char* h = haystack; *h != '\0'; ++h) {
		const char* a = h;
		const char* b = needle;
		while (*a != '\0' && *b != '\0' &&
		       std::tolower(static_cast<unsigned char>(*a)) ==
		           std::tolower(static_cast<unsigned char>(*b))) {
			++a;
			++b;
		}
		if (*b == '\0') {
			return const_cast<char*>(h);
		}
	}
	return nullptr;
}

static void srandom(unsigned seed) {
	std::srand(seed);
}

static long random() {
	return std::rand();
}

// UCRT already typedefs off_t as 32-bit long. These shims stay 64-bit so the
// HLE can pass guest off_t (int64) through without truncating.
static int fseeko(FILE* stream, long long offset, int whence) {
	return _fseeki64(stream, offset, whence);
}

static long long ftello(FILE* stream) {
	return _ftelli64(stream);
}

struct option {
	const char* name;
	int         has_arg;
	int*        flag;
	int         val;
};

static char* optarg = nullptr;
static int   optind = 1;
static int   opterr = 1;
static int   optopt = 0;

static int getopt_long(int argc, char* const argv[], const char* optstring,
                       const struct option* longopts, int* longindex) {
	static const char* cursor       = nullptr;
	static int         cursor_index = -1;

	optarg = nullptr;
	if (optind <= 0) {
		optind       = 1;
		cursor       = nullptr;
		cursor_index = -1;
	}
	if (optstring == nullptr) {
		optstring = "";
	}
	const bool  missing_colon = optstring[0] == ':';
	const char* shorts        = optstring;
	if (shorts[0] == '+' || shorts[0] == '-') {
		++shorts;
	}
	if (shorts[0] == ':') {
		++shorts;
	}

	if (cursor_index != optind) {
		cursor       = nullptr;
		cursor_index = -1;
	}

	if (cursor == nullptr) {
		if (optind >= argc || argv[optind] == nullptr) {
			return -1;
		}
		const char* arg = argv[optind];
		if (arg[0] != '-' || arg[1] == '\0') {
			return -1;
		}
		if (arg[1] == '-' && arg[2] == '\0') {
			++optind;
			return -1;
		}
		if (arg[1] == '-') {
			const char* name    = arg + 2;
			const char* eq      = std::strchr(name, '=');
			const size_t namelen = eq != nullptr ? static_cast<size_t>(eq - name) : std::strlen(name);
			const struct option* match       = nullptr;
			int                  match_index = -1;
			if (longopts != nullptr) {
				for (int i = 0; longopts[i].name != nullptr; ++i) {
					if (std::strncmp(longopts[i].name, name, namelen) == 0 &&
					    longopts[i].name[namelen] == '\0') {
						match       = &longopts[i];
						match_index = i;
						break;
					}
				}
				if (match == nullptr && namelen > 0) {
					int matches = 0;
					for (int i = 0; longopts[i].name != nullptr; ++i) {
						if (std::strncmp(longopts[i].name, name, namelen) == 0) {
							match       = &longopts[i];
							match_index = i;
							++matches;
						}
					}
					if (matches != 1) {
						match = nullptr;
					}
				}
			}
			++optind;
			if (match == nullptr) {
				optopt = 0;
				if (opterr != 0 && !missing_colon) {
					std::fprintf(stderr, "%s: unrecognized option '--%.*s'\n",
					             argv[0] != nullptr ? argv[0] : "", static_cast<int>(namelen), name);
				}
				return '?';
			}
			if (longindex != nullptr) {
				*longindex = match_index;
			}
			if (match->has_arg == 1) {
				if (eq != nullptr) {
					optarg = const_cast<char*>(eq + 1);
				} else if (optind < argc) {
					optarg = argv[optind++];
				} else {
					optopt = match->val;
					if (missing_colon) {
						return ':';
					}
					if (opterr != 0) {
						std::fprintf(stderr, "%s: option '--%s' requires an argument\n",
						             argv[0] != nullptr ? argv[0] : "", match->name);
					}
					return '?';
				}
			} else if (match->has_arg == 2) {
				if (eq != nullptr) {
					optarg = const_cast<char*>(eq + 1);
				}
			} else if (eq != nullptr) {
				if (opterr != 0 && !missing_colon) {
					std::fprintf(stderr, "%s: option '--%s' doesn't allow an argument\n",
					             argv[0] != nullptr ? argv[0] : "", match->name);
				}
				return '?';
			}
			if (match->flag != nullptr) {
				*match->flag = match->val;
				return 0;
			}
			return match->val;
		}
		cursor       = arg + 1;
		cursor_index = optind;
	}

	const char c = *cursor++;
	if (*cursor == '\0') {
		cursor       = nullptr;
		cursor_index = -1;
		++optind;
	}
	const char* spec = std::strchr(shorts, c);
	if (spec == nullptr || c == ':') {
		optopt = static_cast<unsigned char>(c);
		if (opterr != 0 && !missing_colon) {
			std::fprintf(stderr, "%s: invalid option -- '%c'\n", argv[0] != nullptr ? argv[0] : "", c);
		}
		return '?';
	}
	if (spec[1] == ':') {
		const bool optional = spec[2] == ':';
		if (cursor != nullptr) {
			optarg       = const_cast<char*>(cursor);
			cursor       = nullptr;
			cursor_index = -1;
			++optind;
		} else if (!optional && optind < argc) {
			optarg = argv[optind++];
		} else if (!optional) {
			optopt = static_cast<unsigned char>(c);
			if (missing_colon) {
				return ':';
			}
			if (opterr != 0) {
				std::fprintf(stderr, "%s: option requires an argument -- '%c'\n",
				             argv[0] != nullptr ? argv[0] : "", c);
			}
			return '?';
		}
	}
	return static_cast<unsigned char>(c);
}
#endif

namespace Libs {

namespace LibKernel {
void KernelDispatchPendingSignalForCurrentThread();
} // namespace LibKernel

namespace LibC {

LIB_VERSION("libc", 1, "libc", 1, 1);

static uint32_t g_need_flag = 1;

using cxa_destructor_func_t = KYTY_SYSV_ABI void (*)(void*);
using atexit_func_t         = KYTY_SYSV_ABI void (*)();

struct CxaDestructor {
	cxa_destructor_func_t destructor_func;
	void*                 destructor_object;
	void*                 module_id;
};

struct CContext {
	std::list<CxaDestructor> cxa;
	std::list<atexit_func_t> atexit;
};

struct InitEnvParams {
	int         argc;
	uint32_t    pad;
	const char* argv[3];
};

static int                g_argc = 0;
static const char* const* g_argv = nullptr;
static const char* const* g_envp = nullptr;

int GetArgc() {
	return g_argc;
}

const char** GetArgv() {
	return const_cast<const char**>(g_argv);
}

static KYTY_SYSV_ABI void exit(int code) {
	PRINT_NAME();

	::exit(code);
}

static void PrintAbortStringCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	const auto* ptr = reinterpret_cast<const char*>(static_cast<uintptr_t>(addr));
	char        buf[257] {};
	size_t      len = 0;

	for (; len < sizeof(buf) - 1 && Graphics::HostMemoryIsReadable(addr + len); len++) {
		const auto c = static_cast<unsigned char>(ptr[len]);
		if (c == 0) {
			break;
		}
		if (!std::isprint(c) && c != '\n' && c != '\r' && c != '\t') {
			return;
		}
		buf[len] = static_cast<char>(c);
	}

	if (len >= 4) {
		LOGF("\t %s string = \"%s\"\n", name, buf);
	}
}

static void PrintAbortWideStringCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	const auto* ptr = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(addr));
	char        buf[257] {};
	size_t      len = 0;

	for (;
	     len < sizeof(buf) - 1 && Graphics::HostMemoryIsReadable(addr + len * sizeof(uint16_t) + 1);
	     len++) {
		const auto c = ptr[len];
		if (c == 0) {
			break;
		}
		if (c > 0x7f ||
		    (!std::isprint(static_cast<unsigned char>(c)) && c != '\n' && c != '\r' && c != '\t')) {
			return;
		}
		buf[len] = static_cast<char>(c);
	}

	if (len >= 4) {
		LOGF("\t %s u16 = \"%s\"\n", name, buf);
	}
}

static void PrintAbortBytesCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	uint8_t bytes[64] {};
	size_t  len = 0;

	for (; len < sizeof(bytes) && Graphics::HostMemoryIsReadable(addr + len); len++) {
		bytes[len] = *reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr + len));
	}

	if (len == 0) {
		return;
	}

	LOGF("\t %s bytes =", name);
	for (size_t i = 0; i < len; i++) {
		LOGF(" %02" PRIx8, bytes[i]);
	}
	LOGF("\n");
}

static void PrintAbortPointerCandidate(const char* name, uint64_t addr) {
	PrintAbortStringCandidate(name, addr);
	PrintAbortWideStringCandidate(name, addr);
	PrintAbortBytesCandidate(name, addr);
}

static bool LooksLikeAbortPointer(uint64_t addr) {
	return Graphics::HostMemoryIsReadable(addr) && addr >= 0x10000;
}

static void PrintAbortPointerArrayCandidate(const char* name, uint64_t addr) {
	if (!LooksLikeAbortPointer(addr)) {
		return;
	}

	for (int i = 0; i < 8; i++) {
		const auto slot_addr = addr + static_cast<uint64_t>(i) * sizeof(uint64_t);
		if (!Graphics::HostMemoryIsReadable(slot_addr)) {
			break;
		}

		const auto value = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(slot_addr));
		if (!LooksLikeAbortPointer(value)) {
			continue;
		}

		LOGF("\t %s[%d] = 0x%016" PRIx64 "\n", name, i, value);
		const auto child_name = fmt::format("{}[{}]", name, i);
		PrintAbortPointerCandidate(child_name.c_str(), value);
	}
}

[[noreturn]] static KYTY_SYSV_ABI void abort(uint64_t arg0, uint64_t arg1, uint64_t arg2,
                                             uint64_t arg3, uint64_t arg4, uint64_t arg5) {
	PRINT_NAME();

	uint64_t rbp = 0;
	uint64_t rsp = 0;
	// This assembly is x86-64-specific, not Windows-specific.
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("movq %%rbp, %0" : "=r"(rbp));
	asm volatile("movq %%rsp, %0" : "=r"(rsp));
#endif

	const auto ret = reinterpret_cast<uint64_t>(__builtin_return_address(0));

	LOGF("Guest abort diagnostics:\n"
	     "\t return = 0x%016" PRIx64 "\n"
	     "\t rbp    = 0x%016" PRIx64 "\n"
	     "\t rsp    = 0x%016" PRIx64 "\n"
	     "\t arg0   = 0x%016" PRIx64 "\n"
	     "\t arg1   = 0x%016" PRIx64 "\n"
	     "\t arg2   = 0x%016" PRIx64 "\n"
	     "\t arg3   = 0x%016" PRIx64 "\n"
	     "\t arg4   = 0x%016" PRIx64 "\n"
	     "\t arg5   = 0x%016" PRIx64 "\n",
	     ret, rbp, rsp, arg0, arg1, arg2, arg3, arg4, arg5);

	PrintAbortPointerCandidate("arg0", arg0);
	PrintAbortPointerCandidate("arg1", arg1);
	PrintAbortPointerCandidate("arg2", arg2);
	PrintAbortPointerCandidate("arg3", arg3);
	PrintAbortPointerCandidate("arg4", arg4);
	PrintAbortPointerCandidate("arg5", arg5);
	PrintAbortPointerArrayCandidate("arg0", arg0);
	PrintAbortPointerArrayCandidate("arg1", arg1);
	PrintAbortPointerArrayCandidate("arg2", arg2);
	PrintAbortPointerArrayCandidate("arg3", arg3);
	PrintAbortPointerArrayCandidate("arg4", arg4);
	PrintAbortPointerArrayCandidate("arg5", arg5);

	if (rbp != 0) {
		Common::Singleton<Loader::RuntimeLinker>::Instance()->StackTrace(rbp, rsp);
	}

	if (rsp != 0) {
		LOGF("Guest abort stack words:\n");
		auto* linker = Common::Singleton<Loader::RuntimeLinker>::Instance();
		for (int i = 0; i < 16; i++) {
			const auto addr = rsp + static_cast<uint64_t>(i) * sizeof(uint64_t);
			if (!Graphics::HostMemoryIsReadable(addr)) {
				break;
			}
			const auto value   = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(addr));
			auto*      program = linker->FindProgramByAddr(value);
			if (program != nullptr) {
				auto module_name = Common::PathToString(program->file_name.filename());
				LOGF("\t [%02d] 0x%016" PRIx64 " %s+0x%016" PRIx64 "\n", i, value,
				     module_name.c_str(), value - program->base_vaddr);
			} else {
				LOGF("\t [%02d] 0x%016" PRIx64 "\n", i, value);
			}
			PrintAbortPointerCandidate(fmt::format("stack[{:02d}]", i).c_str(), value);
			PrintAbortPointerArrayCandidate(fmt::format("stack[{:02d}]", i).c_str(), value);
		}
	}

	EXIT("Guest abort()\n");
	std::abort();
}

static KYTY_SYSV_ABI int* libc_error() {
	PRINT_NAME();

	return Posix::GetErrorAddr();
}

static KYTY_SYSV_ABI void init_env(const InitEnvParams* params) {
	PRINT_NAME();

	if (params == nullptr) {
		g_argc = 0;
		g_argv = nullptr;
		g_envp = nullptr;
		return;
	}

	constexpr int argv_capacity = static_cast<int>(sizeof(params->argv) / sizeof(params->argv[0]));

	EXIT_NOT_IMPLEMENTED(params->argc < 0 || params->argc >= argv_capacity);

	g_argc = params->argc;
	g_argv = params->argv;
	g_envp = params->argv + params->argc + 1;

	LOGF("\t argc = %d\n"
	     "\t argv = 0x%016" PRIx64 "\n"
	     "\t envp = 0x%016" PRIx64 "\n",
	     g_argc, reinterpret_cast<uint64_t>(g_argv), reinterpret_cast<uint64_t>(g_envp));

	for (int i = 0; i < g_argc; i++) {
		LOGF("\t argv[%d] = %s\n", i, g_argv[i] != nullptr ? g_argv[i] : "<null>");
	}
}

static KYTY_SYSV_ABI int atexit(atexit_func_t func) {
	PRINT_NAME();

	if (func != nullptr) {
		Common::Singleton<CContext>::Instance()->atexit.push_front(func);
	}

	return 0;
}

static KYTY_SYSV_ABI int libc_printf(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)

	PRINT_NAME();

	return GetGuestPrintfCtxFunc()(&ctx);
}

static KYTY_SYSV_ABI int puts(const char* s) {
	PRINT_NAME();

	return GetGuestPrintfStdFunc()("%s\n", s);
}

// Guest-only environment (do not touch host environ).
// Pointers returned by getenv must remain stable across later setenv calls, so
// values live in strdup'd buffers rather than std::string::c_str().
static std::mutex g_guest_env_mutex;
static std::unordered_map<std::string, char*> g_guest_env;

static void EnsureDefaultGuestEnvLocked() {
	static bool initialized = false;
	if (initialized) {
		return;
	}
	initialized = true;
	const std::pair<const char*, const char*> defaults[] = {
	    {"HOME", "/app0"},
	    {"USER", "kyty"},
	    {"TMPDIR", "/temp0"},
	    {"TMP", "/temp0"},
	    {"TEMP", "/temp0"},
	    {"XDG_CONFIG_HOME", "/app0/config"},
	    {"XDG_CACHE_HOME", "/temp0"},
	    {"XDG_DATA_HOME", "/app0"},
	};
	for (const auto& [k, v]: defaults) {
		g_guest_env.emplace(k, ::strdup(v));
	}
}

static const char* FindEnvInEnvp(const char* name) {
	if (g_envp == nullptr || name == nullptr) {
		return nullptr;
	}
	const auto name_len = std::strlen(name);
	for (const char* const* ep = g_envp; *ep != nullptr; ++ep) {
		const char* entry = *ep;
		if (std::strncmp(entry, name, name_len) == 0 && entry[name_len] == '=') {
			return entry + name_len + 1;
		}
	}
	return nullptr;
}

static KYTY_SYSV_ABI char* getenv(const char* name) {
	PRINT_NAME();

	LOGF("\t getenv name = %s\n", name != nullptr ? name : "<null>");

	if (name == nullptr || name[0] == '\0') {
		return nullptr;
	}

	{
		std::lock_guard lock(g_guest_env_mutex);
		EnsureDefaultGuestEnvLocked();
		if (const auto it = g_guest_env.find(name); it != g_guest_env.end()) {
			LOGF("\t getenv -> %s\n", it->second != nullptr ? it->second : "<null>");
			return it->second;
		}
	}

	if (const char* from_envp = FindEnvInEnvp(name); from_envp != nullptr) {
		LOGF("\t getenv (envp) -> %s\n", from_envp);
		return const_cast<char*>(from_envp);
	}

	// Not found — NULL is the correct POSIX result (not a crash).
	LOGF("\t getenv -> <unset>\n");
	return nullptr;
}

static KYTY_SYSV_ABI int setenv(const char* name, const char* value, int overwrite) {
	PRINT_NAME();

	LOGF("\t name      = %s\n"
	     "\t value     = %s\n"
	     "\t overwrite = %d\n",
	     (name != nullptr ? name : "<null>"), (value != nullptr ? value : "<null>"), overwrite);

	if (name == nullptr || value == nullptr || name[0] == '\0' ||
	    std::strchr(name, '=') != nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}

	std::lock_guard lock(g_guest_env_mutex);
	EnsureDefaultGuestEnvLocked();
	if (const auto it = g_guest_env.find(name); it != g_guest_env.end()) {
		if (!overwrite) {
			return 0;
		}
		char* copy = ::strdup(value);
		if (copy == nullptr) {
			*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
			return -1;
		}
		::free(it->second);
		it->second = copy;
		return 0;
	}

	char* copy = ::strdup(value);
	if (copy == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
		return -1;
	}
	g_guest_env.emplace(name, copy);
	return 0;
}

static KYTY_SYSV_ABI int unsetenv(const char* name) {
	PRINT_NAME();

	LOGF("\t unsetenv name = %s\n", name != nullptr ? name : "<null>");

	if (name == nullptr || name[0] == '\0' || std::strchr(name, '=') != nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}

	std::lock_guard lock(g_guest_env_mutex);
	EnsureDefaultGuestEnvLocked();
	if (const auto it = g_guest_env.find(name); it != g_guest_env.end()) {
		::free(it->second);
		g_guest_env.erase(it);
	}
	return 0;
}

static KYTY_SYSV_ABI int64_t libc_time(int64_t* timer) {
	const auto now = static_cast<int64_t>(std::time(nullptr));
	if (timer != nullptr) {
		*timer = now;
	}
	return now;
}

static KYTY_SYSV_ABI double libc_difftime(int64_t time1, int64_t time0) {
	return std::difftime(static_cast<std::time_t>(time1), static_cast<std::time_t>(time0));
}

struct GuestTm {
	int tm_sec;
	int tm_min;
	int tm_hour;
	int tm_mday;
	int tm_mon;
	int tm_year;
	int tm_wday;
	int tm_yday;
	int tm_isdst;
};

static_assert(sizeof(GuestTm) == 36);

static GuestTm ToGuestTm(const std::tm& time) {
	return {time.tm_sec,  time.tm_min,  time.tm_hour, time.tm_mday, time.tm_mon,
	        time.tm_year, time.tm_wday, time.tm_yday, time.tm_isdst};
}

static std::tm ToHostTm(const GuestTm& time) {
	std::tm result {};
	result.tm_sec   = time.tm_sec;
	result.tm_min   = time.tm_min;
	result.tm_hour  = time.tm_hour;
	result.tm_mday  = time.tm_mday;
	result.tm_mon   = time.tm_mon;
	result.tm_year  = time.tm_year;
	result.tm_wday  = time.tm_wday;
	result.tm_yday  = time.tm_yday;
	result.tm_isdst = time.tm_isdst;
	return result;
}

static KYTY_SYSV_ABI GuestTm* libc_gmtime(const int64_t* timer) {
	if (timer == nullptr) {
		return nullptr;
	}

	thread_local GuestTm result {};
	std::tm              host_result {};
	const auto           t = static_cast<std::time_t>(*timer);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (_gmtime64_s(&host_result, &t) != 0) {
		return nullptr;
	}
#else
	if (gmtime_r(&t, &host_result) == nullptr) {
		return nullptr;
	}
#endif

	result = ToGuestTm(host_result);
	return &result;
}

static KYTY_SYSV_ABI GuestTm* libc_localtime(const int64_t* timer) {
	if (timer == nullptr) {
		return nullptr;
	}

	thread_local GuestTm result {};
	std::tm              host_result {};
	const auto           t = static_cast<std::time_t>(*timer);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (_localtime64_s(&host_result, &t) != 0) {
		return nullptr;
	}
#else
	if (localtime_r(&t, &host_result) == nullptr) {
		return nullptr;
	}
#endif

	result = ToGuestTm(host_result);
	return &result;
}

static KYTY_SYSV_ABI int64_t libc_mktime(GuestTm* timeptr) {
	if (timeptr == nullptr) {
		return -1;
	}

	auto host_time = ToHostTm(*timeptr);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto result = static_cast<int64_t>(_mktime64(&host_time));
#else
	const auto result = static_cast<int64_t>(std::mktime(&host_time));
#endif

	*timeptr = ToGuestTm(host_time);
	return result;
}

static KYTY_SYSV_ABI size_t libc_strftime(char* str, size_t count, const char* format,
                                          const GuestTm* timeptr) {
	if (str == nullptr || format == nullptr || timeptr == nullptr) {
		return 0;
	}

	const auto host_time = ToHostTm(*timeptr);
	return std::strftime(str, count, format, &host_time);
}

static KYTY_SYSV_ABI void catchReturnFromMain(int status) {
	PRINT_NAME();

	uint64_t ret_addr = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	uint64_t rsp      = 0;
	uint64_t rbp      = 0;
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("movq %%rsp, %0\n\t"
	             "movq %%rbp, %1\n\t"
	             : "=r"(rsp), "=r"(rbp));
#endif

	LOGF("\t status     = %d\n"
	     "\t ret_addr   = 0x%016" PRIx64 "\n"
	     "\t rsp        = 0x%016" PRIx64 "\n"
	     "\t rbp        = 0x%016" PRIx64 "\n",
	     status, ret_addr, rsp, rbp);
	if (rsp != 0) {
		auto* stack = reinterpret_cast<const uint64_t*>(rsp);
		for (uint32_t i = 0; i < 12; i++) {
			LOGF("\t stack[%02" PRIu32 "] = 0x%016" PRIx64 "\n", i, stack[i]);
		}
	}

	::printf("return from main = %d\n", status);
}

using execute_once_func_t = KYTY_SYSV_ABI int (*)(void*, void*, void**);

static std::mutex              g_execute_once_mutex;
static std::condition_variable g_execute_once_cv;

static KYTY_SYSV_ABI int std_execute_once(int* flag, execute_once_func_t func, void* arg) {
	PRINT_NAME();

	if (flag == nullptr || func == nullptr) {
		return 0;
	}

	constexpr int once_init    = 0;
	constexpr int once_done    = 1;
	constexpr int once_running = 2;

	{
		std::unique_lock lock(g_execute_once_mutex);
		while (*flag == once_running) {
			g_execute_once_cv.wait_for(lock, std::chrono::microseconds(10000));
			if (*flag == once_running) {
				lock.unlock();
				LibKernel::KernelDispatchPendingSignalForCurrentThread();
				lock.lock();
			}
		}
		if (*flag == once_done) {
			return 1;
		}
		*flag = once_running;
	}

	void* callback_context = nullptr;
	int   result           = func(nullptr, arg, &callback_context);

	{
		std::lock_guard lock(g_execute_once_mutex);
		*flag = (result != 0 ? once_done : once_init);
	}
	g_execute_once_cv.notify_all();

	return result;
}

static KYTY_SYSV_ABI int cxa_atexit(cxa_destructor_func_t func, void* arg, void* d) {
	PRINT_NAME();

	auto* cc = Common::Singleton<CContext>::Instance();

	CxaDestructor c {};
	c.destructor_func   = func;
	c.destructor_object = arg;
	c.module_id         = d;

	cc->cxa.push_back(c);

	return 0;
}

void KYTY_SYSV_ABI cxa_finalize(void* d) {
	PRINT_NAME();

	auto* cc = Common::Singleton<CContext>::Instance();

	for (auto i = cc->cxa.rbegin(); i != cc->cxa.rend(); ++i) {
		auto& c = *i;
		if ((d == nullptr || c.module_id == d) && c.destructor_func != nullptr) {
			auto* func        = c.destructor_func;
			auto* object      = c.destructor_object;
			c.destructor_func = nullptr;
			func(object);
		}
	}
}

} // namespace LibC

namespace LibcInternalExt {

LIB_VERSION("LibcInternalExt", 1, "LibcInternal", 1, 1);

static uint64_t g_mspace_atomic_id_mask = 0;
static uint64_t g_mstate_table[64]      = {0};

using thread_atexit_destructor_t = KYTY_SYSV_ABI void (*)(void*);

struct ThreadAtexitDestructor {
	thread_atexit_destructor_t destructor;
	void*                      object;
};

static thread_local std::vector<ThreadAtexitDestructor> g_thread_atexit_destructors;

struct Info {
	uint64_t  size;
	uint32_t  unknown1;
	uint32_t  unknown2;
	uint64_t* mspace_atomic_id_mask;
	uint64_t* mstate_table;
};

void KYTY_SYSV_ABI LibcHeapGetTraceInfo(Info* info) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(info->size != 32);

	info->mspace_atomic_id_mask = &g_mspace_atomic_id_mask;
	info->mstate_table          = g_mstate_table;
}

int KYTY_SYSV_ABI LibcInternalExtCxaThreadAtexit(thread_atexit_destructor_t destructor,
                                                 void* object, void* /*module_id*/) {
	PRINT_NAME();

	g_thread_atexit_destructors.push_back({destructor, object});

	return 0;
}

void RunThreadAtexitDestructors() {
	while (!g_thread_atexit_destructors.empty()) {
		auto destructor = g_thread_atexit_destructors.back();
		g_thread_atexit_destructors.pop_back();

		if (destructor.destructor != nullptr) {
			destructor.destructor(destructor.object);
		}
	}
}

LIB_DEFINE(InitLibcInternalExt_1) {
	LIB_FUNC("NWtTN10cJzE", LibcInternalExt::LibcHeapGetTraceInfo);
	LIB_FUNC("qBS714-Jr3g", LibcInternalExt::LibcInternalExtCxaThreadAtexit);
}

} // namespace LibcInternalExt

namespace LibcInternal {

LIB_VERSION("LibcInternal", 1, "LibcInternal", 1, 1);

static uint32_t g_need_flag = 1;

int KYTY_SYSV_ABI vprintf(const char* str, VaList* c) {
	PRINT_NAME();

	return GetGuestVprintfFunc()(str, c);
}

static KYTY_SYSV_ABI int snprintf(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)

	PRINT_NAME();

	return GetGuestSnprintfCtxFunc()(&ctx);
}

// Guest-path aware stdio. The title's sce_module/libc.prx is a loader companion
// whose every FUNC export is `xor eax,eax; ret` (see tooling/native/libc_builder.cpp
// VA 0x50). Eboot imports fopen/freopen/etc. from "libc", so without HLE under
// that library name those calls always return NULL and KernelOpen never sees a
// regular file. Resolve /app0/... through the mount table, then use host FILE*.

static FILE* g_stdin_p  = stdin;
static FILE* g_stdout_p = stdout;
static FILE* g_stderr_p = stderr;

static void SetErrnoFromHost() {
	switch (errno) {
		case ENOENT: *Posix::GetErrorAddr() = Posix::POSIX_ENOENT; break;
		case EACCES: *Posix::GetErrorAddr() = Posix::POSIX_EACCES; break;
		case EEXIST: *Posix::GetErrorAddr() = Posix::POSIX_EEXIST; break;
		case EINVAL: *Posix::GetErrorAddr() = Posix::POSIX_EINVAL; break;
		case EISDIR: *Posix::GetErrorAddr() = Posix::POSIX_EISDIR; break;
		case ENOTDIR: *Posix::GetErrorAddr() = Posix::POSIX_ENOTDIR; break;
		case EBADF: *Posix::GetErrorAddr() = Posix::POSIX_EBADF; break;
		case ENOMEM: *Posix::GetErrorAddr() = Posix::POSIX_ENOMEM; break;
		case EROFS: *Posix::GetErrorAddr() = Posix::POSIX_EROFS; break;
		case EBUSY: *Posix::GetErrorAddr() = Posix::POSIX_EBUSY; break;
		default: *Posix::GetErrorAddr() = Posix::POSIX_EIO; break;
	}
}

static std::string ResolveGuestFilePath(const char* path) {
	if (path == nullptr || path[0] == '\0') {
		return {};
	}
	// Absolute host paths (rare) pass through; guest mounts go via /app0 etc.
	if (path[0] == '/' && std::strncmp(path, "/app0", 5) != 0 && std::strncmp(path, "/hostapp", 8) != 0 &&
	    std::strncmp(path, "/mnt", 4) != 0 && std::strncmp(path, "/download0", 10) != 0 &&
	    std::strncmp(path, "/temp0", 6) != 0 && std::strncmp(path, "/data", 5) != 0 &&
	    std::strncmp(path, "/usb", 4) != 0) {
		return path;
	}
	const auto real = LibKernel::FileSystem::GetRealFilename(path);
	if (real.empty()) {
		return {};
	}
	return Common::PathToString(real);
}

static FILE* MapStdStream(FILE* stream) {
	if (stream == nullptr) {
		return nullptr;
	}
	if (stream == stdin || stream == g_stdin_p) {
		return stdin;
	}
	if (stream == stdout || stream == g_stdout_p) {
		return stdout;
	}
	if (stream == stderr || stream == g_stderr_p) {
		return stderr;
	}
	return stream; // host FILE* from our fopen/freopen
}

int KYTY_SYSV_ABI fflush(FILE* stream) {
	PRINT_NAME();

	if (stream == nullptr) {
		return ::fflush(nullptr);
	}
	FILE* host = MapStdStream(stream);
	return ::fflush(host);
}

static KYTY_SYSV_ABI void* malloc_hle(size_t size) {
	PRINT_NAME();
	if (size == 0) {
		size = 1;
	}
	void* p = ::malloc(size);
	if (p == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
	}
	return p;
}

static KYTY_SYSV_ABI void* calloc_hle(size_t nmemb, size_t size) {
	PRINT_NAME();
	void* p = ::calloc(nmemb == 0 ? 1 : nmemb, size == 0 ? 1 : size);
	if (p == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
	}
	return p;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
struct WinAlignedBlock {
	void*  raw       = nullptr;
	size_t size      = 0;
	size_t alignment = 0;
};

std::mutex                                  g_win_aligned_mutex;
std::unordered_map<void*, WinAlignedBlock>  g_win_aligned;

static void* WinAlignPointer(void* raw, size_t alignment) {
	const auto base = reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*);
	const auto mask = static_cast<std::uintptr_t>(alignment) - 1;
	return reinterpret_cast<void*>((base + mask) & ~mask);
}

static bool WinTakeAligned(void* ptr, WinAlignedBlock* out) {
	std::lock_guard lock(g_win_aligned_mutex);
	auto            it = g_win_aligned.find(ptr);
	if (it == g_win_aligned.end()) {
		return false;
	}
	*out = it->second;
	g_win_aligned.erase(it);
	return true;
}
#endif

static KYTY_SYSV_ABI void* realloc_hle(void* ptr, size_t size) {
	PRINT_NAME();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (ptr != nullptr) {
		WinAlignedBlock block {};
		if (WinTakeAligned(ptr, &block)) {
			if (size == 0) {
				::free(block.raw);
				return nullptr;
			}
			const size_t extra = block.alignment - 1 + sizeof(void*);
			if (size > static_cast<size_t>(-1) - extra) {
				std::lock_guard lock(g_win_aligned_mutex);
				g_win_aligned.insert_or_assign(ptr, block);
				*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
				return nullptr;
			}
			void* raw = ::malloc(size + extra);
			if (raw == nullptr) {
				std::lock_guard lock(g_win_aligned_mutex);
				g_win_aligned.insert_or_assign(ptr, block);
				*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
				return nullptr;
			}
			void* next = WinAlignPointer(raw, block.alignment);
			std::memcpy(next, ptr, std::min(block.size, size));
			{
				std::lock_guard lock(g_win_aligned_mutex);
				g_win_aligned.insert_or_assign(next, WinAlignedBlock {raw, size, block.alignment});
			}
			::free(block.raw);
			return next;
		}
	}
#endif
	if (size == 0) {
		::free(ptr);
		return nullptr;
	}
	void* p = ::realloc(ptr, size);
	if (p == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
	}
	return p;
}

static KYTY_SYSV_ABI void free_hle(void* ptr) {
	PRINT_NAME();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (ptr != nullptr) {
		WinAlignedBlock block {};
		if (WinTakeAligned(ptr, &block)) {
			::free(block.raw);
			return;
		}
	}
#endif
	::free(ptr);
}

static KYTY_SYSV_ABI int posix_memalign_hle(void** memptr, size_t alignment, size_t size) {
	PRINT_NAME();
	if (memptr == nullptr) {
		return Posix::POSIX_EINVAL;
	}
	if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
		return Posix::POSIX_EINVAL;
	}
	if (size == 0) {
		size = 1;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Over-allocate so the guest pointer is aligned and free_hle can recover the
	// malloc base. ::free cannot release a _aligned_malloc pointer.
	const size_t extra = alignment - 1 + sizeof(void*);
	if (size > static_cast<size_t>(-1) - extra) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
		*memptr                = nullptr;
		return Posix::POSIX_ENOMEM;
	}
	void* raw = ::malloc(size + extra);
	if (raw == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
		*memptr                = nullptr;
		return Posix::POSIX_ENOMEM;
	}
	void* user = WinAlignPointer(raw, alignment);
	{
		std::lock_guard lock(g_win_aligned_mutex);
		g_win_aligned.insert_or_assign(user, WinAlignedBlock {raw, size, alignment});
	}
	*memptr = user;
	return 0;
#else
	void* p = nullptr;
	const int rc = ::posix_memalign(&p, alignment, size);
	if (rc != 0) {
		*Posix::GetErrorAddr() = rc;
		*memptr = nullptr;
		return rc;
	}
	*memptr = p;
	return 0;
#endif
}

static KYTY_SYSV_ABI void* aligned_alloc_hle(size_t alignment, size_t size) {
	PRINT_NAME();
	void* p = nullptr;
	if (posix_memalign_hle(&p, alignment, size) != 0) {
		return nullptr;
	}
	return p;
}

static KYTY_SYSV_ABI FILE* fopen_hle(const char* path, const char* mode) {
	PRINT_NAME();

	if (path == nullptr || mode == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return nullptr;
	}

	const auto real = ResolveGuestFilePath(path);
	if (real.empty()) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOENT;
		LOGF_COLOR(Log::Color::Red, "\tfopen: %s [no mount]\n", path);
		return nullptr;
	}

	FILE* file = ::fopen(real.c_str(), mode);
	LOGF_COLOR(file != nullptr ? Log::Color::Green : Log::Color::Red, "\tOpen: %s, %s\n",
	           real.c_str(), file != nullptr ? "[ok]" : "[fail]");
	if (file == nullptr) {
		SetErrnoFromHost();
	}
	return file;
}

static KYTY_SYSV_ABI FILE* freopen_hle(const char* path, const char* mode, FILE* stream) {
	PRINT_NAME();

	if (path == nullptr || mode == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return nullptr;
	}

	const auto real = ResolveGuestFilePath(path);
	if (real.empty()) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOENT;
		LOGF_COLOR(Log::Color::Red, "\tfreopen: %s [no mount]\n", path);
		return nullptr;
	}

	FILE* host = MapStdStream(stream);
	// libc.prx exports zeroed FILE blobs; never pass those to host freopen.
	if (host != stdin && host != stdout && host != stderr) {
		host = (std::strpbrk(mode, "wa") != nullptr) ? stderr : stdin;
		LOGF("\tfreopen: remapping foreign stream %p -> %s\n", static_cast<void*>(stream),
		     host == stderr ? "stderr" : "stdin");
	}

	FILE* file = ::freopen(real.c_str(), mode, host);
	LOGF_COLOR(file != nullptr ? Log::Color::Green : Log::Color::Red, "\tOpen: %s (freopen), %s\n",
	           real.c_str(), file != nullptr ? "[ok]" : "[fail]");
	if (file == nullptr) {
		SetErrnoFromHost();
		return nullptr;
	}
	if (host == stdin) {
		g_stdin_p = file;
	} else if (host == stdout) {
		g_stdout_p = file;
	} else if (host == stderr) {
		g_stderr_p = file;
	}
	return file;
}

static KYTY_SYSV_ABI FILE* fdopen_hle(int fd, const char* mode) {
	PRINT_NAME();
	if (mode == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return nullptr;
	}
	// Kyty descriptors are not host fds; only allow stdio fds 0/1/2.
	if (fd == 0) {
		return stdin;
	}
	if (fd == 1) {
		return stdout;
	}
	if (fd == 2) {
		return stderr;
	}
	*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
	return nullptr;
}

static KYTY_SYSV_ABI int fclose_hle(FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return EOF;
	}
	FILE* host = MapStdStream(stream);
	if (host == stdin || host == stdout || host == stderr) {
		return 0; // do not close std streams
	}
	const int rc = ::fclose(host);
	if (rc != 0) {
		SetErrnoFromHost();
	}
	return rc;
}

static KYTY_SYSV_ABI size_t fread_hle(void* ptr, size_t size, size_t nmemb, FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return 0;
	}
	return ::fread(ptr, size, nmemb, MapStdStream(stream));
}

static KYTY_SYSV_ABI size_t fwrite_hle(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return 0;
	}
	return ::fwrite(ptr, size, nmemb, MapStdStream(stream));
}

static KYTY_SYSV_ABI int fseek_hle(FILE* stream, long offset, int whence) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return -1;
	}
	const int rc = ::fseek(MapStdStream(stream), offset, whence);
	if (rc != 0) {
		SetErrnoFromHost();
	}
	return rc;
}

static KYTY_SYSV_ABI long ftell_hle(FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return -1;
	}
	const long rc = ::ftell(MapStdStream(stream));
	if (rc < 0) {
		SetErrnoFromHost();
	}
	return rc;
}

static KYTY_SYSV_ABI int fseeko_hle(FILE* stream, int64_t offset, int whence) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return -1;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Host off_t is 32-bit here; the Windows shim takes long long.
	const int rc = ::fseeko(MapStdStream(stream), offset, whence);
#else
	const int rc = ::fseeko(MapStdStream(stream), static_cast<off_t>(offset), whence);
#endif
	if (rc != 0) {
		SetErrnoFromHost();
	}
	return rc;
}

static KYTY_SYSV_ABI int64_t ftello_hle(FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return -1;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto rc = ::ftello(MapStdStream(stream));
#else
	const off_t rc = ::ftello(MapStdStream(stream));
#endif
	if (rc < 0) {
		SetErrnoFromHost();
	}
	return static_cast<int64_t>(rc);
}

static KYTY_SYSV_ABI int fileno_hle(FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return -1;
	}
	FILE* host = MapStdStream(stream);
	if (host == stdin) {
		return 0;
	}
	if (host == stdout) {
		return 1;
	}
	if (host == stderr) {
		return 2;
	}
	const int fd = ::fileno(host);
	if (fd < 0) {
		SetErrnoFromHost();
	}
	return fd;
}

static KYTY_SYSV_ABI int feof_hle(FILE* stream) {
	PRINT_NAME();
	return stream == nullptr ? 0 : ::feof(MapStdStream(stream));
}

static KYTY_SYSV_ABI int ferror_hle(FILE* stream) {
	PRINT_NAME();
	return stream == nullptr ? 0 : ::ferror(MapStdStream(stream));
}

static KYTY_SYSV_ABI void clearerr_hle(FILE* stream) {
	PRINT_NAME();
	if (stream != nullptr) {
		::clearerr(MapStdStream(stream));
	}
}

static KYTY_SYSV_ABI int setvbuf_hle(FILE* stream, char* buf, int type, size_t size) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return -1;
	}
	return ::setvbuf(MapStdStream(stream), buf, type, size);
}

static KYTY_SYSV_ABI void setbuf_hle(FILE* stream, char* buf) {
	PRINT_NAME();
	if (stream != nullptr) {
		::setbuf(MapStdStream(stream), buf);
	}
}

static KYTY_SYSV_ABI int ungetc_hle(int c, FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		return EOF;
	}
	return ::ungetc(c, MapStdStream(stream));
}

static KYTY_SYSV_ABI int fputs_hle(const char* s, FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return EOF;
	}
	return ::fputs(s != nullptr ? s : "", MapStdStream(stream));
}

static KYTY_SYSV_ABI int fputc_hle(int c, FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return EOF;
	}
	return ::fputc(c, MapStdStream(stream));
}

static KYTY_SYSV_ABI int fgetc_hle(FILE* stream) {
	PRINT_NAME();
	if (stream == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EBADF;
		return EOF;
	}
	return ::fgetc(MapStdStream(stream));
}

static KYTY_SYSV_ABI char* fgets_hle(char* s, int n, FILE* stream) {
	PRINT_NAME();
	if (s == nullptr || stream == nullptr || n <= 0) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return nullptr;
	}
	return ::fgets(s, n, MapStdStream(stream));
}

static KYTY_SYSV_ABI void rewind_hle(FILE* stream) {
	PRINT_NAME();
	if (stream != nullptr) {
		::rewind(MapStdStream(stream));
	}
}

static KYTY_SYSV_ABI int remove_hle(const char* path) {
	PRINT_NAME();
	if (path == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}
	const auto real = ResolveGuestFilePath(path);
	if (real.empty()) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOENT;
		LOGF_COLOR(Log::Color::Red, "\tremove: %s [no mount]\n", path);
		return -1;
	}
	const int rc = ::remove(real.c_str());
	LOGF_COLOR(rc == 0 ? Log::Color::Green : Log::Color::Red, "\tremove: %s, %s\n", real.c_str(),
	           rc == 0 ? "[ok]" : "[fail]");
	if (rc != 0) {
		SetErrnoFromHost();
	}
	return rc;
}

static KYTY_SYSV_ABI int vsnprintf_hle(char* buf, size_t size, const char* fmt, VaList* ap) {
	PRINT_NAME();
	if (fmt == nullptr || (buf == nullptr && size != 0)) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}
	return GuestVsnprintf(buf, size, fmt, ap);
}

static KYTY_SYSV_ABI int vsprintf_hle(char* buf, const char* fmt, VaList* ap) {
	PRINT_NAME();
	if (buf == nullptr || fmt == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}
	return GuestVsnprintf(buf, static_cast<size_t>(-1), fmt, ap);
}

static KYTY_SYSV_ABI int sprintf_hle(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	PRINT_NAME();
	auto* buf = VaArg_ptr<char>(&ctx.va_list);
	auto* fmt = VaArg_ptr<const char>(&ctx.va_list);
	if (buf == nullptr || fmt == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}
	return GuestVsnprintf(buf, static_cast<size_t>(-1), fmt, &ctx.va_list);
}

static KYTY_SYSV_ABI int vfprintf_hle(FILE* stream, const char* fmt, VaList* ap) {
	PRINT_NAME();
	if (stream == nullptr || fmt == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}
	// Format once to learn the length (va_list copy), then again into a buffer
	// big enough for the whole string. fprintf does not truncate.
	VaList          sized = *ap;
	const int       n     = GuestVsnprintf(nullptr, 0, fmt, &sized);
	FILE*           host  = MapStdStream(stream);
	if (n < 0) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}
	std::vector<char> storage(static_cast<size_t>(n) + 1U);
	GuestVsnprintf(storage.data(), storage.size(), fmt, ap);
	LOGF("\tfprintf: %s", storage.data());
	return ::fputs(storage.data(), host) >= 0 ? n : -1;
}

static KYTY_SYSV_ABI int fprintf_hle(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	PRINT_NAME();
	auto* stream = VaArg_ptr<FILE>(&ctx.va_list);
	auto* fmt    = VaArg_ptr<const char>(&ctx.va_list);
	if (stream == nullptr || fmt == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}
	return vfprintf_hle(stream, fmt, &ctx.va_list);
}

void* KYTY_SYSV_ABI memset(void* s, int c, size_t n) {
	PRINT_NAME();

	return ::memset(s, c, n);
}

void* KYTY_SYSV_ABI memcpy(void* dest, const void* src, size_t n) {
	return ::memcpy(dest, src, n);
}

void* KYTY_SYSV_ABI memmove(void* dest, const void* src, size_t n) {
	return ::memmove(dest, src, n);
}

int KYTY_SYSV_ABI memcmp(const void* s1, const void* s2, size_t n) {
	return ::memcmp(s1, s2, n);
}

int KYTY_SYSV_ABI strcmp(const char* s1, const char* s2) {
	return ::strcmp(s1, s2);
}

int KYTY_SYSV_ABI strncmp(const char* s1, const char* s2, size_t n) {
	return ::strncmp(s1, s2, n);
}

static KYTY_SYSV_ABI int strcasecmp_hle(const char* s1, const char* s2) {
	return ::strcasecmp(s1, s2);
}

static KYTY_SYSV_ABI int strncasecmp_hle(const char* s1, const char* s2, size_t n) {
	return ::strncasecmp(s1, s2, n);
}

static KYTY_SYSV_ABI int tolower_hle(int c) {
	return ::tolower(c);
}

static KYTY_SYSV_ABI int toupper_hle(int c) {
	return ::toupper(c);
}

size_t KYTY_SYSV_ABI strlen(const char* s) {
	return ::strlen(s);
}

char* KYTY_SYSV_ABI strcpy(char* dest, const char* src) {
	return ::strcpy(dest, src);
}

char* KYTY_SYSV_ABI strncpy(char* dest, const char* src, size_t count) {
	return ::strncpy(dest, src, count);
}

char* KYTY_SYSV_ABI strcat(char* dest, const char* src) {
	return ::strcat(dest, src);
}

char* KYTY_SYSV_ABI strncat_hle(char* dest, const char* src, size_t n) {
	return ::strncat(dest, src, n);
}

const char* KYTY_SYSV_ABI strchr(const char* s, int c) {
	return ::strchr(s, c);
}

char* KYTY_SYSV_ABI strrchr(const char* s, int c) {
	return const_cast<char*>(::strrchr(s, c));
}

char* KYTY_SYSV_ABI strstr(const char* haystack, const char* needle) {
	static int log_count = 0;
	if (log_count++ < 8) {
		LOGF("LibcInternal::strstr(\"%s\", \"%s\")\n", (haystack != nullptr ? haystack : "<null>"),
		     (needle != nullptr ? needle : "<null>"));
	}
	return const_cast<char*>(::strstr(haystack, needle));
}


static KYTY_SYSV_ABI char* strtok_hle(char* str, const char* delim) {
	// libc.prx exports strtok as xor-eax-eax;ret. RetroArch nested_list /
	// string_split use strtok_r; keep strtok covered the same way.
	return ::strtok(str, delim);
}

static KYTY_SYSV_ABI char* strtok_r_hle(char* str, const char* delim, char** saveptr) {
	// Critical for RetroArch core_option_manager: nested_list_add_item ->
	// string_split_noalloc -> strtok_r. Without HLE under "libc", exact resolve
	// binds the libc.prx null stub, nested_list_add_item fails, core_options
	// stays NULL, GET_VARIABLE returns true with value=NULL, and fceumm does
	// strcmp(NULL, "disabled") on fceumm_apu_*.
	return ::strtok_r(str, delim, saveptr);
}

char* KYTY_SYSV_ABI strdup_hle(const char* s) {
	PRINT_NAME();
	static int log_count = 0;
	if (log_count++ < 64) {
		LOGF("\t strdup src = %s\n", s != nullptr ? s : "<null>");
	}
	if (s == nullptr) {
		// POSIX leaves strdup(NULL) undefined. RetroArch inserts the result into
		// an FNV string map without a NULL check; return an allocated "" so boot
		// can proceed past the NULL deref at guest pc ~0x900852fbb.
		LOGF("\t strdup(NULL) -> allocated empty string\n");
		char* empty = ::strdup("");
		if (empty == nullptr) {
			*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
		}
		return empty;
	}
	char* copy = ::strdup(s);
	if (copy == nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_ENOMEM;
	}
	return copy;
}

long KYTY_SYSV_ABI strtol(const char* str, char** endptr, int base) {
	static int log_count = 0;
	if (log_count++ < 16) {
		LOGF("LibcInternal::strtol(\"%s\", base=%d)\n", (str != nullptr ? str : "<null>"), base);
	}
	return ::strtol(str, endptr, base);
}

unsigned long KYTY_SYSV_ABI strtoul(const char* str, char** endptr, int base) {
	static int log_count = 0;
	if (log_count++ < 16) {
		LOGF("LibcInternal::strtoul(\"%s\", base=%d)\n", (str != nullptr ? str : "<null>"), base);
	}
	return ::strtoul(str, endptr, base);
}

int KYTY_SYSV_ABI atoi(const char* str) {
	return ::atoi(str);
}

float KYTY_SYSV_ABI sinf(float x) {
	return std::sin(x);
}

float KYTY_SYSV_ABI cosf(float x) {
	return std::cos(x);
}

void KYTY_SYSV_ABI sincosf(float x, float* sinp, float* cosp) {
	if (sinp != nullptr) {
		*sinp = std::sin(x);
	}
	if (cosp != nullptr) {
		*cosp = std::cos(x);
	}
}

double KYTY_SYSV_ABI sin(double x) {
	return std::sin(x);
}

double KYTY_SYSV_ABI cos(double x) {
	return std::cos(x);
}

void KYTY_SYSV_ABI sincos(double x, double* sinp, double* cosp) {
	if (sinp != nullptr) {
		*sinp = std::sin(x);
	}
	if (cosp != nullptr) {
		*cosp = std::cos(x);
	}
}

int KYTY_SYSV_ABI LibcHeapErrorReportForGame(uint64_t msp, uint64_t ptr, uint64_t error,
                                             uint64_t arg3, uint64_t arg4, uint64_t arg5) {
	PRINT_NAME();

	LOGF("\t temporary: heap error report ignored, msp=0x%016" PRIx64 ", ptr=0x%016" PRIx64
	     ", error=0x%016" PRIx64 ", args=(0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 ")\n",
	     msp, ptr, error, arg3, arg4, arg5);

	return 0;
}

// FreeBSD/Orbis libgen: may mutate path and return a pointer into it.
// Guard NULL/empty like the title's own trampolines (return ".").
static KYTY_SYSV_ABI char* basename_hle(char* path) {
	PRINT_NAME();
	LOGF("\t basename path = %s\n", path != nullptr ? path : "<null>");
	if (path == nullptr || path[0] == '\0') {
		return const_cast<char*>(".");
	}
	return ::basename(path);
}

static KYTY_SYSV_ABI char* dirname_hle(char* path) {
	PRINT_NAME();
	LOGF("\t dirname path = %s\n", path != nullptr ? path : "<null>");
	if (path == nullptr || path[0] == '\0') {
		return const_cast<char*>(".");
	}
	return ::dirname(path);
}

static KYTY_SYSV_ABI void* memchr_hle(const void* s, int c, size_t n) {
	return const_cast<void*>(::memchr(s, c, n));
}

// memrchr is glibc/BSD-only; macOS and Windows do not provide it.
static void* Memrchr(const void* s, int c, size_t n) {
	if (s == nullptr || n == 0) {
		return nullptr;
	}
	const auto* bytes = static_cast<const unsigned char*>(s);
	const auto  uc    = static_cast<unsigned char>(c);
	for (size_t i = n; i-- > 0;) {
		if (bytes[i] == uc) {
			return const_cast<unsigned char*>(bytes + i);
		}
	}
	return nullptr;
}

static KYTY_SYSV_ABI void* memrchr_hle(const void* s, int c, size_t n) {
	return Memrchr(s, c, n);
}

static KYTY_SYSV_ABI char* strcasestr_hle(const char* haystack, const char* needle) {
	PRINT_NAME();
	if (haystack == nullptr || needle == nullptr) {
		return nullptr;
	}
	return const_cast<char*>(::strcasestr(haystack, needle));
}

static KYTY_SYSV_ABI void srandom_hle(unsigned seed) {
	::srandom(seed);
}

static KYTY_SYSV_ABI long random_hle() {
	return ::random();
}

// Guest struct option matches Linux/FreeBSD LP64 layout.
struct GuestOption {
	const char* name;
	int         has_arg;
	int*        flag;
	int         val;
};

int KYTY_SYSV_ABI getopt_long(int argc, char* const* argv, const char* optstring,
                              const GuestOption* longopts, int* longindex) {
	PRINT_NAME();

	LOGF("\t getopt_long argc=%d optind=%d optstring=%s\n", argc, ::optind,
	     optstring != nullptr ? optstring : "<null>");

	if (argv == nullptr || optstring == nullptr || argc <= 0) {
		return -1;
	}
	if (::optind <= 0) {
		::optind = 1;
	}

	// Host struct option matches GuestOption (LP64, and LLP64 for this layout).
	static_assert(sizeof(GuestOption) == sizeof(struct option));
	static_assert(offsetof(GuestOption, name) == offsetof(struct option, name));
	static_assert(offsetof(GuestOption, has_arg) == offsetof(struct option, has_arg));
	static_assert(offsetof(GuestOption, flag) == offsetof(struct option, flag));
	static_assert(offsetof(GuestOption, val) == offsetof(struct option, val));
	return ::getopt_long(argc, const_cast<char**>(argv), optstring,
	                     reinterpret_cast<const struct option*>(longopts), longindex);
}

LIB_DEFINE(InitLibcInternal_1) {
	LibcInternalExt::InitLibcInternalExt_1(s);

	LIB_OBJECT("ZT4ODD2Ts9o", &LibcInternal::g_need_flag);
	LIB_OBJECT("2sWzhYqFH4E", stdout);
	LIB_OBJECT("zG0BNJOZdm4", &::optarg);   // optarg
	LIB_OBJECT("zCnSJWp-Qj8", &::optind);   // optind
	LIB_OBJECT("yaFXXViLWPw", &::opterr);   // opterr
	LIB_OBJECT("FwzVaZ8Vnus", &::optopt);   // optopt

	LIB_FUNC("GMpvxPFW924", LibcInternal::vprintf);
	LIB_FUNC("MUjC4lbHrK4", LibcInternal::fflush);
	LIB_FUNC("gQX+4GDQjpM", LibcInternal::malloc_hle);          // malloc
	LIB_FUNC("2X5agFjKxMc", LibcInternal::calloc_hle);          // calloc
	LIB_FUNC("Y7aJ1uydPMo", LibcInternal::realloc_hle);         // realloc
	LIB_FUNC("tIhsqj0qsFE", LibcInternal::free_hle);            // free
	LIB_FUNC("cVSk9y8URbc", LibcInternal::posix_memalign_hle);  // posix_memalign
	LIB_FUNC("2Btkg8k24Zg", LibcInternal::aligned_alloc_hle);   // aligned_alloc
	LIB_FUNC("Q2V+iqvjgC0", LibcInternal::vsnprintf_hle);       // vsnprintf
	LIB_FUNC("jbz9I9vkqkk", LibcInternal::vsprintf_hle);        // vsprintf
	LIB_FUNC("tcVi5SivF7Q", LibcInternal::sprintf_hle);         // sprintf
	LIB_FUNC("xeYO4u7uyJ0", LibcInternal::fopen_hle);   // fopen
	LIB_FUNC("gkWgn0p1AfU", LibcInternal::freopen_hle); // freopen
	LIB_FUNC("qdlHjTa9hQ4", LibcInternal::fdopen_hle);  // fdopen
	LIB_FUNC("uodLYyUip20", LibcInternal::fclose_hle);  // fclose
	LIB_FUNC("lbB+UlZqVG0", LibcInternal::fread_hle);   // fread
	LIB_FUNC("MpxhMh8QFro", LibcInternal::fwrite_hle);  // fwrite
	LIB_FUNC("rQFVBXp-Cxg", LibcInternal::fseek_hle);   // fseek
	LIB_FUNC("Qazy8LmXTvw", LibcInternal::ftell_hle);   // ftell
	LIB_FUNC("pkYiKw09PRA", LibcInternal::fseeko_hle);  // fseeko
	LIB_FUNC("5qP1iVQkdck", LibcInternal::ftello_hle);  // ftello
	LIB_FUNC("Fm-dmyywH9Q", LibcInternal::fileno_hle);  // fileno
	LIB_FUNC("LxcEU+ICu8U", LibcInternal::feof_hle);    // feof
	LIB_FUNC("AHxyhN96dy4", LibcInternal::ferror_hle);  // ferror
	LIB_FUNC("St9nbxSoezk", LibcInternal::clearerr_hle); // clearerr
	LIB_FUNC("QMFyLoqNxIg", LibcInternal::setvbuf_hle); // setvbuf
	LIB_FUNC("vZMcAfsA31I", LibcInternal::setbuf_hle);  // setbuf
	LIB_FUNC("-LFO7jhD5CE", LibcInternal::ungetc_hle);  // ungetc
	LIB_FUNC("QrZZdJ8XsX0", LibcInternal::fputs_hle);   // fputs
	LIB_FUNC("aZK8lNei-Qw", LibcInternal::fputc_hle);   // fputc
	LIB_FUNC("AEuF3F2f8TA", LibcInternal::fgetc_hle);   // fgetc
	LIB_FUNC("KdP-nULpuGw", LibcInternal::fgets_hle);   // fgets
	LIB_FUNC("3QIPIh-GDjw", LibcInternal::rewind_hle);  // rewind
	LIB_FUNC("MZO7FXyAPU8", LibcInternal::remove_hle);  // remove
	LIB_FUNC("fffwELXNVFA", LibcInternal::fprintf_hle); // fprintf
	LIB_FUNC("pDBDcY6uLSA", LibcInternal::vfprintf_hle); // vfprintf
	LIB_OBJECT("bgAcsbcEznc", &LibcInternal::g_stdin_p);  // __stdinp
	LIB_OBJECT("zqJhBxAKfsc", &LibcInternal::g_stdout_p); // __stdoutp
	LIB_OBJECT("as8Od-tH1BI", &LibcInternal::g_stderr_p); // __stderrp
	LIB_FUNC("8zTFvBIAIN8", LibcInternal::memset);
	LIB_FUNC("Q3VBxCXhUHs", LibcInternal::memcpy);
	LIB_FUNC("+P6FRGH4LfA", LibcInternal::memmove);
	LIB_FUNC("DfivPArhucg", LibcInternal::memcmp);
	LIB_FUNC("Ovb2dSJOAuE", LibcInternal::strcmp); // strcmp
	LIB_FUNC("aesyjrHVWy4", LibcInternal::strncmp); // strncmp
	LIB_FUNC("AV6ipCNa4Rw", LibcInternal::strcasecmp_hle); // strcasecmp
	LIB_FUNC("pXvbDfchu6k", LibcInternal::strncasecmp_hle); // strncasecmp
	LIB_FUNC("PqF+kHW-2WQ", LibcInternal::tolower_hle); // tolower
	LIB_FUNC("TYE4irxSmko", LibcInternal::toupper_hle); // toupper
	LIB_FUNC("j4ViWNHEgww", LibcInternal::strlen);
	LIB_FUNC("kiZSXIWd9vg", LibcInternal::strcpy); // strcpy (was wrong NID 5Xa2ACNECdo)
	LIB_FUNC("6sJWiWSRuqk", LibcInternal::strncpy);
	LIB_FUNC("Ls4tzzhimqQ", LibcInternal::strcat);
	LIB_FUNC("kHg45qPC6f0", LibcInternal::strncat_hle); // strncat
	LIB_FUNC("ob5xAW4ln-0", LibcInternal::strchr);
	LIB_FUNC("9yDWMxEFdJU", LibcInternal::strrchr);
	LIB_FUNC("viiwFMaNamA", LibcInternal::strstr);
	LIB_FUNC("oVkZ8W8-Q8A", LibcInternal::strtok_hle);   // strtok
	LIB_FUNC("enqPGLfmVNU", LibcInternal::strtok_r_hle); // strtok_r
	LIB_FUNC("g7zzzLDYGw0", LibcInternal::strdup_hle); // strdup
	LIB_FUNC("mXlxhmLNMPg", LibcInternal::strtol);
	LIB_FUNC("QxmSHBCuKTk", LibcInternal::strtoul);
	LIB_FUNC("zlfEH8FmyUA", LibcInternal::strtoul);
	LIB_FUNC("fPxypibz2MY", LibcInternal::atoi);
	LIB_FUNC("Q4rRL34CEeE", LibcInternal::sinf);
	LIB_FUNC("-P6FNMzk2Kc", LibcInternal::cosf);
	LIB_FUNC("pztV4AF18iI", LibcInternal::sincosf);
	LIB_FUNC("H8ya2H00jbI", LibcInternal::sin);
	LIB_FUNC("2WE3BTYVwKM", LibcInternal::cos);
	LIB_FUNC("jMB7EFyu30Y", LibcInternal::sincos);
	LIB_FUNC("eLdDw6l0-bU", LibcInternal::snprintf);

	LIB_FUNC("L1SBTkC+Cvw", LibC::abort);
	LIB_FUNC("tsvEmnenz48", LibC::cxa_atexit);
	LIB_FUNC("H2e8t5ScQGc", LibC::cxa_finalize);
	LIB_FUNC("DiGVep5yB5w", LibC::std_execute_once);

	LIB_FUNC("al3JzFI9MQ0", LibcInternal::LibcHeapErrorReportForGame);
	LIB_FUNC("8VVXJxB5nlk", LibcInternal::getopt_long); // getopt_long
	LIB_FUNC("smbQukfxYJM", LibC::getenv);              // getenv
	LIB_FUNC("CRJcH8CnPSI", LibC::unsetenv);            // unsetenv
	LIB_FUNC("rg5JEBlKCuo", LibcInternal::basename_hle); // basename
	LIB_FUNC("E4wZaG1zSFc", LibcInternal::dirname_hle);  // dirname
	LIB_FUNC("8u8lPzUEq+U", LibcInternal::memchr_hle);   // memchr
	LIB_FUNC("5G2ONUzRgjY", LibcInternal::memrchr_hle);  // memrchr
	LIB_FUNC("eDmbt0P120g", LibcInternal::strcasestr_hle); // strcasestr
	LIB_FUNC("w1o05aHJT4c", LibcInternal::random_hle);   // random
	LIB_FUNC("sPC7XE6hfFY", LibcInternal::srandom_hle);  // srandom
}

} // namespace LibcInternal

namespace LibC {

LIB_DEFINE(InitLibC_1) {
	LibcInternal::InitLibcInternal_1(s);

	LIB_OBJECT("P330P3dFF68", &LibC::g_need_flag);

	LIB_FUNC("uMei1W9uyNo", LibC::exit);
	LIB_FUNC("L1SBTkC+Cvw", LibC::abort);
	LIB_FUNC("9BcDykPmo1I", LibC::libc_error);
	// Override sce_module/libc.prx null stubs (xor eax,eax;ret) for RetroArch stdio/heap.
	LIB_FUNC("gQX+4GDQjpM", LibcInternal::malloc_hle);
	LIB_FUNC("2X5agFjKxMc", LibcInternal::calloc_hle);
	LIB_FUNC("Y7aJ1uydPMo", LibcInternal::realloc_hle);
	LIB_FUNC("tIhsqj0qsFE", LibcInternal::free_hle);
	LIB_FUNC("cVSk9y8URbc", LibcInternal::posix_memalign_hle);
	LIB_FUNC("2Btkg8k24Zg", LibcInternal::aligned_alloc_hle);
	LIB_FUNC("Q2V+iqvjgC0", LibcInternal::vsnprintf_hle);
	LIB_FUNC("jbz9I9vkqkk", LibcInternal::vsprintf_hle);
	LIB_FUNC("tcVi5SivF7Q", LibcInternal::sprintf_hle);
	LIB_FUNC("eLdDw6l0-bU", LibcInternal::snprintf);
	LIB_FUNC("8zTFvBIAIN8", LibcInternal::memset);
	LIB_FUNC("Q3VBxCXhUHs", LibcInternal::memcpy);
	LIB_FUNC("+P6FRGH4LfA", LibcInternal::memmove);
	LIB_FUNC("DfivPArhucg", LibcInternal::memcmp);
	LIB_FUNC("8u8lPzUEq+U", LibcInternal::memchr_hle);
	LIB_FUNC("Ovb2dSJOAuE", LibcInternal::strcmp); // strcmp
	LIB_FUNC("aesyjrHVWy4", LibcInternal::strncmp); // strncmp
	LIB_FUNC("AV6ipCNa4Rw", LibcInternal::strcasecmp_hle); // strcasecmp
	LIB_FUNC("pXvbDfchu6k", LibcInternal::strncasecmp_hle); // strncasecmp
	LIB_FUNC("PqF+kHW-2WQ", LibcInternal::tolower_hle); // tolower
	LIB_FUNC("TYE4irxSmko", LibcInternal::toupper_hle); // toupper
	LIB_FUNC("j4ViWNHEgww", LibcInternal::strlen);
	LIB_FUNC("kiZSXIWd9vg", LibcInternal::strcpy);
	LIB_FUNC("6sJWiWSRuqk", LibcInternal::strncpy);
	LIB_FUNC("Ls4tzzhimqQ", LibcInternal::strcat);
	LIB_FUNC("kHg45qPC6f0", LibcInternal::strncat_hle);
	LIB_FUNC("ob5xAW4ln-0", LibcInternal::strchr);
	LIB_FUNC("9yDWMxEFdJU", LibcInternal::strrchr);
	LIB_FUNC("viiwFMaNamA", LibcInternal::strstr);
	LIB_FUNC("oVkZ8W8-Q8A", LibcInternal::strtok_hle);   // strtok (override libc.prx)
	LIB_FUNC("enqPGLfmVNU", LibcInternal::strtok_r_hle); // strtok_r (override libc.prx)
	LIB_FUNC("xeYO4u7uyJ0", LibcInternal::fopen_hle);
	LIB_FUNC("gkWgn0p1AfU", LibcInternal::freopen_hle);
	LIB_FUNC("qdlHjTa9hQ4", LibcInternal::fdopen_hle);
	LIB_FUNC("uodLYyUip20", LibcInternal::fclose_hle);
	LIB_FUNC("lbB+UlZqVG0", LibcInternal::fread_hle);
	LIB_FUNC("MpxhMh8QFro", LibcInternal::fwrite_hle);
	LIB_FUNC("rQFVBXp-Cxg", LibcInternal::fseek_hle);
	LIB_FUNC("Qazy8LmXTvw", LibcInternal::ftell_hle);
	LIB_FUNC("pkYiKw09PRA", LibcInternal::fseeko_hle);
	LIB_FUNC("5qP1iVQkdck", LibcInternal::ftello_hle);
	LIB_FUNC("Fm-dmyywH9Q", LibcInternal::fileno_hle);
	LIB_FUNC("LxcEU+ICu8U", LibcInternal::feof_hle);
	LIB_FUNC("AHxyhN96dy4", LibcInternal::ferror_hle);
	LIB_FUNC("St9nbxSoezk", LibcInternal::clearerr_hle);
	LIB_FUNC("QMFyLoqNxIg", LibcInternal::setvbuf_hle);
	LIB_FUNC("vZMcAfsA31I", LibcInternal::setbuf_hle);
	LIB_FUNC("-LFO7jhD5CE", LibcInternal::ungetc_hle);
	LIB_FUNC("QrZZdJ8XsX0", LibcInternal::fputs_hle);
	LIB_FUNC("aZK8lNei-Qw", LibcInternal::fputc_hle);
	LIB_FUNC("AEuF3F2f8TA", LibcInternal::fgetc_hle);
	LIB_FUNC("KdP-nULpuGw", LibcInternal::fgets_hle);
	LIB_FUNC("3QIPIh-GDjw", LibcInternal::rewind_hle);
	LIB_FUNC("MZO7FXyAPU8", LibcInternal::remove_hle);
	LIB_FUNC("fffwELXNVFA", LibcInternal::fprintf_hle);
	LIB_FUNC("pDBDcY6uLSA", LibcInternal::vfprintf_hle);
	LIB_FUNC("MUjC4lbHrK4", LibcInternal::fflush);
	LIB_OBJECT("bgAcsbcEznc", &LibcInternal::g_stdin_p);
	LIB_OBJECT("zqJhBxAKfsc", &LibcInternal::g_stdout_p);
	LIB_OBJECT("as8Od-tH1BI", &LibcInternal::g_stderr_p);
	LIB_FUNC("bzQExy189ZI", LibC::init_env);
	LIB_FUNC("8G2LB+A3rzg", LibC::atexit);
	LIB_FUNC("hcuQgD53UxM", LibC::libc_printf);
	LIB_FUNC("YQ0navp+YIc", LibC::puts);
	LIB_FUNC("M4YYbSFfJ8g", LibC::setenv);
	LIB_FUNC("smbQukfxYJM", LibC::getenv);
	LIB_FUNC("CRJcH8CnPSI", LibC::unsetenv);
	LIB_FUNC("g7zzzLDYGw0", LibcInternal::strdup_hle); // strdup
	LIB_FUNC("wLlFkwG9UcQ", LibC::libc_time);
	LIB_FUNC("-VVn74ZyhEs", LibC::libc_difftime);
	LIB_FUNC("1mecP7RgI2A", LibC::libc_gmtime);
	LIB_FUNC("efhK-YSUYYQ", LibC::libc_localtime);
	LIB_FUNC("n7AepwR0s34", LibC::libc_mktime);
	LIB_FUNC("Av3zjWi64Kw", LibC::libc_strftime);
	LIB_FUNC("XKRegsFpEpk", LibC::catchReturnFromMain);
	LIB_FUNC("tsvEmnenz48", LibC::cxa_atexit);
	LIB_FUNC("H2e8t5ScQGc", LibC::cxa_finalize);
	LIB_FUNC("DiGVep5yB5w", LibC::std_execute_once);
}

} // namespace LibC

} // namespace Libs
