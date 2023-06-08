#pragma once

#ifdef _WIN32
#if defined(ST_EXPORTS)
#define ST_API __declspec(dllexport)
#else
#define ST_API __declspec(dllimport)
#endif
#else
#define ST_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define ST_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define ST_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define ST_INLINE_VARIABLE __attribute__((weak))
#endif
#endif