/**
 * @file logging.h
 * @author rileyhorrix (riley@horrix.com)
 * @brief Definition of logging macros.
 * @version 0.1
 * @date 2025-09-24
 *
 * Copyright (c) Riley Horrix 2025
 */
#pragma once

#include <cstdio>
#include <ctime>
#include <csignal>

#include "common/Common.h"

namespace Dae {

/**
 * @brief Print an error message to stderr.
 *
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define error(fmt, ...)                                                     \
    fprintf(stderr, "\033[31m[" __FILE_NAME__                               \
                    ":" MSTR(__LINE__) "]\033[0m " fmt "!\n" __VA_OPT__(, ) \
                        __VA_ARGS__)

/**
 * @brief Print a warning message to stdout.
 *
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define warn(fmt, ...)                                                      \
    fprintf(stdout, "\033[93m[" __FILE_NAME__                               \
                    ":" MSTR(__LINE__) "]\033[0m " fmt "!\n" __VA_OPT__(, ) \
                        __VA_ARGS__)

/**
 * @brief Print an info message to stdout.
 *
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define info(fmt, ...) \
    fprintf(stdout,    \
            "\033[32m[info]\033[0m " fmt ".\n" __VA_OPT__(, ) __VA_ARGS__)

/**
 * @brief Print a test info message to stdout.
 *
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define testInfo(test, fmt, ...) \
    fprintf(stdout,    \
            "\033[38;2;255;165;0m%s : \033[0m" fmt ".\n", test  __VA_OPT__(, ) __VA_ARGS__)

/**
 * @brief Prompt the user for command line input. Should be followed with a
 * call to read stdin.
 *
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define prompt(fmt, ...) \
    fprintf(stdout, "\033[96m" fmt "\033[0m" __VA_OPT__(, ) __VA_ARGS__)

/**
 * @brief Print an error message when a standard library function fails.
 *
 * @param code Error code, should be `errno` for most standard library calls.
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define stl_error(code, fmt, ...) \
    error(fmt ", cause %d: %s" __VA_OPT__(, ) __VA_ARGS__, code, strerror(code))

/**
 * @brief Print a warning message when a standard library function fails.
 *
 * @param code Error code, should be `errno` for most standard library calls.
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define stl_warn(code, fmt, ...) \
    warn(fmt ", cause %d: %s" __VA_OPT__(, ) __VA_ARGS__, code, strerror(code))

#ifdef DEBUG
/**
 * @brief Print a debug message. This is only defined when the program is built
 * in debug mode, with `DEBUG` defined.
 *
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define debug(fmt, ...)                                                     \
    fprintf(stdout, "\033[95m[" __FILE_NAME__                               \
                    ":" MSTR(__LINE__) "]\033[0m " fmt "!\n" __VA_OPT__(, ) \
                        __VA_ARGS__)
#else
/**
 * @brief Print a debug message. This is only defined when the program is built
 * in debug mode, with `DEBUG` defined.
 *
 * @param fmt Format string.
 * @param ... Variadic arguments.
 * @return fprintf return code.
 */
#define debug(fmt, ...)
#endif

} // namespace Dae