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

#define LOGGING_DEFINED

#include <cstdio>
#include <ctime>
#include <csignal>

#include "common/Common.h"

#define error(fmt, ...) fprintf(stderr, "\033[31m[" __FILE_NAME__ ":" MSTR(__LINE__)"]\033[0m " fmt "!\n" __VA_OPT__(,) __VA_ARGS__ )
#define warn(fmt, ...) fprintf(stdout, "\033[93m[" __FILE_NAME__ ":" MSTR(__LINE__)"]\033[0m " fmt "!\n" __VA_OPT__(,) __VA_ARGS__ )
#define info(fmt, ...) fprintf(stdout, "\033[32m[info]\033[0m " fmt ".\n" __VA_OPT__(,) __VA_ARGS__ )
#define prompt(fmt, ...) fprintf(stdout, "\033[96m" fmt "\033[0m" __VA_OPT__(,) __VA_ARGS__)

#define stl_error(code, fmt, ...) error(fmt ", cause %d: %s" __VA_OPT__(,) __VA_ARGS__, code, strerror(code))
#define stl_warn(code, fmt, ...) warn(fmt ", cause %d: %s" __VA_OPT__(,) __VA_ARGS__, code, strerror(code))

#ifdef DEBUG
#define debug(fmt, ...) fprintf(stdout, "\033[95m[" __FILE_NAME__ ":" MSTR(__LINE__)"]\033[0m " fmt "!\n" __VA_OPT__(,) __VA_ARGS__ )
#else
#define debug(fmt, ...)
#endif
