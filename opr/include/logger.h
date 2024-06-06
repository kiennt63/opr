#pragma once

#include <cstdlib>
#include <memory>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/spdlog.h"

namespace common {

class Logger
{
public:
    static void Init();
    inline static std::shared_ptr<spdlog::logger> GetLogger() { return logger_; }

private:
    static std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace common

#define log_inf(...) SPDLOG_INFO(__VA_ARGS__)
#define log_dbg(...) SPDLOG_DEBUG(__VA_ARGS__)
#define log_wrn(...) SPDLOG_WARN(__VA_ARGS__)

#define log_err(...)               \
    do                             \
    {                              \
        SPDLOG_ERROR(__VA_ARGS__); \
        std::exit(EXIT_FAILURE);   \
    } while (false)

#define check_inf(value, ...)         \
    do                                \
    {                                 \
        if (!(value))                 \
        {                             \
            SPDLOG_INFO(__VA_ARGS__); \
        }                             \
    } while (false)

#define check_wrn(value, ...)         \
    do                                \
    {                                 \
        if (!(value))                 \
        {                             \
            SPDLOG_WARN(__VA_ARGS__); \
        }                             \
    } while (false)

#define check_err(value, ...)          \
    do                                 \
    {                                  \
        if (!(value))                  \
        {                              \
            SPDLOG_ERROR(__VA_ARGS__); \
            exit(EXIT_FAILURE);        \
        }                              \
    } while (false)
