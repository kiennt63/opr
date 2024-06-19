#include "logger.h"
#include "spdlog/sinks/stdout_color_sinks.h"

using namespace opr;

std::shared_ptr<spdlog::logger> logger::logger_;

void logger::init()
{
    logger_ = spdlog::stdout_color_mt("opr");
    logger_->set_level(spdlog::level::trace);
    spdlog::set_default_logger(logger_);
    spdlog::set_pattern("%^[%T.%e][%s:%#]%$ %v");
}