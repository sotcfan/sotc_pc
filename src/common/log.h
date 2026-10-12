#pragma once

#include <format>
#include <string_view>

namespace logging {

enum class Level : int { Trace, Debug, Info, Warning, Error, Critical };

void SetLevel(Level level);
bool Enabled(Level level);
bool ParseLevel(std::string_view text, Level& out);

// Name shown in every line written by the calling thread.
void SetThreadName(std::string_view name);
const char* CurrentThreadName();

void Write(Level level, std::string_view category, std::string_view message);

} // namespace logging

#define LOG_AT(level, category, ...)                                                               \
    do {                                                                                           \
        if (::logging::Enabled(level)) {                                                           \
            ::logging::Write(level, category, std::format(__VA_ARGS__));                           \
        }                                                                                          \
    } while (0)

#define LOG_TRACE(category, ...) LOG_AT(::logging::Level::Trace, category, __VA_ARGS__)
#define LOG_DEBUG(category, ...) LOG_AT(::logging::Level::Debug, category, __VA_ARGS__)
#define LOG_INFO(category, ...) LOG_AT(::logging::Level::Info, category, __VA_ARGS__)
#define LOG_WARNING(category, ...) LOG_AT(::logging::Level::Warning, category, __VA_ARGS__)
#define LOG_ERROR(category, ...) LOG_AT(::logging::Level::Error, category, __VA_ARGS__)
#define LOG_CRITICAL(category, ...) LOG_AT(::logging::Level::Critical, category, __VA_ARGS__)
