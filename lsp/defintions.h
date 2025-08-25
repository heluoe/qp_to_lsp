#ifndef LSP_DEFINTIONS_H
#define LSP_DEFINTIONS_H

#include <cstdint>

/// @brief 最小二乘求解问题算法（LSP）支持的最大行数，可以按需调整内存和初始化时堆栈空间消耗
constexpr std::int32_t kMaxNumberOfRows = 10000;
/// @brief 最小二乘求解问题算法（LSP）支持的最大列数，可以按需调整内存和初始化时堆栈空间消耗
constexpr std::int32_t kMaximumNumberOfCols = 1000;

template <typename E>
constexpr auto to_underlying(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}

enum class EigenAlignOption : std::int32_t
{
    kEigenColMajor = 0,
    kEigenRowMajor = 0x1,
    kEigenAutoAlign = 0,
    kEigenDontAlign = 0x2
};

#endif  // LSP_DEFINTIONS_H
