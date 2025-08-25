#ifndef LSP_REGULARIZED_LEAST_SQUARES_SOLVER_H
#define LSP_REGULARIZED_LEAST_SQUARES_SOLVER_H

#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 320000000  // 320MB
#endif

// #define EIGEN_NO_MALLOC
// #define EIGEN_DONT_ALIGN
#define EIGEN_NO_DEBUG
#define EIGEN_VECTORIZE_SSE4_2

// #ifndef EIGEN_MPL2_ONLY
// #error According to Mozilla Public License 2.0 (MPL-2.0) is the only allowed license for Eigen library
// #else
#include <Eigen/Dense>
// #endif
#include <array>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "lsp\defintions.h"

using DefMatrixQ_mu = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

using DefMatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

using DefMatrixA = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

using DefVectorb = Eigen::VectorXf;

using DefVectorx = Eigen::VectorXf;

///
/// 提供了正则化最小二乘问题的解决方案：
/// min{ || A x - b || ^ 2 + mu || Lx || ^ 2 }
/// 对于此算法，必须满足L为对角矩阵且可逆的条件。
/// 并且对于L的所有对角线元素值都必须非零。
///
class RegularizedLeastSquaresSolver
{
  public:
    ///
    /// @brief 初始化构造
    ///
    RegularizedLeastSquaresSolver(const DefMatrixA& matrixA, const DefVectorb& vectorB)
        : matrixA_(matrixA), vectorB_(vectorB), mu_(0.0F), sqrt_of_mu_(0.0F)
    {
        if (vectorB.size() == matrixA.rows())
        {

            for (auto i = 0; i < matrixA.cols(); i++)
            {
                L_values_inv_[i] = 1.0F;
            }
            std::ignore = InitializeSolver(matrixA.rows(), matrixA.cols());
        }
        else
        {
            initialization_done_ = false;
        }
    }

    ///
    /// @brief 针对给定规模的系统初始化最小二乘求解器
    ///
    /// @param number_of_rows 矩阵A的行数
    /// @param number_of_cols 矩阵A的列数
    /// @return 初始化成功返回 true，若矩阵尺寸过大则返回 false
    ///
    bool InitializeSolver(const std::int32_t number_of_rows, const std::int32_t number_of_cols);
    bool GetInitializationStatus() { return initialization_done_; }
    ///
    /// 设置mu参数值。若未设置mu，将使用默认值0.0。
    /// mu必须>=0，否则将使用0.0并返回false。
    ///
    /// @param[in] mu 正则化参数
    /// @return 设置成功返回true，参数无效返回false
    ///
    bool SetMu(const float mu);

    ///
    /// @brief 返回当前 mu值.
    /// @return mu
    ///
    float GetMu() const { return mu_; }
    ///
    /// 设置矩阵 L 的对角线元素值。
    /// 若求解器尚未初始化，返回 false。
    /// 若索引值超出矩阵 L 的维度范围，返回 false。
    /// 若设置的值小于 FLT_EPSILON，返回 false。
    /// @param[in] i 需要设置值的位置索引（在矩阵 L 中的行&列位置）
    /// @param[in] value 要设置在矩阵 L 中 (i, i) 位置的值
    /// @return 函数执行结果（成功返回 true，否则返回 false）
    ///
    bool SetDiagonalValueOfL(const float value);
    bool SetDiagonalValueOfL(const std::uint32_t i, const float value);

    ///
    /// 计算正则化最小二乘问题的解。
    /// @return 如果成功计算出结果则返回 true；否则返回 false，例如
    /// 当矩阵不满秩或该类尚未初始化时。
    ///
    DefVectorx solve();

    ///
    /// 计算正则化最小二乘问题的解
    /// @param[in] random_data 设置为true时表示使用随机数据(GTest 测试用)
    /// @return 如果成功计算出结果则返回true；否则返回false，例如
    /// 当矩阵出现不满秩情况时
    ///
    bool FillRandomData();

  protected:
    /// @brief  matrix A Eigen 格式存储
    ///
    DefMatrixA matrixA_;
    /// @brief vector b Eigen 格式存储
    ///
    DefVectorb vectorB_;
    /// @brief lsp 求解结果 x Eigen 格式存储并返回
    ///
    DefVectorx vectorX_;

    std::int32_t number_of_rows_;
    std::int32_t number_of_cols_;
    std::array<float, kMaximumNumberOfCols> L_values_inv_;

    float mu_;
    float sqrt_of_mu_;

    bool calculation_done_;
    bool initialization_done_;

  private:
    /// 用于计算中间矩阵 B_mu 和 Q_mu 的辅助变量
    ///
    DefMatrixR matrixLinv_;
    DefMatrixR matrixR_;
    DefMatrixQ_mu B_mu_;
    DefMatrixQ_mu Q_mu_;
    DefVectorb b_derived_;

  private:
    /// 用于计算中间矩阵 B_mu 和 Q_mu 的辅助函数
    ///
    void CalculateB_mu(DefMatrixQ_mu& Q_mu, DefMatrixQ_mu& B_mu);
    static void MultQ_Mu(DefMatrixQ_mu& Q_mu, const std::int32_t row, const std::int32_t col, const float factor);
    static void ExecuteGivenRotation(DefMatrixQ_mu& Q_mu,
                                     const std::int32_t lower_index,
                                     const std::int32_t upper_index,
                                     const float top_left_value,
                                     const float top_right_value,
                                     const float bottom_left_value,
                                     const float bottom_right_value);
    /// 通过回代法求解 B_mu * x = b 的问题。矩阵 B_mu 已是三角矩阵。
    ///
    bool ExecuteBacksubstitution(const DefMatrixQ_mu& B_mu, const DefVectorb& b, DefVectorx& x) const;
};

#endif  // LSP_REGULARIZED_LEAST_SQUARES_SOLVER_H
