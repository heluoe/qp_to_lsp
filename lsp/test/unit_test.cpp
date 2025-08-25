

#include <chrono>
#include <tuple>

#include <gtest/gtest.h>

#include "lsp\test\solver_fixture.hpp"

namespace
{
using namespace Eigen;
void expect_ret(const VectorXf& x, const VectorXf& y, float tol = 1e-5)
{
    ASSERT_EQ(x.size(), y.size());
    for (int i = 0; i < x.size(); ++i)
    {
        EXPECT_NEAR(x(i), y(i), tol) << "index " << i;
    }
}

// 使用 SVD 求解线性方程组 Ax = b
VectorXf solveWithSVD(const DefMatrixA& A, const VectorXf& b)
{
    // 计算 SVD 分解
    JacobiSVD<DefMatrixA> svd(A, ComputeThinU | ComputeThinV);

    // 使用 SVD 求解
    return svd.solve(b);
}

// 带截断的 SVD 求解（用于处理病态系统）
VectorXf solveWithTruncatedSVD(const DefMatrixA& A, const VectorXf& b, float threshold = 1e-6f)
{
    // 计算 SVD 分解
    BDCSVD<DefMatrixA> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    svd.setThreshold(threshold);  // 丢弃小于最大奇异值 * 1e-6 的值

    // 使用 SVD 求解
    return svd.solve(b);
}

// 使用 ColPivHouseholderQR  求解线性方程组 Ax = b
VectorXf solveWithColPivHouseholderQR(const DefMatrixA& A, const VectorXf& b)
{
    ColPivHouseholderQR<DefMatrixA> qr(A);

    return qr.solve(b);
}

}  // namespace

///
/// @test   Tests the CalculateXRandomizeData() with invalid size.
///

TEST_P(RegularizedLeastSquaresSolverFixture, CalculateXRandomizeData)
{
    std::int32_t kNumRows;
    std::int32_t kNumCols;
    // 获取参数
    std::tie(kNumRows, kNumCols) = GetParam();
    CreateRandomDataScenario(kNumRows, kNumCols);

    RegularizedLeastSquaresSolver lse_solver(g_matrixA, g_vectorB);
    EXPECT_TRUE(lse_solver.GetInitializationStatus());

    EXPECT_EQ(lse_solver.SetMu(0.01F * kNumRows), true);
    EXPECT_EQ(lse_solver.SetDiagonalValueOfL(0.01F), true);

    // 方法1: 直接使用 SVD solve
    //
    auto start = std::chrono::high_resolution_clock::now();
    VectorXf x1 = solveWithSVD(g_matrixA, g_vectorB);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Dircet svd solver cost: " << duration1.count() << " ms" << std::endl;
    // std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "errors ||Ax1 - b||: " << (g_matrixA * x1 - g_vectorB).norm() << std::endl;

    // 方法2: 使用截断 SVD
    //
    start = std::chrono::high_resolution_clock::now();
    VectorXf x2 = solveWithTruncatedSVD(g_matrixA, g_vectorB, 1e-6f);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nTruncatedSVD solver cost: " << duration2.count() << " ms" << std::endl;
    // std::cout << " x2:\n" << x2 << std::endl;
    std::cout << "errors ||Ax2 - b||: " << (g_matrixA * x2 - g_vectorB).norm() << std::endl;

    // 方法3: ColPivHouseholderQR
    //
    start = std::chrono::high_resolution_clock::now();
    VectorXf x3 = solveWithColPivHouseholderQR(g_matrixA, g_vectorB);
    end = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nQR solver cost: " << duration3.count() << " ms" << std::endl;
    // std::cout << " x3:\n" << x3 << std::endl;
    std::cout << "errors ||Ax3 - b||: " << (g_matrixA * x3 - g_vectorB).norm() << std::endl;

    // 方法4: Elden 算对角线变换求解
    //
    start = std::chrono::high_resolution_clock::now();
    auto x4 = lse_solver.solve();
    end = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nElden solver cost: " << duration4.count() << " ms" << std::endl;
    // std::cout << "x4:\n" << x1 << std::endl;
    std::cout << "errors ||A x4 - b||: " << (g_matrixA * x4 - g_vectorB).norm() << std::endl;

    expect_ret(x3, x4);
}

INSTANTIATE_TEST_SUITE_P(RegularizedLeastSquaresSolverTests,
                         RegularizedLeastSquaresSolverFixture,
                         ::testing::Values(std::make_tuple(100, 10),     // 小矩阵
                                           std::make_tuple(1000, 100),   // 中等矩阵
                                           std::make_tuple(10000, 1000)  // 大矩阵
                                                                         // 可以添加更多测试用例
                                           ));