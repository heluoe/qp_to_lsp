#ifndef LSP_TEST_SOLVER_FIXTURE_HPP
#define LSP_TEST_SOLVER_FIXTURE_HPP

#include <random>

#include <gtest/gtest.h>

#include "lsp\regularized_least_squares_solver.h"

class RegularizedLeastSquaresSolverFixture : public ::testing::TestWithParam<std::tuple<std::int32_t, std::int32_t>>
{
  public:
    void SetUp() {}
    void CreateScenario1()
    {  /// Test data, which has been derived/used in Matlab.
        g_matrixA.resize(46, 5);
        g_vectorB.resize(5);
    }

    void CreateRandomDataScenario(const std::int32_t number_of_rows, const std::int32_t number_of_cols)
    {
        g_matrixA.resize(number_of_rows, number_of_cols);
        g_vectorB.resize(number_of_rows, 1);

        const float low = -100.0f;
        const float high = 100.0f;
        g_matrixA = DefMatrixA::NullaryExpr(number_of_rows, number_of_cols, [&]() {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dis(low, high);
            return dis(gen);
        });

        g_vectorB = DefVectorb::NullaryExpr(number_of_rows, [&]() {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dis(low, high);
            return dis(gen);
        });
    }

  protected:
    /// 测试Fixture 用例
    DefMatrixA g_matrixA;
    DefVectorb g_vectorB;
};

#endif  // LSP_TEST_SOLVER_FIXTURE_HPP
