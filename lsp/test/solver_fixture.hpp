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
        g_vectorB.resize(46);

        g_vectorB << -0.52415f, -0.51579f, -0.54646f, -0.53983f, -0.51231f, -0.50347f, -0.51721f, -0.54853f, -0.59551f,
            -0.60073f, -0.53504f, -0.54239f, -0.53166f, -0.45732f, -0.48999f, -0.45534f, -0.47186f, -0.42974f,
            -0.43866f, -0.35803f, -0.346f, -0.34062f, -0.33194f, 3.1569f, 3.1725f, 3.143f, 3.1672f, 3.1532f, 3.2121f,
            3.1933f, 3.2606f, 3.2439f, 3.2181f, 3.2037f, 3.1573f, 3.1235f, 3.1847f, 3.1221f, 3.1308f, 2.9978f, 2.9786f,
            2.8782f, 2.8803f, 2.833f, 2.7654f, 2.8014f;

        g_matrixA << 8.2144f, 67.477f, 554.28f, 1.0f, 0.0f, 10.631f, 113.01f, 1201.4f, 1.0f, 0.0f, 13.273f, 176.17f,
            2338.3f, 1.0f, 0.0f, 15.73f, 247.42f, 3891.9f, 1.0f, 0.0f, 18.348f, 336.65f, 6177.0f, 1.0f, 0.0f, 21.112f,
            445.72f, 9410.0f, 1.0f, 0.0f, 23.728f, 563.02f, 13359.0f, 1.0f, 0.0f, 26.2f, 686.47f, 17986.0f, 1.0f, 0.0f,
            28.833f, 831.33f, 23969.0f, 1.0f, 0.0f, 31.383f, 984.9f, 30909.0f, 1.0f, 0.0f, 33.865f, 1146.8f, 38837.0f,
            1.0f, 0.0f, 36.42f, 1326.4f, 48310.0f, 1.0f, 0.0f, 39.284f, 1543.2f, 60624.0f, 1.0f, 0.0f, 41.842f, 1750.7f,
            73255.0f, 1.0f, 0.0f, 44.385f, 1970.0f, 87441.0f, 1.0f, 0.0f, 46.969f, 2206.1f, 103620.0f, 1.0f, 0.0f,
            49.512f, 2451.5f, 121380.0f, 1.0f, 0.0f, 52.07f, 2711.3f, 141180.0f, 1.0f, 0.0f, 54.906f, 3014.6f,
            165520.0f, 1.0f, 0.0f, 57.458f, 3301.4f, 189690.0f, 1.0f, 0.0f, 60.008f, 3600.9f, 216080.0f, 1.0f, 0.0f,
            62.613f, 3920.4f, 245470.0f, 1.0f, 0.0f, 63.911f, 4084.6f, 261050.0f, 1.0f, 0.0f, 9.8241f, 96.513f, 948.15f,
            0.0f, 1.0f, 12.912f, 166.71f, 2152.5f, 0.0f, 1.0f, 15.745f, 247.9f, 3903.1f, 0.0f, 1.0f, 18.882f, 356.52f,
            6731.8f, 0.0f, 1.0f, 21.735f, 472.41f, 10268.0f, 0.0f, 1.0f, 25.151f, 632.56f, 15909.0f, 0.0f, 1.0f,
            27.963f, 781.95f, 21866.0f, 0.0f, 1.0f, 31.087f, 966.41f, 30043.0f, 0.0f, 1.0f, 34.035f, 1158.4f, 39425.0f,
            0.0f, 1.0f, 36.963f, 1366.2f, 50500.0f, 0.0f, 1.0f, 40.152f, 1612.2f, 64731.0f, 0.0f, 1.0f, 43.163f,
            1863.0f, 80414.0f, 0.0f, 1.0f, 46.426f, 2155.4f, 100070.0f, 0.0f, 1.0f, 49.438f, 2444.1f, 120830.0f, 0.0f,
            1.0f, 52.424f, 2748.2f, 144070.0f, 0.0f, 1.0f, 55.477f, 3077.7f, 170740.0f, 0.0f, 1.0f, 58.493f, 3421.4f,
            200130.0f, 0.0f, 1.0f, 61.511f, 3783.6f, 232740.0f, 0.0f, 1.0f, 64.836f, 4203.7f, 272550.0f, 0.0f, 1.0f,
            67.857f, 4604.6f, 312460.0f, 0.0f, 1.0f, 70.947f, 5033.4f, 357100.0f, 0.0f, 1.0f, 73.896f, 5460.6f,
            403510.0f, 0.0f, 1.0f, 75.444f, 5691.8f, 429410.0f, 0.0f, 1.0f;
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
