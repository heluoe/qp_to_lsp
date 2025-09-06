
#include <benchmark/benchmark.h>

#include "lsp\test\solver_fixture.hpp"

// 工具：生成随机矩阵/向量
//
struct TestCase
{
    DefMatrixA A;
    DefVectorx b;
    std::int32_t m, n;
};

static TestCase GenerateTestCase(std::int32_t m, std::int32_t n)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    DefMatrixA A(m, n);
    DefVectorx b(m);

    const float low = -100.0f;
    const float high = 100.0f;
    A = DefMatrixA::NullaryExpr(m, n, [&]() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(low, high);
        return dis(gen);
    });

    b = DefVectorb::NullaryExpr(m, [&]() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(low, high);
        return dis(gen);
    });

    return {A, b, m, n};
}

// ---------------- JacobiSVD ----------------
static void BM_JacobiSVD(benchmark::State& state)
{
    std::int32_t m = state.range(0);
    std::int32_t n = state.range(1);
    for (auto _ : state)
    {
        auto tc = GenerateTestCase(m, n);
        benchmark::DoNotOptimize(tc);
        Eigen::JacobiSVD<DefMatrixA> svd(tc.A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto x = svd.solve(tc.b);
        float residual = (tc.A * x - tc.b).norm();
        benchmark::DoNotOptimize(residual);
        state.counters["Residual"] = residual;
    }
}
BENCHMARK(BM_JacobiSVD)->Args({100, 10})->Args({1000, 100})->Unit(benchmark::kMillisecond);

// ---------------- BDCSVD ----------------
static void BM_BDCSVD(benchmark::State& state)
{
    std::int32_t m = state.range(0);
    std::int32_t n = state.range(1);

    for (auto _ : state)
    {
        auto tc = GenerateTestCase(m, n);
        benchmark::DoNotOptimize(tc);
        Eigen::BDCSVD<DefMatrixA> svd(tc.A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.setThreshold(1e-6f);
        auto x = svd.solve(tc.b);
        float residual = (tc.A * x - tc.b).norm();
        benchmark::DoNotOptimize(residual);
        state.counters["Residual"] = residual;
    }
}
BENCHMARK(BM_BDCSVD)->Args({100, 10})->Args({1000, 100})->Unit(benchmark::kMillisecond);

// ---------------- QR ----------------
static void BM_QR(benchmark::State& state)
{
    std::int32_t m = state.range(0);
    std::int32_t n = state.range(1);

    for (auto _ : state)
    {
        auto tc = GenerateTestCase(m, n);
        benchmark::DoNotOptimize(tc);
        Eigen::ColPivHouseholderQR<DefMatrixA> qr(tc.A);
        auto x = qr.solve(tc.b);
        float residual = (tc.A * x - tc.b).norm();
        benchmark::DoNotOptimize(residual);
        state.counters["Residual"] = residual;
    }
}
BENCHMARK(BM_QR)->Args({100, 10})->Args({1000, 100})->Args({1000, 300})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_QR)->Iterations(20)->Args({10000, 1000})->Unit(benchmark::kMillisecond);

// ---------------- Elden LSP  ----------------
static void BM_RLSP(benchmark::State& state)
{
    std::int32_t m = state.range(0);
    std::int32_t n = state.range(1);
    float L = state.range(2) / 100.0F;

    for (auto _ : state)
    {
        auto tc = GenerateTestCase(m, n);
        benchmark::DoNotOptimize(tc);
        RegularizedLeastSquaresSolver lse_solver(tc.A, tc.b);
        lse_solver.SetMu(0.001F * m);
        lse_solver.SetDiagonalValueOfL(L);

        auto x = lse_solver.solve();
        float residual = (tc.A * x - tc.b).norm();
        benchmark::DoNotOptimize(residual);
        state.counters["Residual"] = residual;
    }
}
BENCHMARK(BM_RLSP)->Args({100, 10, 100})->Args({1000, 100, 10})->Args({1000, 300, 10})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RLSP)->Iterations(20)->Args({10000, 1000, 1})->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();