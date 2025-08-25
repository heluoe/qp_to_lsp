cc_library(
    name = "regularized_least_squares_solver",
    hdrs = [
            "regularized_least_squares_solver.h", 
            "defintions.h",
        ],
    srcs = ["regularized_least_squares_solver.cpp"],
    deps = [
        "@eigen",
    ],
)


cc_test(
    name = "unit_test",
    timeout = "long",
    srcs = [
        "test/solver_fixture.hpp", 
        "test/unit_test.cpp",
    ],
    deps = [
        "//lsp:regularized_least_squares_solver",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "benchmark_test",
    srcs = [
        "test/benchmark_test.cpp",
    ],
    tags = [
        "benchmark",
    ],
    deps = [
        "//lsp:regularized_least_squares_solver",
        "@googletest//:gtest",
        "@google_benchmark//:benchmark",
    ],
)
