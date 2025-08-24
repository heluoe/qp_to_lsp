cc_library(
    name = "projected_dual_qp_solver",
    hdrs = ["projected_dual_qp_solver.hpp"],
)

cc_test(
    name = "test_qp_solver",
    timeout = "short",
    srcs = glob(["test/*.cpp"]),
    deps = [
        "//qp_solver:projected_dual_qp_solver",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@osqp_eigen",
    ],
)
