load("@//bazel/rules/licenses:licenses.bzl", "licenses")

licenses(["Apache License 2.0 (Apache-2.0)"])

alias(
    name = "osqp",
    actual = select(
        {
            "//conditions:default": "@osqp//:osqp_local",
        },
        no_match_error = "No known platform specified.",
    ),
    visibility = ["//visibility:public"],
)

alias(
    name = "osqp_eigen",
    actual = select(
        {
            "//conditions:default": "@osqp_eigen",
        },
        no_match_error = "No known platform specified.",
    ),
    visibility = ["//visibility:public"],
)

cc_test(
    name = "osqp_eigen_test",
    srcs = glob(["test/*.cpp"]),
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@osqp_eigen",
    ],
)
