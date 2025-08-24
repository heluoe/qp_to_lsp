package(default_visibility = ["//visibility:public"])

cc_library(
    name = "osqp_eigen",
    srcs = glob([
        "src/*.cpp",
    ]),
    hdrs = glob([
        "include/OsqpEigen/*.h",
        "include/OsqpEigen/*.hpp",
        "include/OsqpEigen/*.tpp",
    ]),
    defines = select({
        "//conditions:default": [
            "OSQP_EIGEN_DEBUG_OUTPUT",
            "EIGEN_USE_NEW_STDVECTOR",
        ],
    }),
    features = [
        "treat_warnings_as_errors",
        "strict_clang_tidy_warnings",
        "strict_warnings",
    ],
    includes = [
        "include",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@osqp//:osqp_local",
    ],
)

cc_binary(
    name = "osqp_eigen_demo",
    srcs = ["example/src/MPCExample.cpp"],
    defines = ["_USE_MATH_DEFINES"],
    deps = [
        ":osqp_eigen",
    ],
)
