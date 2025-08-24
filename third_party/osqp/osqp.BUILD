load("@aspect_bazel_lib//lib:expand_template.bzl", "expand_template")

package(default_visibility = ["//visibility:public"])

expand_template(
    name = "qdldl_types_h",
    out = "include/qdldl_types.h",
    substitutions = {
        "@QDLDL_INT_TYPE@": "long long",
        "@QDLDL_FLOAT_TYPE@": "double",
        "@QDLDL_BOOL_TYPE@": "unsigned char",
        "@QDLDL_INT_TYPE_MAX@": "LLONG_MAX",
    },
    template = "lin_sys/direct/qdldl/qdldl_sources/configure/qdldl_types.h.in",
)

expand_template(
    name = "osqp_configure_h",
    out = "include/osqp_configure.h",
    substitutions = {
        "#cmakedefine DEBUG": "/* #undef DEBUG */",
        "#cmakedefine IS_LINUX": "/* #undef IS_LINUX */",
        "#cmakedefine IS_MAC": "/* #undef IS_MAC */",
        "#cmakedefine IS_WINDOWS": "#define IS_WINDOWS",
        "#cmakedefine EMBEDDED (@EMBEDDED@)": "/* #undef EMBEDDED */",
        "#cmakedefine PRINTING": "/* #undef PRINTING */",
        "#cmakedefine PROFILING": "#define PROFILING",
        "#cmakedefine CTRLC": "#define CTRLC",
        "#cmakedefine DFLOAT": "/* #undef DFLOAT */",
        "#cmakedefine DLONG": "#define DLONG",
        "#cmakedefine ENABLE_MKL_PARDISO": "/* #undef ENABLE_MKL_PARDISO */",
        "#cmakedefine OSQP_CUSTOM_MEMORY": "/* #undef OSQP_CUSTOM_MEMORY */",
        "@OSQP_CUSTOM_MEMORY@": "",
    },
    template = "configure/osqp_configure.h.in",
)

cc_library(
    name = "osqp_local",
    srcs = glob([
        "src/*.c",
        "lin_sys/direct/pardiso/*.c",
        "lin_sys/direct/qdldl/amd/**/*.c",
        "lin_sys/direct/qdldl/qdldl_sources/src/*.c",
        "lin_sys/direct/qdldl/*.c",
        "lin_sys/*.c",
    ]),
    hdrs = glob([
        "lin_sys/*.h",
        "lin_sys/direct/pardiso/*.h",
        "lin_sys/direct/qdldl/amd/**/*.h",
        "lin_sys/direct/qdldl/*.h",
        "lin_sys/direct/qdldl/qdldl_sources/include/*.h",
        "include/*.h",
    ]) + [
        "include/osqp_configure.h",
        "include/qdldl_types.h",
    ],
    copts = [
        "-ldl",
    ],
    defines = select({
        "//conditions:default": [],
    }),
    features = [
        "treat_warnings_as_errors",
        "strict_clang_tidy_warnings",
        "strict_warnings",
    ],
    includes = [
        "include",
        "lin_sys",
        "lin_sys/direct/pardiso",
        "lin_sys/direct/qdldl",
        "lin_sys/direct/qdldl/amd/include",
        "lin_sys/direct/qdldl/qdldl_sources/include",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "osqp_demo",
    srcs = ["examples/osqp_demo.c"],
    deps = [
        ":osqp_local",
    ],
)

cc_binary(
    name = "qdldl_demo",
    srcs = ["lin_sys/direct/qdldl/qdldl_sources/examples/example.c"],
    deps = [
        ":osqp_local",
    ],
)

cc_binary(
    name = "osqp_test",
    srcs = glob([
        "tests/**/*.cpp",
        "tests/**/*.h",
        "tests/**/*.hpp",
    ]),
    includes = ["tests"],
    deps = [
        ":osqp_local",
    ],
)
