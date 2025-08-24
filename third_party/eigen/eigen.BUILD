package(default_visibility = ["//visibility:public"])

# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.

licenses([
    # Note: Eigen is an MPL2 library that includes GPL v3 and LGPL v2.1+ code.
    #       We've taken special care to not reference any restricted code.
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["COPYING.MPL2"])

EIGEN_FILES = [
    "Eigen/**",
    "unsupported/**",
]

EIGEN_EXCLUDE_FILES = [
    "Eigen/*Support",
]

# Files known to be under MPL2 license.
EIGEN_MPL2_HEADER_FILES = glob(
    EIGEN_FILES,
    exclude = EIGEN_EXCLUDE_FILES + [
        # Guarantees any file missed by excludes above will not compile.
        "Eigen/src/Core/util/NonMPL2.h",
        "Eigen/**/CMakeLists.txt",
    ],
)

cc_library(
    name = "eigen",
    hdrs = EIGEN_MPL2_HEADER_FILES,
    copts = [
        "/wd4127",
        "/wd4819",
        "/wd4244",
        "/wd4305",
        "/wd4267",
        "/wd4522",
        "-DEIGEN_NO_DEBUG",  # 禁用 Eigen 的调试断言
    ],
    defines = [
        "EIGEN_MPL2_ONLY",
    ],
    includes = ["."],
)
