load("@//bazel/rules:artifactory.bzl", "artifactory_archive")

def osqp():
    artifactory_archive(
        name = "osqp",
        build_file = "//third_party/osqp:osqp.BUILD",
        path = "osqp/releases/download/v0.6.3/osqp-v0.6.3-src.tar.gz",
        repo = "https://github.com/osqp/",
        sha256 = "285b2a60f68d113a1090767ec8a9c81a65b3af2d258f8c78a31cc3f98ba58456",
    )

def osqp_eigen():
    artifactory_archive(
        name = "osqp_eigen",
        build_file = "//third_party/osqp:osqp_eigen.BUILD",
        path = "osqp-eigen/archive/refs/tags/v0.8.1.tar.gz",
        repo = "https://github.com/robotology/",
        sha256 = "21f04878bed68cb433c4341570ee2e5755f7d499d8ab550e7dc2308569dabf71",
        strip_prefix = "osqp-eigen-0.8.1",
    )
