load("@//bazel/rules:artifactory.bzl", "artifactory_archive")

# TYPE: upstream
# LICENSE: MPL-2
# CONTACT:
def eigen():
    if "eigen" not in native.existing_rules():
        artifactory_archive(
            name = "eigen",
            repo = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/",
            path = "eigen-3.3.7.tar.bz2",
            sha256 = "685adf14bd8e9c015b78097c1dc22f2f01343756f196acdc76a678e1ae352e11",
            build_file = "//third_party/eigen:eigen.BUILD",
            strip_prefix = "eigen-3.3.7",
        )
