load("@//third_party/eigen:eigen.bzl", "eigen")
load("@//third_party/osqp:osqp.bzl", "osqp", "osqp_eigen")

def _third_party_deps_impl(_ctx):
    eigen()
    osqp()
    osqp_eigen()

third_party_deps = module_extension(
    implementation = _third_party_deps_impl,
)
