# From QP to LSP

## 目录结构

bazel 简单bazel 脚本
qp_solver 包含实现和测试用例
lsp 包含实现，测试用例和benchmark测试
third_party 包含 Eigen, osqp, osqp-eigen 第三方依赖
papers 参考论文pdf

## 环境准备

按照文档配置好 [Bazel 工具链](https://bazel.build/configure/windows?hl=zh-cn)  (Bazel 版本bazel 8.4.0rc1, 含 MSVC BuildTools 构建编译器)

## 代码运行

[qp_solver](https://zhuanlan.zhihu.com/p/1938906212096775921)

```bazel
bazel test --test_output=all //qp_solver:all
```

[lsp]()


```bazel
bazel test -c opt --test_output=all --nocache_test_results //lsp:unit_test
```

```bazel
bazel run -c opt --test_output=all --nocache_test_results //lsp:benchmark_test
```

benchmark 在i9 gen12 12900Hk 下输出：
```
Run on (20 X 2918 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x10)
  L1 Instruction 32 KiB (x10)
  L2 Unified 1280 KiB (x10)
  L3 Unified 24576 KiB (x1)
---------------------------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------------
BM_JacobiSVD/100/10                     0.037 ms        0.035 ms        19478 Residual=538.312
BM_JacobiSVD/1000/100                    26.0 ms         26.0 ms           30 Residual=1.73872k
BM_BDCSVD/100/10                        0.037 ms        0.035 ms        18667 Residual=556.81
BM_BDCSVD/1000/100                       5.01 ms         4.82 ms          149 Residual=1.68754k
BM_QR/100/10                            0.009 ms        0.009 ms        74667 Residual=532.59
BM_QR/1000/100                           1.53 ms         1.54 ms          498 Residual=1.68179k
BM_QR/10000/1000/iterations:20           2020 ms         1970 ms           20 Residual=5.45027k
BM_RLSP/100/10/100                      0.016 ms        0.016 ms        44800 Residual=575.205
BM_RLSP/1000/100/10                      1.97 ms         1.95 ms          345 Residual=1.73745k
BM_RLSP/10000/1000/1/iterations:20       2017 ms         1983 ms           20 Residual=5.45534k
```