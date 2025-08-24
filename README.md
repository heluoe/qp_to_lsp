# From QP to LSP

## 目录结构

bazel 简单bazel 脚本
qp_solver 包含实现和测试用例
third_party 包含 Eigen, osqp, osqp-eigen 第三方依赖
papers 参考论文pdf

## 环境准备

按照文档配置好 [Bazel 工具链](https://bazel.build/configure/windows?hl=zh-cn)  (Bazel 版本bazel 8.4.0rc1, 含 MSVC BuildTools 构建编译器)

## 代码运行

[qp_solver](https://zhuanlan.zhihu.com/p/1938906212096775921)

```bazel
bazel test --test_output=all //qp_solver:all
```
