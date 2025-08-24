#include <OsqpEigen/OsqpEigen.h>
#include <gtest/gtest.h>

#include "qp_solver/projected_dual_qp_solver.hpp"

#include <Eigen/Dense>

namespace
{

VectorXd solveWithOSQPEigen(const Eigen::MatrixXd& G,      // 正定矩阵 (n x n)
                            const Eigen::VectorXd& a,      // 向量 (n)
                            const Eigen::MatrixXd& Ceq,    // 等式约束矩阵 (n x m_e)
                            const Eigen::VectorXd& Beq,    // 等式约束向量 (m_e)
                            const Eigen::MatrixXd& Cineq,  // 不等式约束矩阵 (n x m_ineq)
                            const Eigen::VectorXd& Bineq,  // 不等式约束向量 (m_ineq)
                            const Eigen::VectorXd& xl,     // 变量下界 (n)
                            const Eigen::VectorXd& xu,     // 变量上界 (n)
                            double eps_abs = 1e-9,         // 绝对精度
                            int max_iter = 10000)          // 最大迭代次数)
{
    int n = G.rows();       //变量维度
    int me = Ceq.cols();    // 等式约束数
    int mi = Cineq.cols();  // 不等式约束数
    // 转成标准 QP 形式：
    // min 0.5 x^T H x + g^T x
    // s.t. A_eq x + beq = 0, A_ineq x + b_ineq >=0
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);  // 关闭详细输出
    solver.settings()->setAbsoluteTolerance(eps_abs);
    solver.settings()->setMaxIteration(max_iter);

    // OSQP-Eigen 需要稀疏矩阵
    Eigen::SparseMatrix<double> H = G.sparseView();
    Eigen::VectorXd g(n);
    g.setZero();
    g = a;

    // 合并约束（A，lb，ub）
    int m_total = me + mi + n;  // 等式 + 不等式 + 变量bounds
    // A = [Ceq^T; Cineq^T; I] (m_total * n)
    Eigen::SparseMatrix<double> A_s;
    MatrixXd A(m_total, n);
    A.setZero();

    Eigen::VectorXd lb(m_total), ub(m_total);
    lb.setZero();
    ub.setZero();
    int row = 0;

    // 等式约束:  lb = ub = -Beq
    for (int i = 0; i < me; ++i, ++row)
    {
        A.row(row) = Ceq.col(i).transpose();
        ub(row) = -Beq(i);
        lb(row) = -Beq(i);
    }

    // 不等式: c^T x + b >= 0 => lb(-b) <=  c^T x
    for (int j = 0; j < mi; ++j, ++row)
    {
        A.row(row) = Cineq.col(j).transpose();
        ub(row) = OsqpEigen::INFTY;
        lb(row) = -Bineq(j);
    }
    // x bounds
    for (int i = 0; i < n; ++i, ++row)
    {
        // upper bound: x_i <= xu_i
        // lower bound: x_i >= xl_i
        A(row, i) = 1.0;
        ub(row) = xu(i);
        lb(row) = xl(i);
    }
    A_s = A.sparseView();

    solver.data()->setNumberOfVariables(n);
    solver.data()->setNumberOfConstraints(m_total);
    solver.data()->setHessianMatrix(H);
    solver.data()->setGradient(g);
    solver.data()->setLinearConstraintsMatrix(A_s);
    solver.data()->setLowerBound(lb);
    solver.data()->setUpperBound(ub);

    if (!solver.initSolver())
    {
        return VectorXd(n);
    }

    auto res_osqp = solver.solveProblem();

    // std::cout << "osqp status: " << static_cast<int>(res_osqp) << std::endl;

    return solver.getSolution();
}

void expect_ret(const VectorXd& x, const VectorXd& y, double tol = 1e-5)
{
    ASSERT_EQ(x.size(), y.size());
    for (int i = 0; i < x.size(); ++i)
    {
        EXPECT_NEAR(x(i), y(i), tol) << "index " << i;
    }
}

}

TEST(QPSolverTest, WithConstraintsCase1)
{
    int n = 2;
    // min f(x) = (x1 - 1)^2 + (x2 - 2.5)^2
    MatrixXd G(n, n);  // 2 * [1 0; 0 1]
    G << 2.0, 0.0, 0.0, 2.0;
    std::cout << "G: " << std::endl << G << std::endl;
    VectorXd a(n);  // -2x1, -5x2
    a << -2.0, -5.0;
    MatrixXd Ceq(n, 0);
    VectorXd Beq(0);
    //  x1 -2x2 + 2 >=0
    // -x1 -2x2 + 6 >=0
    // -x1 +2x2 + 2 >=0
    MatrixXd Cineq(n, 3);
    Cineq << 1, -1, -1, -2, -2, 2;
    std::cout << "Cineq^T: " << Cineq.transpose() << std::endl;
    VectorXd Bineq(3);
    Bineq << 2, 6, 2;
    // x1 >= 0; x2 >=0
    VectorXd xl(2);
    VectorXd xu(2);
    xl << 0, 0;
    xu << 1e6, 1e6;

    ProjectedDualActiveSetQP solver(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    auto res = solver.solve(1e-9, 200);

    EXPECT_TRUE(res.success);
    // Our solver should detect infeasible (or fail); either res.success==false OR osqp says infeasible
    auto osqp_res = solveWithOSQPEigen(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    std::cout << "qp x*: " << res.x.transpose() << ", iter: " << res.iterations << std::endl;
    std::cout << "osqp x*: " << osqp_res.transpose() << std::endl;
}

TEST(QPSolverTest, FiveStatesWithConstraintsCase2)
{
    int n = 5;
    // min f(x) = 0.5*(x1^2 + x2^2 + x3^2 + x4^2 + x5^2)
    //            -21.98x1 -1.26x2 +61.39x3 +5.3x4 +101.3x5
    MatrixXd G = Eigen::MatrixXd::Identity(n, n);
    std::cout << "G: " << std::endl << G << std::endl;
    VectorXd a(n);
    a << -21.98, -1.26, 61.39, 5.3, 101.3;
    MatrixXd Ceq(n, 0);
    VectorXd Beq(0);
    //  -7.56x1 +0.5x5 + 39.1 >=0
    MatrixXd Cineq(n, 1);
    Cineq << -7.56, 0, 0, 0, 0.5;
    std::cout << "Cineq^T: " << Cineq.transpose() << std::endl;
    VectorXd Bineq(1);
    Bineq << 39.1;
    // -100 <= xi <= 100
    VectorXd xl(n);
    VectorXd xu(n);
    xl << -100, -100, -100, -100, -100;
    xu << 100, 100, 100, 100, 100;

    ProjectedDualActiveSetQP solver(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    auto res = solver.solve(1e-9, 200);

    EXPECT_TRUE(res.success);
    // Our solver should detect infeasible (or fail); either res.success==false OR osqp says infeasible
    auto osqp_res = solveWithOSQPEigen(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    std::cout << "qp x*: " << res.x.transpose() << ", iter num: " << res.iterations << std::endl;
    std::cout << "op fun val: " << 0.5 * res.x.transpose() * G * res.x + a.transpose() * res.x << std::endl;
    std::cout << "osqp x*: " << osqp_res.transpose() << std::endl;
}

TEST(QPSolverTest, WithEqConstraintsCase3)
{
    int n = 2;
    MatrixXd G(n, n);
    G << 4.0, -2.0, -2.0, 4.0;
    std::cout << "G: " << std::endl << G << std::endl;
    VectorXd a(n);
    a << 6.0, 0.0;
    MatrixXd Ceq(n, 1);  // x1 + x2 -1 = 0
    VectorXd Beq(1);
    Ceq << 1.0, 1.0;
    Beq << -1;
    MatrixXd Cineq(n, 0);
    VectorXd Bineq(0);
    VectorXd xl(2);
    VectorXd xu(2);
    xl << 0, 0;
    xu << 0.7, 0.7;

    ProjectedDualActiveSetQP solver(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    auto res = solver.solve(1e-9, 200);

    EXPECT_TRUE(res.success);

    auto osqp_res = solveWithOSQPEigen(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    std::cout << "qp x*: " << res.x.transpose() << std::endl;
    std::cout << "osqp x*: " << osqp_res.transpose() << std::endl;
}

TEST(QPSolverTest, WithIneqConstraintsCase4)
{
    int n = 2;
    MatrixXd G(n, n);
    G << 4.0, -2.0, -2.0, 4.0;
    std::cout << "G: " << std::endl << G << std::endl;
    VectorXd a(n);
    a << 6.0, 0.0;
    MatrixXd Ceq(n, 0);
    VectorXd Beq(0);
    MatrixXd Cineq(n, 1);  // x1 + x2 -2 >= 0
    Cineq << 1.0, 1.0;
    VectorXd Bineq(1);
    Bineq << -2;
    VectorXd xl(2);
    VectorXd xu(2);
    xl << 0, 0;
    xu << 1e6, 1e6;

    ProjectedDualActiveSetQP solver(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    auto res = solver.solve(1e-9, 200);
    EXPECT_TRUE(res.success);

    auto osqp_res = solveWithOSQPEigen(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    std::cout << "qp x*: " << res.x.transpose() << std::endl;
    std::cout << "osqp x*: " << osqp_res.transpose() << std::endl;
}


TEST(QPSolverTest, InfeasibleProblemCase5)
{
    int n = 1;
    MatrixXd G(n, n);
    G << 2.0;
    VectorXd a(n);
    a << 0.0;
    // equality x = 0
    MatrixXd Ceq(n, 1);
    Ceq.col(0) << 1.0;
    VectorXd Beq(1);
    Beq(0) = 1.0;  // x + 1 = 0 -> x = -1
    // inequality x + 2 >= 0 -> x >= -2 -> consistent
    MatrixXd Cineq(n, 0);
    VectorXd Bineq(0);
    VectorXd xl(n);
    xl << 0.0;  // xl = 0 contradicts equality x = -1 -> infeasible
    VectorXd xu(n);
    xu << 10.0;

    ProjectedDualActiveSetQP solver(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    auto res = solver.solve(1e-9, 200);
    std::cout << "qp solve status: " << res.success << std::endl;
    EXPECT_FALSE(res.success);
    // Our solver should detect infeasible (or fail); either res.success==false OR osqp says infeasible
    auto osqp_res = solveWithOSQPEigen(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    std::cout << "qp x*: " << res.x.transpose() << std::endl;
    std::cout << "osqp x*: " << osqp_res.transpose() << std::endl;
    // expect_ret(res.x, osqp_res, 1e-3);
}

TEST(QPSolverTest, DegenerateConstraintsCase6)
{
    int n = 2;
    MatrixXd G(n, n);
    G << 2, 0, 0, 2;
    VectorXd a(n);
    a << 0, 0;
    // Two inequalities identical: x1 >= 0 and x1 >= 0 (duplicate)
    MatrixXd Cineq(n, 2);
    Cineq.col(0) << 1, 0;
    Cineq.col(1) << 1, 0;
    VectorXd Bineq(2);
    Bineq << 0, 0;
    MatrixXd Ceq(n, 0);
    VectorXd Beq(0);
    VectorXd xl(n), xu(n);
    xl << -1e6, -1e6;
    xu << 1e6, 1e6;

    ProjectedDualActiveSetQP solver(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    auto res = solver.solve(1e-9, 200);
    // solver should succeed and produce x ~ 0
    EXPECT_TRUE(res.success);
    auto osqp_res = solveWithOSQPEigen(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);

    std::cout << "qp x*: " << res.x.transpose() << std::endl;
    std::cout << "osqp x*: " << osqp_res.transpose() << std::endl;
    expect_ret(res.x, osqp_res, 1e-5);
}

TEST(QPSolverTest, SimpleEqualityAndInequalityCase7)
{
    MatrixXd G(2, 2);
    G << 4, 1, 1, 2;
    VectorXd a(2);
    a << -8, -3;

    MatrixXd Ceq(2, 1);
    Ceq.col(0) << 1, 1;
    VectorXd Beq(1);
    Beq << -1;  // x1+x2-1=0

    MatrixXd Cineq(2, 1);
    Cineq.col(0) << 1, 0;
    VectorXd Bineq(1);
    Bineq << -0.2;  // x1-0.2>=0

    VectorXd xl(2), xu(2);
    xl << 0.0, 0.0;
    xu << 5.0, 5.0;

    ProjectedDualActiveSetQP mySolver(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);
    QPResult res = mySolver.solve(1e-12, 200);

    ASSERT_TRUE(res.success);

    VectorXd osqp_sol = solveWithOSQPEigen(G, a, Ceq, Beq, Cineq, Bineq, xl, xu);

    ASSERT_NEAR(res.x(0), osqp_sol(0), 1e-3);
    ASSERT_NEAR(res.x(1), osqp_sol(1), 1e-3);
    std::cout << "qp x*: " << res.x.transpose() << std::endl;
    std::cout << "osqp x*: " << osqp_sol.transpose() << std::endl;
    expect_ret(res.x, osqp_sol, 1e-3);
}