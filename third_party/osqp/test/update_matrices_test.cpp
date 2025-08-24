///
/// @file
/// @copyright Copyright (C) 2021-2023, smart Automobile Co., Ltd (smart JV)
///

// gtest
#include <gtest/gtest.h>

// OsqpEigen
#include <OsqpEigen/OsqpEigen.h>

// colors
#define ANSI_TXT_GRN "\033[0;32m"
#define ANSI_TXT_MGT "\033[0;35m"  // Magenta
#define ANSI_TXT_DFT "\033[0;0m"   // Console default
#define GTEST_BOX "[     cout ] "
#define COUT_GTEST ANSI_TXT_GRN << GTEST_BOX  // You could add the Default
#define COUT_GTEST_MGT COUT_GTEST << ANSI_TXT_MGT

Eigen::Matrix<c_float, 2, 2> H;
Eigen::SparseMatrix<c_float> H_s;
Eigen::Matrix<c_float, 3, 2> A;
Eigen::SparseMatrix<c_float> A_s;
Eigen::Matrix<c_float, 2, 1> gradient;
Eigen::Matrix<c_float, 3, 1> lowerBound;
Eigen::Matrix<c_float, 3, 1> upperBound;

OsqpEigen::Solver solver;

TEST(OSQPEigenTests, QPProblemFirstRun)
{
    // hessian matrix
    H << 4, 0, 0, 2;
    H_s = H.sparseView();
    H_s.pruned(0.01);

    // linear constraint matrix
    A << 1, 1, 1, 0, 0, 1;
    A_s = A.sparseView();

    gradient << 1, 1;
    lowerBound << 1, 0, 0;
    upperBound << 1, 0.7, 0.7;

    solver.settings()->setVerbosity(false);

    solver.data()->setNumberOfVariables(2);
    solver.data()->setNumberOfConstraints(3);
    solver.settings()->setScaling(0);
    EXPECT_TRUE(solver.data()->setHessianMatrix(H_s));
    EXPECT_TRUE(solver.data()->setGradient(gradient));
    EXPECT_TRUE(solver.data()->setLinearConstraintsMatrix(A_s));
    EXPECT_TRUE(solver.data()->setLowerBound(lowerBound));
    EXPECT_TRUE(solver.data()->setUpperBound(upperBound));

    EXPECT_TRUE(solver.initSolver());
    EXPECT_TRUE(solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError);

    auto solution = solver.getSolution();
    std::cout << COUT_GTEST_MGT << "Solution [" << solution(0) << " " << solution(1) << "]" << ANSI_TXT_DFT
              << std::endl;
}

TEST(OSQPEigenTests, QPProblemSparsityConstant)
{
    // update hessian matrix
    H << 4, 0, 0, 2;
    H_s = H.sparseView();
    A << 2, 1, 1, 0, 0, 1;
    A_s = A.sparseView();

    EXPECT_TRUE(solver.updateHessianMatrix(H_s));
    EXPECT_TRUE(solver.updateLinearConstraintsMatrix(A_s));
    EXPECT_TRUE(solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError);

    auto solution = solver.getSolution();
    std::cout << COUT_GTEST_MGT << "Solution [" << solution(0) << " " << solution(1) << "]" << ANSI_TXT_DFT
              << std::endl;
};

TEST(OSQPEigenTests, QPProblemSparsityChange)
{
    // update hessian matrix
    H << 1, 1, 1, 2;
    H_s = H.sparseView();
    A << 1, 1, 1, 0.4, 0, 1;
    A_s = A.sparseView();

    EXPECT_TRUE(solver.updateHessianMatrix(H_s));
    EXPECT_TRUE(solver.updateLinearConstraintsMatrix(A_s));
    EXPECT_TRUE(solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError);

    auto solution = solver.getSolution();
    std::cout << COUT_GTEST_MGT << "Solution [" << solution(0) << " " << solution(1) << "]" << ANSI_TXT_DFT
              << std::endl;
};
