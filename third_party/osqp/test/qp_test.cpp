///
/// @file
/// @copyright Copyright (C) 2021-2023, smart Automobile Co., Ltd (smart JV)
///

// gtest
#include <gtest/gtest.h>

// OsqpEigen
#include <OsqpEigen/OsqpEigen.h>

TEST(OSQPEigenTests, QPProblem_Unconstrained)
{
    constexpr double tolerance = 1e-4;

    Eigen::SparseMatrix<c_float> H_s(2, 2);
    H_s.insert(0, 0) = 3;
    H_s.insert(0, 1) = 2;
    H_s.insert(1, 0) = 2;
    H_s.insert(1, 1) = 4;

    Eigen::Matrix<c_float, 2, 1> gradient;
    gradient << 3, 1;

    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(true);
    solver.settings()->setAlpha(1.0);

    solver.data()->setNumberOfVariables(2);
    solver.data()->setNumberOfConstraints(0);

    EXPECT_TRUE(solver.data()->setHessianMatrix(H_s));
    EXPECT_TRUE(solver.data()->setGradient(gradient));

    EXPECT_TRUE(solver.initSolver());
    EXPECT_TRUE(solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError);

    // expected solution
    Eigen::Matrix<c_float, 2, 1> expectedSolution;
    expectedSolution << -1.2500, 0.3750;

    EXPECT_TRUE(solver.getSolution().isApprox(expectedSolution, tolerance));
}

TEST(OSQPEigenTests, QPProblem)
{
    constexpr double tolerance = 1e-4;

    Eigen::SparseMatrix<c_float> H_s(2, 2);
    H_s.insert(0, 0) = 4;
    H_s.insert(0, 1) = 1;
    H_s.insert(1, 0) = 1;
    H_s.insert(1, 1) = 2;

    Eigen::SparseMatrix<c_float> A_s(3, 2);
    A_s.insert(0, 0) = 1;
    A_s.insert(0, 1) = 1;
    A_s.insert(1, 0) = 1;
    A_s.insert(2, 1) = 1;

    Eigen::Matrix<c_float, 2, 1> gradient;
    gradient << 1, 1;

    Eigen::Matrix<c_float, 3, 1> lowerBound;
    lowerBound << 1, 0, 0;

    Eigen::Matrix<c_float, 3, 1> upperBound;
    upperBound << 1, 0.7, 0.7;

    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(true);
    solver.settings()->setAlpha(1.0);

    EXPECT_FALSE(solver.data()->setHessianMatrix(H_s));
    solver.data()->setNumberOfVariables(2);

    solver.data()->setNumberOfConstraints(3);
    EXPECT_TRUE(solver.data()->setHessianMatrix(H_s));
    EXPECT_TRUE(solver.data()->setGradient(gradient));
    EXPECT_TRUE(solver.data()->setLinearConstraintsMatrix(A_s));
    EXPECT_TRUE(solver.data()->setLowerBound(lowerBound));
    EXPECT_TRUE(solver.data()->setUpperBound(upperBound));

    EXPECT_TRUE(solver.initSolver());

    EXPECT_TRUE(solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError);
    Eigen::Matrix<c_float, 2, 1> expectedSolution;
    expectedSolution << 0.3, 0.7;

    EXPECT_TRUE(solver.getSolution().isApprox(expectedSolution, tolerance));
}
