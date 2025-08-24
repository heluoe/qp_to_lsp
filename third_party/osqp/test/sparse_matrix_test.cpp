///
/// @file
/// @copyright Copyright (C) 2021-2023, smart Automobile Co., Ltd (smart JV)
///

// gtest
#include <gtest/gtest.h>

// OsqpEigen
#include <OsqpEigen/OsqpEigen.h>
#include <osqp.h>

template <typename T, int n, int m>
bool computeTest(const Eigen::Matrix<T, n, m>& mEigen)
{
    Eigen::SparseMatrix<T, Eigen::ColMajor> matrix, newMatrix, newMatrixFromCSR;
    matrix = mEigen.sparseView();

    csc* osqpSparseMatrix = nullptr;
    // NOTE: Dynamic memory allocation
    if (!OsqpEigen::SparseMatrixHelper::createOsqpSparseMatrix(matrix, osqpSparseMatrix))
        return false;

    Eigen::SparseMatrix<T, Eigen::RowMajor> csrMatrix;
    csrMatrix = matrix;
    csc* otherOsqpSparseMatrix = nullptr;
    if (!OsqpEigen::SparseMatrixHelper::createOsqpSparseMatrix(csrMatrix, otherOsqpSparseMatrix))
        return false;

    if (!OsqpEigen::SparseMatrixHelper::osqpSparseMatrixToEigenSparseMatrix(osqpSparseMatrix, newMatrix))
        return false;

    if (!OsqpEigen::SparseMatrixHelper::osqpSparseMatrixToEigenSparseMatrix(otherOsqpSparseMatrix, newMatrixFromCSR))
        return false;

    if (!newMatrixFromCSR.isApprox(newMatrix))
        return false;

    std::vector<Eigen::Triplet<T>> tripletListCsc;
    if (!OsqpEigen::SparseMatrixHelper::osqpSparseMatrixToTriplets(osqpSparseMatrix, tripletListCsc))
        return false;

    for (const auto& a : tripletListCsc)
        std::cout << a.row() << " " << a.col() << " " << a.value() << std::endl;

    std::vector<Eigen::Triplet<T>> tripletListEigen;
    OsqpEigen::SparseMatrixHelper::eigenSparseMatrixToTriplets(matrix, tripletListEigen);

    std::cout << "***********************************************" << std::endl;
    for (const auto& a : tripletListEigen)
        std::cout << a.row() << " " << a.col() << " " << a.value() << std::endl;

    constexpr double tolerance = 1e-4;
    bool outcome = matrix.isApprox(newMatrix, tolerance);

    csc_spfree(osqpSparseMatrix);
    csc_spfree(otherOsqpSparseMatrix);

    return outcome;
}

TEST(OSQPEigenTests, SparseMatrix_DataTypeWithDouble)
{

    Eigen::Matrix3d m;
    m << 0, 1.002311, 0, 0, 0, 0, 0, 0.90835435, 0;

    EXPECT_TRUE(computeTest(m));
}

TEST(OSQPEigenTests, SparseMatrix_DataTypeWithFloat)
{
    Eigen::Matrix3f m;
    m << 0, 1, 0, 0, 0, 0, 0, 1, 0;

    EXPECT_TRUE(computeTest(m));
}

TEST(OSQPEigenTests, SparseMatrix_DataTypeWithInt)
{

    Eigen::Matrix3i m;
    m << 0, 1, 0, 0, 0, 0, 0, 1, 0;

    EXPECT_TRUE(computeTest(m));
}

TEST(OSQPEigenTests, SparseMatrix_DataTypeWithIntToDouble)
{

    Eigen::Matrix<double, 4, 2> m;
    m << 0, 0, 0, 4, 0, 0, 0, 0;

    EXPECT_TRUE(computeTest(m));
}
