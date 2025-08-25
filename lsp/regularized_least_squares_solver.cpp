#include "lsp/regularized_least_squares_solver.h"

#include <random>

#include <Eigen/QR>
#include <Eigen/SVD>
#pragma comment(linker, "/STACK:120000000")  // 120MB 栈空间

namespace
{

void InternalSolveStep(const Eigen::ColPivHouseholderQR<Eigen::Ref<DefMatrixA>>& qr,
                       const DefVectorb& vectorB,
                       const Eigen::internal::UpperBidiagonalization<DefMatrixR>& bidiag,
                       const DefMatrixQ_mu& Q_mu,
                       const std::int32_t number_of_rows,
                       const std::int32_t number_of_cols,
                       DefVectorb& b_derived)
{
    // 避免创建临时矩阵W，直接使用块操作
    // 计算 transpose(W)*b, 3.3
    // W.topRightCorner = bidiag.householderU()
    // 直接计算: transpose(bidiag.householderU()) * (Q^T * b)的前number_of_cols个元素

    // 先计算 Q^T * b 的前number_of_cols个元素
    //
    Eigen::VectorXf qt_b_top =
        (qr.householderQ().setLength(qr.nonzeroPivots()).transpose() * vectorB).head(number_of_cols);

    // 直接计算 transpose(bidiag.householderU()) * qt_b_top
    //
    b_derived.head(number_of_cols) = bidiag.householderU().transpose() * qt_b_top;

    // transpose(Q_mu)*b_derived, 3.5
    b_derived.head(number_of_cols) =
        Q_mu.topLeftCorner(number_of_cols, number_of_cols) * b_derived.head(number_of_cols);
}

void MultiplyAWithLInverse(const DefMatrixR& matrixLinv,
                           const std::int32_t number_of_rows,
                           const std::int32_t number_of_cols,
                           Eigen::Ref<DefMatrixA> matrixA)
{
    // 使用列操作，因为matrixLinv是对角矩阵
    //
    // #pragma omp parallel for schedule(static) num_threads(4)
    for (int j = 0; j < number_of_cols; ++j)
    {
        matrixA.col(j).array() *= matrixLinv(j, j);
    }
}

void UpperTridiagonalMatrixForA(const Eigen::ColPivHouseholderQR<Eigen::Ref<DefMatrixA>>& qr,
                                const std::int32_t number_of_cols,
                                DefMatrixR& matrixR)
{
    matrixR.topLeftCorner(number_of_cols, number_of_cols) =
        qr.matrixQR().topRows(number_of_cols).triangularView<Eigen::Upper>();
}
}  // namespace

bool RegularizedLeastSquaresSolver::InitializeSolver(const std::int32_t number_of_rows,
                                                     const std::int32_t number_of_cols)
{
    if ((number_of_rows > kMaxNumberOfRows) || (number_of_cols > kMaximumNumberOfCols))
    {
        initialization_done_ = false;
    }
    else
    {
        // 预分配内存
        //
        vectorX_.resize(number_of_cols, 1);
        matrixLinv_.resize(number_of_cols, number_of_cols);
        matrixR_.resize(number_of_cols, number_of_cols);
        B_mu_.resize(2 * number_of_cols, number_of_cols);
        Q_mu_.resize(2 * number_of_cols, number_of_cols);
        b_derived_.resize(number_of_cols);

        vectorX_.setZero();
        matrixLinv_.setZero();
        matrixR_.setZero();
        B_mu_.setZero();
        Q_mu_.setIdentity();
        b_derived_.setZero();

        number_of_rows_ = number_of_rows;
        number_of_cols_ = number_of_cols;
        calculation_done_ = false;
        initialization_done_ = true;
    }

    return initialization_done_;
}

///
/// 设置 Q_mu 在位置 (row, col) 处的值。若该值不为零，
/// 则会与当前值相乘。
///
void RegularizedLeastSquaresSolver::MultQ_Mu(DefMatrixQ_mu& Q_mu,
                                             const std::int32_t row,
                                             const std::int32_t col,
                                             const float factor)
{
    if (Q_mu(row, col) > std::numeric_limits<float>::epsilon())
    {
        Q_mu(row, col) = Q_mu(row, col) * factor;
    }
}

///
/// 高效地将矩阵 B_mu 转换为上双对角格式，并通过Givens变换实现。
///
/// @参数[输入/输出] Q_mu 存储平面旋转变换组合的矩阵，该变换应用于 B_mu。
/// @参数[输入/输出] B_mu 待转换为上双对角格式的矩阵。
///
/// @参数[输入] top_left_value 存储于 (lower_index, lower_index)
/// @参数[输入] top_right_value 存储于 (lower_index, upper_index)
/// @参数[输入] bottom_left_value 存储于 (upper_index, lower_index)
/// @参数[输入] bottom_right_value 存储于 (upper_index, upper_index)
///
void RegularizedLeastSquaresSolver::ExecuteGivenRotation(DefMatrixQ_mu& Q_mu,
                                                         const std::int32_t lower_index,
                                                         const std::int32_t upper_index,
                                                         const float top_left_value,
                                                         const float top_right_value,
                                                         const float bottom_left_value,
                                                         const float bottom_right_value)
{
    // 使用Eigen的向量操作
    //
    Eigen::VectorXf tmp_lower = Q_mu.row(lower_index);
    Eigen::VectorXf tmp_upper = Q_mu.row(upper_index);

    Q_mu.row(lower_index) = top_left_value * tmp_lower + top_right_value * tmp_upper;
    Q_mu.row(upper_index) = bottom_left_value * tmp_lower + bottom_right_value * tmp_upper;
}

///
/// 以高效的方式通过平面旋转变换将矩阵 B_mu 转换为上双对角格式。
///
/// @param[输入/输出] Q_mu 存储平面旋转变换组合的矩阵，该变换将应用于 B_mu。
/// @param[输入/输出] B_mu 需要被转换为上双对角格式的矩阵。
///
void RegularizedLeastSquaresSolver::CalculateB_mu(DefMatrixQ_mu& Q_mu, DefMatrixQ_mu& B_mu)
{
    Eigen::JacobiRotation<float> Givens;

    for (auto i = 0; i < number_of_cols_ - 1; i++)
    {
        auto p_x_idx = i;
        auto p_y_idx = number_of_cols_ + i;

        if (std::abs(B_mu(p_y_idx, p_x_idx)) > std::numeric_limits<float>::epsilon())
        {
            Givens.makeGivens(B_mu(p_x_idx, p_x_idx), B_mu(p_y_idx, p_x_idx));

            const float c = Givens.c();
            const float s = Givens.s();

            ExecuteGivenRotation(Q_mu, p_x_idx, p_y_idx, c, -s, s, c);

            B_mu(p_y_idx, p_x_idx) = 0.0F;
            B_mu(p_y_idx, p_x_idx + 1) = s * B_mu(p_x_idx, p_x_idx + 1);
            B_mu(p_x_idx, p_x_idx + 1) = c * B_mu(p_x_idx, p_x_idx + 1);
            B_mu(p_x_idx, p_x_idx) = Givens.c() * B_mu(p_x_idx, p_x_idx) + Givens.s() * B_mu(p_y_idx, p_x_idx);

            // 2. Execute a plane rotation in the (number_of_cols+i, number_of_cols+i+1) plane
            Givens.makeGivens(B_mu(p_y_idx, p_x_idx + 1), B_mu(p_y_idx + 1, p_x_idx + 1));

            ExecuteGivenRotation(Q_mu, p_y_idx, p_y_idx + 1, Givens.s(), Givens.c(), Givens.c(), -Givens.s());
            B_mu(p_y_idx + 1, p_x_idx + 1) =
                Givens.c() * B_mu(p_y_idx, p_x_idx + 1) + Givens.s() * B_mu(p_y_idx + 1, p_x_idx + 1);
            B_mu(p_y_idx, p_x_idx + 1) = 0.0F;
        }
    }

    // Execute (last) rotation in ( 2*number_of_cols-1, number_of_cols-1) plane
    auto p_x_idx = number_of_cols_ - 1;
    auto p_y_idx = 2 * number_of_cols_ - 1;

    if (std::abs(B_mu(p_y_idx, p_x_idx)) > std::numeric_limits<float>::epsilon())
    {
        Givens.makeGivens(B_mu(p_x_idx, p_x_idx), B_mu(p_y_idx, p_x_idx));

        ExecuteGivenRotation(Q_mu, p_x_idx, p_y_idx, Givens.c(), -Givens.s(), Givens.s(), Givens.c());

        B_mu(p_y_idx, p_x_idx) = 0.0F;
        B_mu(p_x_idx, p_x_idx) = Givens.c() * B_mu(p_x_idx, p_x_idx) + Givens.s() * B_mu(p_y_idx, p_x_idx);
    }
}

///
/// 通过反向传播算法求解问题 B_mux = b。注意：此处的 B_mu 必须是上三角矩阵，否则将无法得到正确结果。
///
/// @param[in] B_mu 作为问题 B_mux = b 输入参数的矩阵
/// @param[in] b 作为问题 B_mux = b 输入参数的向量
/// @param[out] x 存储问题 B_mux = b 求解结果的向量
///
bool RegularizedLeastSquaresSolver::ExecuteBacksubstitution(const DefMatrixQ_mu& B_mu,
                                                            const DefVectorb& b,
                                                            DefVectorx& x) const
{
    bool return_value = true;
    const std::ptrdiff_t max_size = static_cast<std::ptrdiff_t>(number_of_cols_) - 1;
    for (std::ptrdiff_t i = max_size; ((i >= 0) && return_value); --i)
    {
        return_value = false;
        const float diag = B_mu(i, i);

        if (std::abs(diag) >= std::numeric_limits<float>::epsilon())
        {
            if (i == max_size)
            {
                x(i) = b(i) / diag;
            }
            else
            {
                // 使用向量点积
                x(i) = (b(i) - B_mu.row(i).segment(i + 1, max_size - i).dot(x.segment(i + 1, max_size - i))) / diag;
            }
            return_value = true;
        }
    }
    return return_value;
}

bool RegularizedLeastSquaresSolver::SetMu(const float mu)
{
    bool result = false;
    if (mu >= 0.0F)
    {
        mu_ = mu;
        sqrt_of_mu_ = sqrtf(mu_);
        result = true;
    }
    else
    {
    }
    return result;
}

bool RegularizedLeastSquaresSolver::SetDiagonalValueOfL(const float value)
{
    bool result = true;
    if (initialization_done_)
    {
        for (auto i = 0U; i < number_of_cols_; ++i)
        {
            result &= SetDiagonalValueOfL(i, value);
        }
    }
    return result;
}

bool RegularizedLeastSquaresSolver::SetDiagonalValueOfL(const std::uint32_t i, const float value)
{
    bool result = false;
    if (initialization_done_)
    {
        if (i < static_cast<std::uint32_t>(number_of_cols_))
        {
            if (fabsf(value) > std::numeric_limits<float>::epsilon())
            {
                L_values_inv_[i] = 1.0F / value;
                result = true;
            }
        }
    }
    return result;
}

bool RegularizedLeastSquaresSolver::FillRandomData()
{
    if (initialization_done_)
    {

        float low = -100.0f;
        float high = 100.0f;
        matrixA_ = DefMatrixA::NullaryExpr(number_of_rows_, number_of_cols_, [&]() {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dis(low, high);
            return dis(gen);
        });

        vectorB_ = DefVectorb::NullaryExpr(number_of_rows_, [&]() {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dis(low, high);
            return dis(gen);
        });
    }
    return initialization_done_;
}

///
/// The solution of this algorithm is from the paper
/// L. Elden, Algorithms for the regularization of ill-conditioned least squares problems
///
DefVectorx RegularizedLeastSquaresSolver::solve()
{
    if (initialization_done_)
    {
        bool result = false;
        /// Calculate the inverse of matrix L. Because it's a diagonal matrix, the
        /// inverse is simply the inverse of each diagonal entry.
        /// DefMatrixR matrixLinv_(number_of_cols_, number_of_cols_);
        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(number_of_cols_); i++)
        {
            matrixLinv_(i, i) = L_values_inv_[i];
        }
        /// Multiply A with the inverse of L. With this step, the problem can be transformed into a simpler
        /// problem. (使用A的引用，避免拷贝)
        ///

        MultiplyAWithLInverse(matrixLinv_, number_of_rows_, number_of_cols_, matrixA_);
        /// Now solve  min{ || A x' - b || ^ 2 + mu  || x'  || ^ 2 }
        /// DefMatrixR matrixR(number_of_cols_, number_of_cols_);
        /// Execute a tridiagonalization of the matrix A, 3.1(i)
        Eigen::Ref<DefMatrixA> A_ref = matrixA_;

        Eigen::ColPivHouseholderQR<Eigen::Ref<DefMatrixA>> qr(matrixA_);

        UpperTridiagonalMatrixForA(qr, number_of_cols_, matrixR_);
        if (qr.rank() >= number_of_cols_)
        {
            result = true;
        }

        if (result)
        {
            // 使用引用避免拷贝
            //
            Eigen::Ref<DefMatrixR> R_ref = matrixR_.topLeftCorner(number_of_cols_, number_of_cols_);

            /// Create a bidiagonalization of the matrix R, 3.1(ii)
            Eigen::internal::UpperBidiagonalization<DefMatrixR> bidiag(R_ref);
            /// Build (B / sqrt_of_mu_*I), 3.4
            // DefMatrixQ_mu B_mu(2 * number_of_cols_, number_of_cols_);

            B_mu_.topRows(number_of_cols_) = bidiag.bidiagonal().toDenseMatrix();

            for (std::int32_t i = 0; i < number_of_cols_; i++)
            {
                B_mu_(i + number_of_cols_, i) = sqrt_of_mu_;
            }

            // DefMatrixQ_mu Q_mu(2 * number_of_cols_, 2 * number_of_cols_);

            /// Calculate Q_mu, 3.7 and
            /// B_mu = transpose(Q_mu)*(B / sqrt_of_mu_*I), 3.5
            CalculateB_mu(Q_mu_, B_mu_);

            InternalSolveStep(qr, vectorB_, bidiag, Q_mu_, number_of_rows_, number_of_cols_, b_derived_);
            /// Solve B_mu * x_derived = b_derived via backsubstitution, 3.6
            if (ExecuteBacksubstitution(B_mu_, b_derived_, vectorX_))
            {
                /// Multiply V with the current result of 3.6, 3.3
                vectorX_ = bidiag.householderV() * vectorX_;

                /// Turn back the column permutation, which is done during the execution of ColPivHouseHolderQR.
                // 应用列置换
                //
                DefVectorx x_copy = vectorX_;
                for (auto i = 0; i < number_of_cols_; ++i)
                {
                    vectorX_(qr.colsPermutation().indices()(i)) = x_copy(i);
                }

                /// Multiply the inverse of L with the (temporary) result x, to get the final result of the regularized
                /// least squares problem.
                /// 应用L的逆
                ///
                vectorX_.noalias() = matrixLinv_.diagonal().asDiagonal() * vectorX_;

                calculation_done_ = true;
            }
        }
        else
        {
        }
    }

    return vectorX_;
}
