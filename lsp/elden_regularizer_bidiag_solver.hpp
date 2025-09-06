// elden_regularizer_bidiag.hpp
// Full implementation following Lars Eldén (1977):
//  - Transform general L != I Tikhonov to standard form (section 2, eq (2.3)-(2.12))
//  - For standardized problem, perform Golub-Kahan bidiagonalization (eq (3.2))
//  - For each mu, perform Givens rotations (QR) on [B; sqrt(mu) I] and apply same rotations to [g1; 0]
//  - Solve triangular system to obtain f1, map back to original f via eq (2.12)
//

#ifndef LSP_ELDEN_REGULARIZER_BIDIAG_SOLVER_H
#define LSP_ELDEN_REGULARIZER_BIDIAG_SOLVER_H

#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include <Eigen/Dense>

using Eigen::Index;
using Eigen::MatrixXf;
using Eigen::VectorXf;

// Utility: apply Householder reflector stored as v (with v[0]=1 convention) and tau to a vector x (in place).
// Reflector H = I - tau * v * v^T
static void apply_householder_from_left(VectorXf& x, const VectorXf& v, float tau)
{
    // x := (I - tau v v^T) x
    float dot = v.dot(x);
    x -= tau * dot * v;
}
static void apply_householder_on_matrix_left(MatrixXf& A, Index rowStart, Index colStart, const VectorXf& v, float tau)
{
    // Apply H = I - tau v v^T to A(rowStart:, colStart:)
    MatrixXf sub = A.block(rowStart, colStart, A.rows() - rowStart, A.cols() - colStart);
    VectorXf w = v.transpose() * sub;  // size = cols-sub
    sub -= v * (tau * w.transpose());
    A.block(rowStart, colStart, A.rows() - rowStart, A.cols() - colStart) = sub;
}
static void apply_householder_on_matrix_right(MatrixXf& A, Index rowStart, Index colStart, const VectorXf& v, float tau)
{
    // Apply H = I - tau v v^T on the right: A := A * H, where H acts on columns (v defined over columns)
    MatrixXf sub = A.block(rowStart, colStart, A.rows() - rowStart, A.cols() - colStart);
    VectorXf w = sub * v;  // size = rows-sub
    sub -= (tau * w) * v.transpose();
    A.block(rowStart, colStart, A.rows() - rowStart, A.cols() - colStart) = sub;
}

// Givens rotation on pair (a,b): compute c,s so that [c s; -s c]^T * [a; b] = [r; 0] with r = sqrt(a^2 + b^2)
inline void givens_float(float a, float b, float& c, float& s, float& r)
{
    if (b == 0.0F)
    {
        c = 1.0F;
        s = 0.0F;
        r = a;
        return;
    }
    if (std::abs(b) > std::abs(a))
    {
        float t = a / b;
        s = 1.0F / std::sqrt(1.0F + t * t);
        c = s * t;
        r = b / s;
    }
    else
    {
        float t = b / a;
        c = 1.0F / std::sqrt(1.0F + t * t);
        s = c * t;
        r = a / c;
    }
}

// The solver class
class EldenRegularizer
{
  public:
    EldenRegularizer(const MatrixXf& K_in) : K(K_in)
    {
        m = Index(K.rows());
        n = Index(K.cols());
        mu = 0.1 * static_cast<float>(m);
        // default L = 0.01 * I_n
        L = MatrixXf::Zero(n, n);
        L.diagonal().setConstant(0.01);
        preprocessed = false;
    }

    void SetMu(float new_mu)
    {
        if (new_mu <= 0.0F)
            throw std::invalid_argument("mu must be positive");
        mu = new_mu;
    }
    void SetL(const MatrixXf& L_in)
    {
        if ((Index)L_in.rows() != n || (Index)L_in.cols() != n)
            throw std::invalid_argument("L size mismatch");
        L = L_in;
        preprocessed = false;
    }

    // Main API: solve for f given g
    VectorXf solve(const VectorXf& g_in)
    {
        if ((Index)g_in.size() != m)
            throw std::invalid_argument("g size mismatch");
        g = g_in;
        if (!preprocessed)
            preprocess();
        // Compute gbar = Q2^T * g (eq 2.11)
        VectorXf gbar = Q2.transpose() * g;  // dimension r = m - (n-p)
        // Now solve standard-form Tikhonov using bidiagonalization + Givens per mu
        // Steps:
        //   * Ensure Kbar and gbar set (done in preprocess); perform bidiagonalization Kbar = W [B;0] Y^T
        //   * compute g1 = W^T * gbar
        //   * for mu: form augmented A_aug = [B; sqrt(mu) I] and g_aug = [g1; 0], do Givens QR -> R = B_mu, ghat =
        //   rotated g_aug top part
        //   * Solve R f1 = ghat, compute z = Y * f1, backtransform (2.12)
        if (!bidiag_done)
        {
            bidiagonalize_Kbar();  // produces B (diags alpha, beta) and W-reflectors, Y-reflectors, and g1
        }
        // solve via bidiag + Givens for current mu
        VectorXf f = solve_for_mu_and_backtransform(mu);
        return f;
    }

  private:
    // Input
    MatrixXf K;
    MatrixXf L;
    VectorXf g;
    float mu;
    Index m, n;
    bool preprocessed;

    // --- Section 2 decomposition variables (paper notation) ---
    // L^T = V [ R ; 0 ]  with V = [V1 V2]
    MatrixXf V1, V2;
    MatrixXf R;  // p x p (upper triangular)
    Index p;     // rank(L) numerically

    // K V2 = Q [ U ; 0 ] with Q = [Q1 Q2]
    MatrixXf Q1, Q2;
    MatrixXf U;  // (n-p)x(n-p) upper triangular

    // Standard form matrix (eq 2.11)
    MatrixXf Kbar;  // size r x p where r = m - (n-p)
    Index r;        // r = rows of Kbar

    bool bidiag_done = false;

    // --- Golub-Kahan bidiagonalization storage (for Kbar) ---
    // We'll store left Householder vectors v_k (for W), right Householder vectors u_k (for Y),
    // and the bidiagonal elements alpha (diag) and beta (superdiag).
    std::vector<VectorXf> left_vs;   // v vectors (length r-k) for each k
    std::vector<float> left_taus;    // tau for each left reflector
    std::vector<VectorXf> right_vs;  // u vectors (length p-(k+1)) for each k (when k < p-1)
    std::vector<float> right_taus;   // tau for right reflectors
    std::vector<float> alpha;        // diag of B (size p)
    std::vector<float> beta;         // superdiag of B (size p-1)
    VectorXf g1;                     // g1 = W^T * gbar (length r)
    Index p_eff;                     // p (columns of Kbar)

    // Section 2 preprocess: compute V1,V2,R and Q1,Q2,U and Kbar
    void preprocess()
    {
        // Step (2.3): QR of L^T -> L^T = V * [R; 0]
        // We'll use HouseholderQR to get Q (i.e., V) and Rfull. Then determine numerical rank p.
        Eigen::HouseholderQR<MatrixXf> qrLt(L.transpose());
        MatrixXf V = qrLt.householderQ();                                 // n x n orthogonal
        MatrixXf Rfull = qrLt.matrixQR().triangularView<Eigen::Upper>();  // n x n
        // numeric rank p from diagonal of Rfull
        float tol = 1e-12;
        p = n;
        int rankCount = 0;
        for (Index i = 0; i < n; ++i)
            if (std::abs(Rfull(i, i)) > tol)
                ++rankCount;
        p = rankCount;
        if (p <= 0)
            throw std::runtime_error("L appears rank-deficient");
        V1 = V.leftCols(p);
        if (p < n)
            V2 = V.rightCols(n - p);
        else
            V2 = MatrixXf::Zero(n, 0);
        // Extract R as top-left p x p of Rfull
        R = Rfull.topLeftCorner(p, p).triangularView<Eigen::Upper>();

        // Step (2.6): compute K * V2 = Q [ U ; 0 ]
        if (p < n)
        {
            MatrixXf KV2 = K * V2;  // m x (n-p)
            Eigen::HouseholderQR<MatrixXf> qrKV2(KV2);
            MatrixXf Q = qrKV2.householderQ();
            MatrixXf R2 = qrKV2.matrixQR().triangularView<Eigen::Upper>();
            Q1 = Q.leftCols(n - p);
            if (Q.cols() > (Index)(n - p))
                Q2 = Q.rightCols(Q.cols() - (n - p));
            else
                Q2 = MatrixXf::Zero(m, 0);
            U = R2.topLeftCorner(n - p, n - p).triangularView<Eigen::Upper>();
        }
        else
        {
            // p == n
            Q1 = MatrixXf::Zero(m, 0);
            Q2 = MatrixXf::Identity(m, m);
            U = MatrixXf::Zero(0, 0);
        }

        // Step (2.11): Kbar = Q2^T * K * V1 * R^{-T};  size r x p, where r = m - (n - p)
        MatrixXf K_times_V1 = K * V1;  // m x p
        MatrixXf Q2t = Q2.transpose();
        MatrixXf R_invT = R.triangularView<Eigen::Upper>().transpose().solve(MatrixXf::Identity(p, p));  // R^{-T}
        Kbar = Q2t * K_times_V1 * R_invT;
        r = (Index)Kbar.rows();
        p_eff = p;
        preprocessed = true;
        bidiag_done = false;
    }

    // Golub-Kahan bidiagonalization of Kbar
    //   Kbar = W [ B ; 0 ] Y^T
    // where B is p x p upper bidiagonal
    // We store Householder reflectors for W and Y, and alpha,beta of B
    void bidiagonalize_Kbar()
    {
        MatrixXf A = Kbar;  // working copy (r x p)

        left_vs.clear();
        left_taus.clear();
        right_vs.clear();
        right_taus.clear();
        alpha.assign(p_eff, 0.0F);
        beta.assign(std::max(0.0F, p_eff - 1.0F), 0.0F);

        for (Index k = 0; k < p_eff; ++k)
        {
            // --- Step 1: Left Householder (on column k, rows k:r-1) ---
            Index m_k = r - k;
            VectorXf x = A.block(k, k, m_k, 1);

            VectorXf v = x;
            float sigma = x.tail(m_k - 1).norm();
            float tau = 0.0F;

            if (sigma == 0.0F && x(0) >= 0.0F)
            {
                // no reflection needed
                v.setZero();
                v(0) = 1.0F;
                tau = 0.0F;
            }
            else if (sigma == 0.0F && x(0) < 0.0F)
            {
                v.setZero();
                v(0) = 1.0F;
                tau = 2.0;
            }
            else
            {
                float alpha_k =
                    (x(0) >= 0.0F) ? -std::sqrt(x(0) * x(0) + sigma * sigma) : std::sqrt(x(0) * x(0) + sigma * sigma);
                v(0) = x(0) - alpha_k;
                v.tail(m_k - 1) = x.tail(m_k - 1);
                float denom = v.squaredNorm();
                if (denom > 0.0F)
                {
                    v /= v(0);  // normalize so v(0) = 1
                    tau = 2.0 / v.squaredNorm();
                }
                else
                {
                    tau = 0.0F;
                }
                alpha[k] = -alpha_k;
            }

            left_vs.push_back(v);
            left_taus.push_back(tau);

            // Apply H_left to A(k:, k:)
            if (tau != 0.0F)
            {
                MatrixXf Ablock = A.block(k, k, m_k, p_eff - k);
                VectorXf w = v.transpose() * Ablock;
                Ablock -= v * (tau * w.transpose());
                A.block(k, k, m_k, p_eff - k) = Ablock;
            }

            alpha[k] = A(k, k);  // updated diagonal entry

            // --- Step 2: Right Householder (on row k, cols k+1:p-1) ---
            if (k < p_eff - 1)
            {
                Index n_k = p_eff - (k + 1);
                VectorXf xrow = A.block(k, k + 1, 1, n_k).transpose();

                VectorXf u = xrow;
                float sigma2 = (n_k > 1) ? xrow.tail(n_k - 1).norm() : 0.0F;
                float taut = 0.0F;

                if (sigma2 == 0.0F && xrow(0) >= 0.0F)
                {
                    u.setZero();
                    u(0) = 1.0F;
                    taut = 0.0F;
                }
                else
                {
                    float beta_k = (xrow(0) >= 0.0F) ? -std::sqrt(xrow(0) * xrow(0) + sigma2 * sigma2)
                                                     : std::sqrt(xrow(0) * xrow(0) + sigma2 * sigma2);
                    u(0) = xrow(0) - beta_k;
                    u.tail(n_k - 1) = xrow.tail(n_k - 1);
                    float denom = u.squaredNorm();
                    if (denom > 0.0F)
                    {
                        u /= u(0);  // normalize so u(0) = 1
                        taut = 2.0 / u.squaredNorm();
                    }
                    else
                    {
                        taut = 0.0F;
                    }
                    beta[k] = -beta_k;
                }

                right_vs.push_back(u);
                right_taus.push_back(taut);

                if (taut != 0.0F)
                {
                    MatrixXf Ablock2 = A.block(k, k + 1, r - k, n_k);
                    VectorXf tmp = Ablock2 * u;
                    Ablock2 -= tmp * (taut * u.transpose());
                    A.block(k, k + 1, r - k, n_k) = Ablock2;
                }

                beta[k] = A(k, k + 1);  // updated superdiagonal entry
            }
        }

        bidiag_done = true;
    }

    // Given current bidiagonal alpha and beta and saved Householder reflectors,
    // compute g1 = W^T * gbar by applying left reflectors in order (W = H0 H1 ...).
    VectorXf compute_g1_from_gbar(const VectorXf& gbar)
    {
        VectorXf v = gbar;  // length r
        // left reflectors were stored left_vs[k] corresponding to positions starting at row k
        for (Index k = 0; k < (Index)left_vs.size(); ++k)
        {
            const VectorXf& hh = left_vs[k];
            float tau = left_taus[k];
            if (tau == 0.0F)
                continue;
            // apply H_k to v(k:)
            Index len = (Index)hh.size();
            VectorXf sub = v.segment(k, len);
            float dot = hh.dot(sub);
            sub -= tau * dot * hh;
            v.segment(k, len) = sub;
        }
        return v;
    }

    // Given f1 (solution in bidiagonal variable), compute z = Y * f1; Y constructed from right reflectors
    VectorXf apply_Y_to_f1(const VectorXf& f1)
    {
        // Right reflectors correspond to transforms on columns; they were applied in bidiag by:
        // A := A * Hk_right, where Hk_right = I - taut * u * u^T acting on columns k+1:.
        // The overall Y^T = H_right_{n-2} ... H_right_0. We need Y * f1, so apply those reflectors in reverse order.
        VectorXf x = f1;
        // right_vs[k] length = p - (k+1)
        for (int k = (int)right_vs.size() - 1; k >= 0; --k)
        {
            Index idx = k + 1;
            const VectorXf& u = right_vs[k];
            float taut = right_taus[k];
            if (taut == 0.0F)
                continue;
            // u is length n_k = p - (k+1), acting on x(idx : idx + n_k - 1)
            Index n_k = (Index)u.size();
            VectorXf sub = x.segment(idx, n_k);
            float dot = u.dot(sub);
            sub -= taut * dot * u;
            x.segment(idx, n_k) = sub;
        }
        return x;
    }

    // For a given mu, perform Givens QR of augmented matrix [B; sqrt(mu) I] (size 2p x p),
    // but B is stored via alpha (diag) and beta (superdiag). We'll construct the augmented matrix exploiting bidiagonal
    // sparsity, then perform classical Givens QR to produce upper triangular R = B_mu (p x p), and apply same rotations
    // to g_aug = [g1; 0].
    //
    // Return f1 by solving R f1 = hat_g1, where hat_g1 is top p entries after rotations.
    VectorXf solve_bidiag_augmented_and_solve(const VectorXf& g1_local, float mu_local)
    {
        // Dimensions: p_eff = p
        Index p = p_eff;

        // Build augmented matrix A_aug of size (p + p) x p
        Index rows = p + p;
        MatrixXf A_aug = MatrixXf::Zero(rows, p);

        // Fill top p rows with B (upper bidiagonal):
        // B(i,i) = alpha[i], B(i,i+1) = beta[i] (for i=0..p-2)
        for (Index i = 0; i < p; ++i)
        {
            A_aug(i, i) = alpha[i];
            if (i + 1 < p)
                A_aug(i, i + 1) = beta[i];
        }
        // Fill bottom p rows with sqrt(mu) * I
        float s = std::sqrt(mu_local);
        for (Index i = 0; i < p; ++i)
        {
            A_aug(p + i, i) = s;
        }
        // Augmented RHS: g_aug = [g1_local_head_p; zeros(p)]
        VectorXf g_aug = VectorXf::Zero(rows);
        // Note: g1_local has dimension r (rows of Kbar); in eq (3.4) only first p components g1 (top n) used (paper
        // notation). In derivation after bidiag g1 is of length r ; the top p components correspond to g1(0..p-1). We
        // take those.
        if ((Index)g1_local.size() < p)
            throw std::runtime_error("g1 size less than p in solve_bidiag_augmented_and_solve");
        for (Index i = 0; i < p; ++i)
            g_aug(i) = g1_local(i);

        // Now perform Givens QR to zero below-diagonal elements column by column.
        // We'll zero A_aug(i, j) for i = j+1 .. rows-1 with Givens rotations using pivot at row j.
        // Apply rotations to entire A_aug rows and to g_aug entries.
        for (Index j = 0; j < p; ++j)
        {
            // For rows i = rows-1 down to j+1, zero A_aug(i, j) using Givens with A_aug(j, j) as the other element.
            // But more efficient: we proceed downward eliminating the subdiagonal entries in column j.
            for (Index i = rows - 1; i > j; --i)
            {
                float a = A_aug(j, j);
                float b = A_aug(i, j);
                if (b == 0.0F)
                    continue;
                float c, srot, r;
                givens_float(a, b, c, srot, r);
                // Build rotation matrix G that acts on rows j and i (premultiply)
                // Apply to rows j and i for all columns >= j (columns < j already zero below)
                for (Index col = j; col < p; ++col)
                {
                    float temp_j = c * A_aug(j, col) + srot * A_aug(i, col);
                    float temp_i = -srot * A_aug(j, col) + c * A_aug(i, col);
                    A_aug(j, col) = temp_j;
                    A_aug(i, col) = temp_i;
                }
                // Apply same rotation to g_aug entries at positions j and i
                {
                    float gj = g_aug(j);
                    float gi = g_aug(i);
                    float new_gj = c * gj + srot * gi;
                    float new_gi = -srot * gj + c * gi;
                    g_aug(j) = new_gj;
                    g_aug(i) = new_gi;
                }
                // After rotation, A_aug(i,j) should be zero (or very small)
            }
        }
        // After Givens QR, top p x p block A_aug.topRows(p) is upper triangular R (call B_mu).
        MatrixXf R_top = A_aug.topRows(p);
        VectorXf hat_g1 = g_aug.head(p);

        // Solve R_top * f1 = hat_g1 by back substitution (R_top is upper triangular)
        VectorXf f1 = VectorXf::Zero(p);
        for (int i = (int)p - 1; i >= 0; --i)
        {
            float sum = 0.0F;
            for (Index j = i + 1; j < p; ++j)
                sum += R_top(i, j) * f1(j);
            float diag = R_top(i, i);
            if (std::abs(diag) < 1e-14)
            {
                // near singular, regularize slightly
                diag += 1e-14;
            }
            f1(i) = (hat_g1(i) - sum) / diag;
        }
        return f1;
    }

    // Top-level for given mu: compute g1 = W^T gbar (by applying left reflectors) then solve bidiag augmented and
    // obtain f1, then compute z = Y * f1 and finally backtransform to original f using eq (2.12).
    VectorXf solve_for_mu_and_backtransform(float mu_local)
    {
        // compute gbar
        VectorXf gbar = Q2.transpose() * g;  // r vector
        // compute g1 = W^T * gbar (we apply left reflectors)
        g1 = compute_g1_from_gbar(gbar);  // length r

        // Now call solver on bidiagonal structure to get f1 (size p)
        VectorXf f1 = solve_bidiag_augmented_and_solve(g1, mu_local);  // in bidiag coordinates (this is f1)

        // Map to z = Y * f1 (apply right reflectors)
        VectorXf z = apply_Y_to_f1(f1);  // z dimension p

        // Now backtransform per eq (2.12):
        // y1 = R^{-T} * z
        VectorXf y1 = R.triangularView<Eigen::Upper>().transpose().solve(z);  // p-vector
        // y2 = U^{-1} * Q1^T * (g - K * V1 * y1)  (if p < n)
        VectorXf y2;
        if (p < n)
        {
            VectorXf tmp = g - K * (V1 * y1);
            y2 = U.triangularView<Eigen::Upper>().solve(Q1.transpose() * tmp);
        }
        else
        {
            y2 = VectorXf::Zero(0);
        }
        // f = V1*y1 + V2*y2
        VectorXf f;
        if (p < n)
        {
            f = V1 * y1 + V2 * y2;
        }
        else
        {
            f = V1 * y1;
        }
        return f;
    }
};  // end class

#endif  // LSP_ELDEN_REGULARIZER_BIDIAG_SOLVER_H
