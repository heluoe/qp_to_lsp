#ifndef QP_SOLVER_PROJECTED_DUAL_QP_SOLVER_H
#define QP_SOLVER_PROJECTED_DUAL_QP_SOLVER_H

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct QPResult
{
    VectorXd x;
    bool success;
    string message;
    int iterations;
};

class ProjectedDualActiveSetQP
{
  public:
    // 构造函数接口：
    // G (n x n, PD), a (n)
    // Ceq (n x me), Beq (me)   // 等式: Ceq.col(i)^T x + Beq(i) == 0
    // Cineq (n x mi), Bineq(mi) // 不等式: Cineq.col(j)^T x + Bineq(j) >= 0
    // xl, xu: bounds (n) ; may contain -inf/inf (use large numbers or +/-infty)
    ProjectedDualActiveSetQP(const MatrixXd& G,
                             const VectorXd& a,
                             const MatrixXd& Ceq,
                             const VectorXd& Beq,
                             const MatrixXd& Cineq,
                             const VectorXd& Bineq,
                             const VectorXd& xl,
                             const VectorXd& xu)
        : G_(G), a_(a), Ceq_(Ceq), Beq_(Beq), Cineq_(Cineq), Bineq_(Bineq), xl_(xl), xu_(xu)
    {
        n = G_.rows();
        if (G_.rows() != G_.cols())
            throw runtime_error("G must be square");
        if (G_.rows() != a_.size())
            throw runtime_error("Dimension mismatch G / a");
        if (Ceq_.size() != 0 && Ceq_.rows() != n)
            throw runtime_error("Ceq row mismatch");
        if (Cineq_.size() != 0 && Cineq_.rows() != n)
            throw runtime_error("Cineq row mismatch");
        if (Beq_.size() != Ceq_.cols())
            throw runtime_error("Beq length mismatch");
        if (Bineq_.size() != Cineq_.cols())
            throw runtime_error("Bineq length mismatch");
        if (xl_.size() != n || xu_.size() != n)
            throw runtime_error("xl/xu size mismatch");

        me = Ceq_.cols();
        mi = Cineq_.cols();

        // Build combined inequality matrix = [Cineq  bounds_as_cols]
        buildInequalitiesWithBounds();
    }

    // 求解器接口
    QPResult solve(double tol = 1e-9, int maxIter = 1000)
    {
        // 1) 分解 G 用于无约束解快速计算; 按照论文采用 Cholesky 分解 G=LL^T
        LLT<MatrixXd> llt(G_);
        if (llt.info() != Success)
        {
            return {VectorXd(), false, "G is not positive definite (LLT failed)", 0};
        }

        // 初始有效集：等式索引（固定，不可删除）
        // 我们把所有约束（等式+不等式+bounds）在内部编号为：
        //  0..me-1                 => equality constraints
        //  me .. me + mI-1         => original inequality constraints
        //  me + mI .. me + mI + mB -1  => bound constraints appended
        active.clear();
        for (int i = 0; i < me; ++i)
            active.push_back(i);

        // 初始解：在等式约束下求最优点（如果 me == 0 则无约束解）
        optional<pair<VectorXd, VectorXd>> kkt0 = solveKKTForActiveSet(active);
        if (!kkt0.has_value())
        {
            return {VectorXd(),
                    false,
                    "Initial KKT (with equalities) failed: equalities may be inconsistent or dependent",
                    0};
        }
        VectorXd x = kkt0->first;
        VectorXd u;  // will hold multipliers for current active set
        if (active.size() > 0)
            u = kkt0->second;
        else
            u = VectorXd();

        /// TODO: builds B,Q,R,J for current active set; recomputeFactorsForActive(active);

        int iter = 0;
        while (iter < maxIter)
        {
            ++iter;
            // 计算所有不在 active 中的不等式 slack = c^T x + b
            int totalIneq = mI_total;  // number of inequality constraints (original + bounds)
            double smin = numeric_limits<double>::infinity();
            int p_global = -1;  // global index into all constraints (0..me+totalIneq-1)
            // iterate over all inequality indices (their global index offset = me + j)
            for (int j = 0; j < totalIneq; ++j)
            {
                int glob = me + j;
                // skip if in active
                if (isActive(glob))
                    continue;
                double s = C_all.col(j).dot(x) + b_all(j);  // C_all columns correspond to inequality-only set
                if (s < smin)
                {
                    smin = s;
                    p_global = glob;
                }
            }

            // 全部不等式都满足
            if (p_global == -1 || smin >= -tol)
            {
                return {x, true, "Optimal solution found", iter};
            }

            // 尝试将 p_global 加入 active
            vector<int> new_active = active;
            new_active.push_back(p_global);

            /// TODO: 按论文维护 active set
            /// 每次试探约束（加入）时，计算方向 z 和对偶方向 r，采用 J（= L^{-T}Q ）和 R； 通过一次 L 的逆乘与 QR
            /// 得到，即可不用解一个n+q 维的 KKT 线性系统。因此求方向成本为 O（n^2） 或 O(nq) 而不是  O( (n+q)^3 )。

            auto kkt_try = solveKKTForActiveSet(new_active);
            if (!kkt_try.has_value())
            {
                // KKT 不可解：可能说明新列与现有列线性相关或数值病态
                // 我们按简单策略：拒绝加入 p（等于认为它线性相关），并继续（在下一轮可能选其他约束）
                // 若你希望更激进：可以尝试正则化或做列简化等。
                // 这里返回提示但继续循环（如果MaxIter 过大，避免无效死循环需要break; this constraint permanently）
                // 为稳妥我们将把该约束临时标记为 "rejected" to avoid infinite loop.
                rejectedSet.push_back(p_global);
                // if all constraints rejected -> infeasible
                if (rejectedSet.size() >= (size_t)totalIneq)
                {
                    return {
                        x,
                        false,
                        "All violated constraints rejected due to KKT failure; problem may be infeasible or degenerate",
                        iter};
                }
                continue;
            }

            VectorXd x_trial = kkt_try->first;
            VectorXd u_trial = kkt_try->second;  // multipliers aligned with new_active ordering (equalities first)

            // check inequality multipliers (do not consider equalities for deletion)
            int q = (int)new_active.size();
            int q_eq = me;
            bool allIneqNonNeg = true;
            int mostNegIdxInNew = -1;
            double mostNegVal = 0.0;

            for (int i = q_eq; i < q; ++i)
            {
                double ui = u_trial(i);
                if (ui < -tol)
                {
                    if (mostNegIdxInNew == -1 || ui < mostNegVal)
                    {
                        mostNegIdxInNew = i;
                        mostNegVal = ui;
                    }
                    allIneqNonNeg = false;
                }
            }

            if (allIneqNonNeg)
            {
                // accept this active set
                active = new_active;
                x = x_trial;
                u = u_trial;
                // clear rejected set (a successful change could make previously rejected feasible)
                rejectedSet.clear();
                continue;
            }
            else
            {
                // need to delete the most negative inequality multiplier (but not equalities)
                int idxToRemoveInNew = mostNegIdxInNew;  // index in new_active vector
                int globalIdxToRemove = new_active[idxToRemoveInNew];
                // delete it from active
                removeIndexFromActive(active, globalIdxToRemove);
                // recompute KKT for reduced active set
                if (active.empty())
                {
                    x = llt.solve(-a_);  // no eq, no active inequalities
                    u = VectorXd();
                }
                else
                {
                    auto kkt2 = solveKKTForActiveSet(active);
                    if (!kkt2.has_value())
                    {
                        return {x, false, "KKT solve failed after deletion (numerical issues)", iter};
                    }
                    x = kkt2->first;
                    u = kkt2->second;
                }
                // after deletion, we continue main loop (do NOT immediately reattempt same p in same iteration)
                continue;
            }
        }

        return {VectorXd(), false, "Max iterations reached", iter};
    }

  private:
    MatrixXd G_;
    VectorXd a_;
    MatrixXd Ceq_;
    VectorXd Beq_;
    MatrixXd Cineq_;
    VectorXd Bineq_;
    VectorXd xl_, xu_;
    int n;
    int me;  // number of equalities
    int mi;  // original number of inequalities (excluding bounds)

    // combined inequality matrix (original Cineq plus bounds columns)
    MatrixXd C_all;  // n x mI_total
    VectorXd b_all;  // mI_total
    int mI_total;    // total inequality count after adding bounds

    vector<int> active;       // global indices of active constraints (equalities are 0..me-1)
    vector<int> rejectedSet;  // temporarily rejected constraints due to KKT failures
    // helper: flag to indicate a global index is active
    bool isActive(int globalIdx) const { return find(active.begin(), active.end(), globalIdx) != active.end(); }

    // 接口友好的提供完整的约束形式，所以这里统一处理一下 Build C_all and b_all by appending bounds as inequalities:
    // For lower bound x_i >= xl_i  -> e_i^T x - xl_i >= 0  (c = e_i, b = -xl_i)
    // For upper bound x_i <= xu_i  -> -e_i^T x + xu_i >= 0 (c = -e_i, b = xu_i)
    void buildInequalitiesWithBounds()
    {
        // compute number of bounds constraints (only include those with finite bounds)
        vector<pair<VectorXd, double>> boundCols;  // pair (col vector, b)
        // lower bounds
        for (int i = 0; i < n; ++i)
        {
            if (std::isfinite(xl_(i)))
            {
                VectorXd col = VectorXd::Zero(n);
                col(i) = 1.0;
                double b = -xl_(i);  // e_i^T x - xl >=0  => c=e_i, b=-xl
                boundCols.emplace_back(col, b);
            }
        }
        // upper bounds
        for (int i = 0; i < n; ++i)
        {
            if (std::isfinite(xu_(i)))
            {
                VectorXd col = VectorXd::Zero(n);
                col(i) = -1.0;
                double b = xu_(i);  // -e_i^T x + xu >=0 => c=-e_i, b = xu
                boundCols.emplace_back(col, b);
            }
        }

        // original inequality columns
        vector<pair<VectorXd, double>> allCols;
        for (int j = 0; j < Cineq_.cols(); ++j)
        {
            allCols.emplace_back(Cineq_.col(j), Bineq_(j));
        }
        // append bounds
        for (auto& p : boundCols)
            allCols.push_back(p);

        mI_total = (int)allCols.size();
        if (mI_total == 0)
        {
            C_all = MatrixXd(n, 0);
            b_all = VectorXd(0);
        }
        else
        {
            C_all = MatrixXd(n, mI_total);
            b_all = VectorXd(mI_total);
            for (int j = 0; j < mI_total; ++j)
            {
                C_all.col(j) = allCols[j].first;
                b_all(j) = allCols[j].second;
            }
        }
    }

    static void removeIndexFromActive(vector<int>& activeVec, int idxToRemove)
    {
        auto it = std::find(activeVec.begin(), activeVec.end(), idxToRemove);
        if (it != activeVec.end())
            activeVec.erase(it);
    }

    // 给定 active（global indices）构造并解 KKT:
    // 用我们约定的符号：约束格式为 c^T x + b >= 0 或 = 0
    // N 是 n x q 矩阵，按 global order 把相应列提取：
    // KKT 为:
    // [ G   -N ] [ x ] = [ -a ]
    // [ N^T  0 ] [ u ]   [ -bA ]
    // 返回 optional<pair<x,u>> ，u 与 active 的顺序一一对应（前 me 个是 equalities 对应）
    optional<pair<VectorXd, VectorXd>> solveKKTForActiveSet(const vector<int>& activeSet)
    {

        int q = (int)activeSet.size();
        // special case: q == 0 -> just solve G x = -a
        if (q == 0)
        {
            // 按照论文采用 Cholesky 分解 G=LL^T
            LLT<MatrixXd> llt(G_);
            if (llt.info() != Success)
                return nullopt;
            VectorXd x0 = llt.solve(-a_);
            VectorXd u0(0);
            return make_pair(x0, u0);
        }

        // build N and bA according to activeSet
        /// TODO: MatrixXd N = buildNforActive(act);
        MatrixXd N(n, q);
        VectorXd bA(q);
        for (int i = 0; i < q; ++i)
        {
            int gidx = activeSet[i];
            if (gidx < me)
            {
                // equality from Ceq, Beq (index gidx)
                N.col(i) = Ceq_.col(gidx);
                bA(i) = Beq_(gidx);
            }
            else
            {
                // inequality or bound: global index me + j
                int j = gidx - me;
                if (j < 0 || j >= mI_total)
                {
                    // should not happen
                    return nullopt;
                }
                N.col(i) = C_all.col(j);
                bA(i) = b_all(j);
            }
        }

        // assemble KKT: size (n+q)
        MatrixXd K(n + q, n + q);
        K.setZero();
        K.topLeftCorner(n, n) = G_;
        K.topRightCorner(n, q) = -N;
        K.bottomLeftCorner(q, n) = N.transpose();
        // bottom-right is zeros

        VectorXd rhs(n + q);
        rhs.head(n) = -a_;
        rhs.tail(q) = -bA;

        /// TODO: Solve (symmetric indefinite). Use FullPivLU to be robust.
        // For performance you might prefer LDLT with pivoting specialized KKT solver.
        // 避免显示计算 H <---  -G^{-1} a
        // 对比论文维护矩阵的分解更新， N*
        // QR分解： L^{-1}N = Q * [R, 0]^T
        // R J 基于 Givens 变化直接更新， J = L^{-T} Q , 用于代替直接计算完整的Q
        FullPivLU<MatrixXd> lu(K);
        if (!lu.isInvertible())
        {
            return nullopt;
        }
        VectorXd sol = lu.solve(rhs);
        VectorXd x = sol.head(n);
        VectorXd u = sol.tail(q);
        return make_pair(x, u);
    }
};

#endif // QP_SOLVER_PROJECTED_DUAL_QP_SOLVER_H
