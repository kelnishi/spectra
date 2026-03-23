#include <emscripten/bind.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/GenEigsComplexShiftSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/DavidsonSymEigsSolver.h>
#include <Spectra/contrib/PartialSVDSolver.h>
#include <Spectra/contrib/LOBPCGSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>
#include <Spectra/MatOp/DenseGenRealShiftSolve.h>
#include <Spectra/MatOp/DenseGenComplexShiftSolve.h>
#include <Spectra/MatOp/DenseCholesky.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>
#include <Spectra/MatOp/SparseGenComplexShiftSolve.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/MatOp/SparseRegularInverse.h>
#include <Spectra/MatOp/SymShiftInvert.h>

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

using namespace emscripten;
using namespace Spectra;

// ============================================================
// Data conversion helpers
// ============================================================

static Eigen::MatrixXd jsArrayToMatrix(const val& data, int rows, int cols)
{
    std::vector<double> vec = convertJSArrayToNumberVector<double>(data);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(vec.data(), rows, cols);
    return mat;
}

static val eigenVecToJS(const Eigen::VectorXd& v)
{
    val arr = val::array();
    for (Eigen::Index i = 0; i < v.size(); i++)
        arr.call<void>("push", v(i));
    return arr;
}

// Zero-copy TypedArray output — bulk memcpy via typed_memory_view + slice()
static val eigenVecToFloat64Array(const Eigen::VectorXd& v)
{
    return val(typed_memory_view(v.size(), v.data())).call<val>("slice");
}

static val eigenMatToFloat64Array(const Eigen::MatrixXd& m)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajor(m);
    val result = val::object();
    result.set("data", val(typed_memory_view(rowMajor.size(), rowMajor.data())).call<val>("slice"));
    result.set("rows", (int)m.rows());
    result.set("cols", (int)m.cols());
    return result;
}

static val eigenComplexVecToFloat64Array(const Eigen::VectorXcd& v)
{
    // Interleaved [re0, im0, re1, im1, ...] as Float64Array
    val result = val::object();
    std::vector<double> interleaved(v.size() * 2);
    for (Eigen::Index i = 0; i < v.size(); i++)
    {
        interleaved[2 * i] = v(i).real();
        interleaved[2 * i + 1] = v(i).imag();
    }
    result.set("data", val(typed_memory_view(interleaved.size(), interleaved.data())).call<val>("slice"));
    result.set("length", (int)v.size());
    return result;
}

static val eigenComplexMatToFloat64Array(const Eigen::MatrixXcd& m)
{
    // Interleaved [re, im, re, im, ...] in row-major order
    std::vector<double> interleaved(m.rows() * m.cols() * 2);
    int k = 0;
    for (Eigen::Index i = 0; i < m.rows(); i++)
    {
        for (Eigen::Index j = 0; j < m.cols(); j++)
        {
            interleaved[k++] = m(i, j).real();
            interleaved[k++] = m(i, j).imag();
        }
    }
    val result = val::object();
    result.set("data", val(typed_memory_view(interleaved.size(), interleaved.data())).call<val>("slice"));
    result.set("rows", (int)m.rows());
    result.set("cols", (int)m.cols());
    return result;
}

static val eigenMatToJS(const Eigen::MatrixXd& m)
{
    val result = val::object();
    val data = val::array();
    for (Eigen::Index i = 0; i < m.rows(); i++)
        for (Eigen::Index j = 0; j < m.cols(); j++)
            data.call<void>("push", m(i, j));
    result.set("data", data);
    result.set("rows", (int)m.rows());
    result.set("cols", (int)m.cols());
    return result;
}

static val eigenComplexVecToJS(const Eigen::VectorXcd& v)
{
    val arr = val::array();
    for (Eigen::Index i = 0; i < v.size(); i++)
    {
        val c = val::object();
        c.set("re", v(i).real());
        c.set("im", v(i).imag());
        arr.call<void>("push", c);
    }
    return arr;
}

static val eigenComplexMatToJS(const Eigen::MatrixXcd& m)
{
    val result = val::object();
    val data = val::array();
    for (Eigen::Index i = 0; i < m.rows(); i++)
    {
        for (Eigen::Index j = 0; j < m.cols(); j++)
        {
            val c = val::object();
            c.set("re", m(i, j).real());
            c.set("im", m(i, j).imag());
            data.call<void>("push", c);
        }
    }
    result.set("data", data);
    result.set("rows", (int)m.rows());
    result.set("cols", (int)m.cols());
    return result;
}

static Eigen::SparseMatrix<double> jsTripletsToSparse(const val& triplets, int rows, int cols)
{
    unsigned len = triplets["length"].as<unsigned>();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(len);
    for (unsigned i = 0; i < len; i++)
    {
        val t = triplets[i];
        trips.emplace_back(t["row"].as<int>(), t["col"].as<int>(), t["val"].as<double>());
    }
    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.setFromTriplets(trips.begin(), trips.end());
    return mat;
}

// Zero-copy CSR input — accepts raw WASM heap pointers from TypedArrays.
// JS side allocates in WASM heap via _malloc, copies TypedArray data once,
// passes pointers here, then _free's after the call.
static Eigen::SparseMatrix<double> ptrToSparseCSR(uintptr_t rowOffsetsPtr,
                                                    uintptr_t colIndicesPtr,
                                                    uintptr_t valuesPtr,
                                                    int rows, int cols, int nnz)
{
    const int* outerIndex = reinterpret_cast<const int*>(rowOffsetsPtr);
    const int* innerIndices = reinterpret_cast<const int*>(colIndicesPtr);
    const double* values = reinterpret_cast<const double*>(valuesPtr);

    Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int>>
        mapped(rows, cols, nnz, outerIndex, innerIndices, values);
    return Eigen::SparseMatrix<double>(mapped);
}

// Zero-copy dense matrix input from WASM heap pointer (row-major Float64Array)
static Eigen::MatrixXd ptrToMatrix(uintptr_t dataPtr, int rows, int cols)
{
    const double* data = reinterpret_cast<const double*>(dataPtr);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mapped(data, rows, cols);
    return Eigen::MatrixXd(mapped);
}

// Build a sparse initial block vector for LOBPCG with guaranteed full rank.
// Each column has exactly one non-zero entry (value 1.0) at a strided row,
// so the block vector is always rank-nev regardless of matrix size.
// This avoids the sparseView() pitfall where near-zero random values get
// dropped, leaving a rank-deficient X that causes OOB inside the solver.
static Eigen::SparseMatrix<double> makeLobpcgInitial(int n, int nev)
{
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nev);
    for (int j = 0; j < nev; j++)
    {
        // Spread pivots evenly across rows to avoid clustering
        int row = (int)(((long long)j * n) / nev);
        trips.emplace_back(row, j, 1.0);
    }
    Eigen::SparseMatrix<double> X(n, nev);
    X.setFromTriplets(trips.begin(), trips.end());
    return X;
}

static SortRule parseSortRule(const std::string& rule)
{
    if (rule == "LargestMagn") return SortRule::LargestMagn;
    if (rule == "LargestReal") return SortRule::LargestReal;
    if (rule == "LargestImag") return SortRule::LargestImag;
    if (rule == "LargestAlge") return SortRule::LargestAlge;
    if (rule == "SmallestMagn") return SortRule::SmallestMagn;
    if (rule == "SmallestReal") return SortRule::SmallestReal;
    if (rule == "SmallestImag") return SortRule::SmallestImag;
    if (rule == "SmallestAlge") return SortRule::SmallestAlge;
    if (rule == "BothEnds") return SortRule::BothEnds;
    return SortRule::LargestMagn;
}

static std::string compInfoToString(CompInfo info)
{
    switch (info)
    {
        case CompInfo::Successful: return "Successful";
        case CompInfo::NotComputed: return "NotComputed";
        case CompInfo::NotConverging: return "NotConverging";
        case CompInfo::NumericalIssue: return "NumericalIssue";
        default: return "Unknown";
    }
}

// ============================================================
// Progress reporting
// ============================================================

// Global progress state — safe in WASM's single-threaded model.
// Set before calling compute(), cleared after.
struct ProgressState
{
    val callback = val::undefined();
    int opsCompleted = 0;
    int estimatedTotalOps = 0;
    int reportInterval = 1;

    void reset(int estTotal, int interval)
    {
        opsCompleted = 0;
        estimatedTotalOps = estTotal;
        reportInterval = (std::max)(1, interval);
    }

    void tick()
    {
        opsCompleted++;
        if (callback.isUndefined())
            return;
        if (opsCompleted % reportInterval == 0)
        {
            double pct = estimatedTotalOps > 0
                ? (std::min)(1.0, (double)opsCompleted / estimatedTotalOps)
                : 0.0;
            val info = val::object();
            info.set("opsCompleted", opsCompleted);
            info.set("estimatedTotalOps", estimatedTotalOps);
            info.set("progress", pct);
            callback(info);
        }
    }
};

static ProgressState g_progress;

// MatOp proxy that intercepts perform_op to report progress.
// Unused member functions (set_shift, operator*, etc.) are only instantiated
// if actually called by the solver, so this single template works for all ops.
template <typename InnerOp>
class ProgressOp
{
public:
    using Scalar = typename InnerOp::Scalar;

private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    InnerOp& m_inner;

public:
    ProgressOp(InnerOp& inner) : m_inner(inner) {}

    Index rows() const { return m_inner.rows(); }
    Index cols() const { return m_inner.cols(); }

    // Core mat-vec product — this is where the work happens
    void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        m_inner.perform_op(x_in, y_out);
        g_progress.tick();
    }

    // Shift-invert solvers (single real shift)
    void set_shift(const Scalar& sigma)
    {
        m_inner.set_shift(sigma);
    }

    // Complex shift-invert solvers (real + imaginary parts)
    void set_shift(const Scalar& sigmar, const Scalar& sigmai)
    {
        m_inner.set_shift(sigmar, sigmai);
    }

    // Davidson solver uses operator* for matrix-matrix products
    Matrix operator*(const Eigen::Ref<const Matrix>& mat_in) const
    {
        // Count one op per column (each column is one mat-vec)
        for (Index i = 0; i < mat_in.cols(); i++)
            g_progress.tick();
        return m_inner * mat_in;
    }

    // Davidson solver reads diagonal via operator()
    Scalar operator()(Index i, Index j) const
    {
        return m_inner(i, j);
    }
};

// RAII helper to set up and tear down progress state around a compute() call.
// estimatedTotalOps = ncv * maxIter for Arnoldi/Lanczos solvers.
class ProgressScope
{
public:
    ProgressScope(const val& cb, int estTotal, int ncv)
    {
        g_progress.callback = cb;
        // Report roughly once per iteration (every ncv ops), minimum 1
        g_progress.reset(estTotal, (std::max)(1, ncv));
    }

    ~ProgressScope()
    {
        g_progress.callback = val::undefined();
    }
};

// ============================================================
// Result builders
// ============================================================

template <typename SolverType>
static val makeRealResult(SolverType& solver)
{
    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", (int)solver.num_iterations());
    result.set("numOps", g_progress.opsCompleted);
    if (solver.info() == CompInfo::Successful)
    {
        result.set("eigenvalues", eigenVecToJS(solver.eigenvalues()));
        result.set("eigenvectors", eigenMatToJS(solver.eigenvectors()));
    }
    return result;
}

template <typename SolverType>
static val makeComplexResult(SolverType& solver)
{
    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", (int)solver.num_iterations());
    result.set("numOps", g_progress.opsCompleted);
    if (solver.info() == CompInfo::Successful)
    {
        result.set("eigenvalues", eigenComplexVecToJS(solver.eigenvalues()));
        result.set("eigenvectors", eigenComplexMatToJS(solver.eigenvectors()));
    }
    return result;
}

// TypedArray result builders — used by CSR/pointer-based functions
template <typename SolverType>
static val makeRealResultTyped(SolverType& solver)
{
    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", (int)solver.num_iterations());
    result.set("numOps", g_progress.opsCompleted);
    if (solver.info() == CompInfo::Successful)
    {
        result.set("eigenvalues", eigenVecToFloat64Array(solver.eigenvalues()));
        result.set("eigenvectors", eigenMatToFloat64Array(solver.eigenvectors()));
    }
    return result;
}

template <typename SolverType>
static val makeComplexResultTyped(SolverType& solver)
{
    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", (int)solver.num_iterations());
    result.set("numOps", g_progress.opsCompleted);
    if (solver.info() == CompInfo::Successful)
    {
        result.set("eigenvalues", eigenComplexVecToFloat64Array(solver.eigenvalues()));
        result.set("eigenvectors", eigenComplexMatToFloat64Array(solver.eigenvectors()));
    }
    return result;
}

// ============================================================
// Exported: estimate total operations for a solver configuration
// ============================================================
static val estimateOps(int ncv, int maxIter)
{
    // Each Arnoldi/Lanczos restart iteration does ~ncv matrix-vector products.
    // The initial factorization also does ncv ops.
    // Total estimate: ncv * (maxIter + 1)
    val result = val::object();
    result.set("opsPerIteration", ncv);
    result.set("estimatedTotalOps", ncv * (maxIter + 1));
    return result;
}

// ============================================================
// 1. Dense symmetric eigenvalue solver
// ============================================================
static val symEigs(const val& matData, int n, int nev, int ncv,
                   const std::string& rule, int maxIter, double tol,
                   val cb)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseSymMatProd<double> inner(mat);
    ProgressOp<DenseSymMatProd<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymEigsSolver<ProgressOp<DenseSymMatProd<double>>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 2. Dense symmetric eigenvalue solver with shift-and-invert
// ============================================================
static val symEigsShift(const val& matData, int n, int nev, int ncv,
                        double sigma, const std::string& rule,
                        int maxIter, double tol, val cb)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseSymShiftSolve<double> inner(mat);
    ProgressOp<DenseSymShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymEigsShiftSolver<ProgressOp<DenseSymShiftSolve<double>>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 3. Dense general eigenvalue solver
// ============================================================
static val genEigs(const val& matData, int n, int nev, int ncv,
                   const std::string& rule, int maxIter, double tol,
                   val cb)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseGenMatProd<double> inner(mat);
    ProgressOp<DenseGenMatProd<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsSolver<ProgressOp<DenseGenMatProd<double>>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 4. Dense general eigenvalue solver with real shift
// ============================================================
static val genEigsRealShift(const val& matData, int n, int nev, int ncv,
                            double sigma, const std::string& rule,
                            int maxIter, double tol, val cb)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseGenRealShiftSolve<double> inner(mat);
    ProgressOp<DenseGenRealShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsRealShiftSolver<ProgressOp<DenseGenRealShiftSolve<double>>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 5. Dense general eigenvalue solver with complex shift
// ============================================================
static val genEigsComplexShift(const val& matData, int n, int nev, int ncv,
                               double sigmaR, double sigmaI,
                               const std::string& rule, int maxIter, double tol,
                               val cb)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseGenComplexShiftSolve<double> inner(mat);
    ProgressOp<DenseGenComplexShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsComplexShiftSolver<ProgressOp<DenseGenComplexShiftSolve<double>>> solver(op, nev, ncv, sigmaR, sigmaI);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 6. Dense generalized symmetric eigensolver (Cholesky mode)
// ============================================================
static val symGEigsCholesky(const val& matA, const val& matB, int n,
                            int nev, int ncv, const std::string& rule,
                            int maxIter, double tol, val cb)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    DenseSymMatProd<double> innerA(A);
    ProgressOp<DenseSymMatProd<double>> opA(innerA);
    DenseCholesky<double> opB(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsSolver<ProgressOp<DenseSymMatProd<double>>, DenseCholesky<double>, GEigsMode::Cholesky> solver(opA, opB, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 7. Dense generalized symmetric eigensolver (shift-invert)
// ============================================================
static val symGEigsShiftInvert(const val& matA, const val& matB, int n,
                               int nev, int ncv, double sigma,
                               const std::string& rule, int maxIter, double tol,
                               val cb)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    using InnerOpType = SymShiftInvert<double, Eigen::Dense, Eigen::Dense>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    DenseSymMatProd<double> Bop(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, DenseSymMatProd<double>, GEigsMode::ShiftInvert> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 8. Dense generalized symmetric eigensolver (buckling mode)
// ============================================================
static val symGEigsBuckling(const val& matA, const val& matB, int n,
                            int nev, int ncv, double sigma,
                            const std::string& rule, int maxIter, double tol,
                            val cb)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    using InnerOpType = SymShiftInvert<double, Eigen::Dense, Eigen::Dense>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    DenseSymMatProd<double> Bop(A);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, DenseSymMatProd<double>, GEigsMode::Buckling> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 9. Dense generalized symmetric eigensolver (Cayley mode)
// ============================================================
static val symGEigsCayley(const val& matA, const val& matB, int n,
                          int nev, int ncv, double sigma,
                          const std::string& rule, int maxIter, double tol,
                          val cb)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    using InnerOpType = SymShiftInvert<double, Eigen::Dense, Eigen::Dense>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    DenseSymMatProd<double> Bop(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, DenseSymMatProd<double>, GEigsMode::Cayley> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 10. Davidson symmetric eigensolver (dense)
// ============================================================
static val davidsonSymEigs(const val& matData, int n, int nev,
                           const std::string& rule, int maxIter, double tol,
                           val cb)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseSymMatProd<double> inner(mat);
    ProgressOp<DenseSymMatProd<double>> op(inner);
    // Davidson does ~nev mat-vec products per iteration (correction vectors)
    ProgressScope scope(cb, nev * (maxIter + 1), nev);
    DavidsonSymEigsSolver<ProgressOp<DenseSymMatProd<double>>> solver(op, nev);
    solver.compute(parseSortRule(rule), maxIter, tol);

    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", (int)solver.num_iterations());
    result.set("numOps", g_progress.opsCompleted);
    if (solver.info() == CompInfo::Successful)
    {
        result.set("eigenvalues", eigenVecToJS(solver.eigenvalues()));
        result.set("eigenvectors", eigenMatToJS(solver.eigenvectors()));
    }
    return result;
}

// ============================================================
// 11. Partial SVD (dense)
// ============================================================
static val partialSVD(const val& matData, int rows, int cols,
                      int ncomp, int ncv, int maxIter, double tol)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, rows, cols);
    PartialSVDSolver<Eigen::MatrixXd> solver(mat, ncomp, ncv);
    solver.compute(maxIter, tol);

    val result = val::object();
    result.set("singularValues", eigenVecToJS(solver.singular_values()));
    result.set("matrixU", eigenMatToJS(solver.matrix_U(ncomp)));
    result.set("matrixV", eigenMatToJS(solver.matrix_V(ncomp)));
    return result;
}

// ============================================================
// 12. Sparse symmetric eigenvalue solver
// ============================================================
static val sparseSymEigs(const val& triplets, int rows, int cols,
                         int nev, int ncv, const std::string& rule,
                         int maxIter, double tol, val cb)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseSymMatProd<double> inner(mat);
    ProgressOp<SparseSymMatProd<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymEigsSolver<ProgressOp<SparseSymMatProd<double>>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 13. Sparse symmetric eigenvalue solver with shift-and-invert
// ============================================================
static val sparseSymEigsShift(const val& triplets, int rows, int cols,
                              int nev, int ncv, double sigma,
                              const std::string& rule, int maxIter, double tol,
                              val cb)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseSymShiftSolve<double> inner(mat);
    ProgressOp<SparseSymShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymEigsShiftSolver<ProgressOp<SparseSymShiftSolve<double>>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 14. Sparse general eigenvalue solver
// ============================================================
static val sparseGenEigs(const val& triplets, int rows, int cols,
                         int nev, int ncv, const std::string& rule,
                         int maxIter, double tol, val cb)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseGenMatProd<double> inner(mat);
    ProgressOp<SparseGenMatProd<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsSolver<ProgressOp<SparseGenMatProd<double>>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 15. Sparse general eigenvalue solver with real shift
// ============================================================
static val sparseGenEigsRealShift(const val& triplets, int rows, int cols,
                                  int nev, int ncv, double sigma,
                                  const std::string& rule, int maxIter, double tol,
                                  val cb)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseGenRealShiftSolve<double> inner(mat);
    ProgressOp<SparseGenRealShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsRealShiftSolver<ProgressOp<SparseGenRealShiftSolve<double>>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 16. Sparse general eigenvalue solver with complex shift
// ============================================================
static val sparseGenEigsComplexShift(const val& triplets, int rows, int cols,
                                     int nev, int ncv,
                                     double sigmaR, double sigmaI,
                                     const std::string& rule, int maxIter, double tol,
                                     val cb)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseGenComplexShiftSolve<double> inner(mat);
    ProgressOp<SparseGenComplexShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsComplexShiftSolver<ProgressOp<SparseGenComplexShiftSolve<double>>> solver(op, nev, ncv, sigmaR, sigmaI);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 17. Sparse generalized symmetric eigensolver (Cholesky mode)
// ============================================================
static val sparseSymGEigsCholesky(const val& tripsA, const val& tripsB,
                                  int rows, int cols,
                                  int nev, int ncv, const std::string& rule,
                                  int maxIter, double tol, val cb)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    SparseSymMatProd<double> innerA(A);
    ProgressOp<SparseSymMatProd<double>> opA(innerA);
    SparseCholesky<double> opB(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsSolver<ProgressOp<SparseSymMatProd<double>>, SparseCholesky<double>, GEigsMode::Cholesky> solver(opA, opB, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 18. Sparse generalized symmetric eigensolver (regular inverse mode)
// ============================================================
static val sparseSymGEigsRegularInverse(const val& tripsA, const val& tripsB,
                                        int rows, int cols,
                                        int nev, int ncv, const std::string& rule,
                                        int maxIter, double tol, val cb)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    SparseSymMatProd<double> innerA(A);
    ProgressOp<SparseSymMatProd<double>> opA(innerA);
    SparseRegularInverse<double> opB(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsSolver<ProgressOp<SparseSymMatProd<double>>, SparseRegularInverse<double>, GEigsMode::RegularInverse> solver(opA, opB, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 19. Sparse generalized symmetric eigensolver (shift-invert)
// ============================================================
static val sparseSymGEigsShiftInvert(const val& tripsA, const val& tripsB,
                                     int rows, int cols,
                                     int nev, int ncv, double sigma,
                                     const std::string& rule, int maxIter, double tol,
                                     val cb)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    using InnerOpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    SparseSymMatProd<double> Bop(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, SparseSymMatProd<double>, GEigsMode::ShiftInvert> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 20. Sparse generalized symmetric eigensolver (buckling mode)
// ============================================================
static val sparseSymGEigsBuckling(const val& tripsA, const val& tripsB,
                                  int rows, int cols,
                                  int nev, int ncv, double sigma,
                                  const std::string& rule, int maxIter, double tol,
                                  val cb)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    using InnerOpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    SparseSymMatProd<double> Bop(A);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, SparseSymMatProd<double>, GEigsMode::Buckling> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 21. Sparse generalized symmetric eigensolver (Cayley mode)
// ============================================================
static val sparseSymGEigsCayley(const val& tripsA, const val& tripsB,
                                int rows, int cols,
                                int nev, int ncv, double sigma,
                                const std::string& rule, int maxIter, double tol,
                                val cb)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    using InnerOpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    SparseSymMatProd<double> Bop(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, SparseSymMatProd<double>, GEigsMode::Cayley> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 22. Partial SVD (sparse)
// ============================================================
static val sparsePartialSVD(const val& triplets, int rows, int cols,
                            int ncomp, int ncv, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    PartialSVDSolver<Eigen::SparseMatrix<double>> solver(mat, ncomp, ncv);
    solver.compute(maxIter, tol);

    val result = val::object();
    result.set("singularValues", eigenVecToJS(solver.singular_values()));
    result.set("matrixU", eigenMatToJS(solver.matrix_U(ncomp)));
    result.set("matrixV", eigenMatToJS(solver.matrix_V(ncomp)));
    return result;
}

// ============================================================
// 23. LOBPCG solver (sparse symmetric, finds smallest eigenvalues)
// ============================================================
static val lobpcg(const val& tripsA, int n, int nev,
                  int maxIter, double tol)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, n, n);
    Eigen::SparseMatrix<double> X = makeLobpcgInitial(n, nev);

    LOBPCGSolver<double> solver(A, X);
    solver.compute(maxIter, tol);

    val result = val::object();
    result.set("converged", solver.info() == 0);
    result.set("eigenvalues", eigenVecToJS(solver.eigenvalues()));
    result.set("eigenvectors", eigenMatToJS(solver.eigenvectors()));
    result.set("residuals", eigenMatToJS(solver.residuals()));
    return result;
}

// ============================================================
// 24. LOBPCG solver with B matrix (sparse generalized, Ax=λBx)
// ============================================================
static val lobpcgGeneralized(const val& tripsA, const val& tripsB,
                             int n, int nev, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, n, n);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, n, n);
    Eigen::SparseMatrix<double> X = makeLobpcgInitial(n, nev);

    LOBPCGSolver<double> solver(A, X);
    solver.setB(B);
    solver.compute(maxIter, tol);

    val result = val::object();
    result.set("converged", solver.info() == 0);
    result.set("eigenvalues", eigenVecToJS(solver.eigenvalues()));
    result.set("eigenvectors", eigenMatToJS(solver.eigenvectors()));
    result.set("residuals", eigenMatToJS(solver.residuals()));
    return result;
}

// ============================================================
// CSR pointer-based sparse solvers (zero-copy TypedArray input)
// All accept raw WASM heap pointers for rowOffsets (int*),
// colIndices (int*), and values (double*) in CSR format.
// ============================================================

// 12b. Sparse symmetric eigenvalue solver (CSR)
static val sparseSymEigsCSR(uintptr_t roPtr, uintptr_t ciPtr, uintptr_t vPtr,
                             int rows, int cols, int nnz,
                             int nev, int ncv, const std::string& rule,
                             int maxIter, double tol, val cb)
{
    auto mat = ptrToSparseCSR(roPtr, ciPtr, vPtr, rows, cols, nnz);
    SparseSymMatProd<double> inner(mat);
    ProgressOp<SparseSymMatProd<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymEigsSolver<ProgressOp<SparseSymMatProd<double>>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResultTyped(solver);
}

// 13b. Sparse symmetric eigenvalue solver with shift (CSR)
static val sparseSymEigsShiftCSR(uintptr_t roPtr, uintptr_t ciPtr, uintptr_t vPtr,
                                  int rows, int cols, int nnz,
                                  int nev, int ncv, double sigma,
                                  const std::string& rule, int maxIter, double tol,
                                  val cb)
{
    auto mat = ptrToSparseCSR(roPtr, ciPtr, vPtr, rows, cols, nnz);
    SparseSymShiftSolve<double> inner(mat);
    ProgressOp<SparseSymShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymEigsShiftSolver<ProgressOp<SparseSymShiftSolve<double>>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResultTyped(solver);
}

// 14b. Sparse general eigenvalue solver (CSR)
static val sparseGenEigsCSR(uintptr_t roPtr, uintptr_t ciPtr, uintptr_t vPtr,
                             int rows, int cols, int nnz,
                             int nev, int ncv, const std::string& rule,
                             int maxIter, double tol, val cb)
{
    auto mat = ptrToSparseCSR(roPtr, ciPtr, vPtr, rows, cols, nnz);
    SparseGenMatProd<double> inner(mat);
    ProgressOp<SparseGenMatProd<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsSolver<ProgressOp<SparseGenMatProd<double>>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResultTyped(solver);
}

// 15b. Sparse general eigenvalue solver with real shift (CSR)
static val sparseGenEigsRealShiftCSR(uintptr_t roPtr, uintptr_t ciPtr, uintptr_t vPtr,
                                      int rows, int cols, int nnz,
                                      int nev, int ncv, double sigma,
                                      const std::string& rule, int maxIter, double tol,
                                      val cb)
{
    auto mat = ptrToSparseCSR(roPtr, ciPtr, vPtr, rows, cols, nnz);
    SparseGenRealShiftSolve<double> inner(mat);
    ProgressOp<SparseGenRealShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsRealShiftSolver<ProgressOp<SparseGenRealShiftSolve<double>>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResultTyped(solver);
}

// 16b. Sparse general eigenvalue solver with complex shift (CSR)
static val sparseGenEigsComplexShiftCSR(uintptr_t roPtr, uintptr_t ciPtr, uintptr_t vPtr,
                                         int rows, int cols, int nnz,
                                         int nev, int ncv,
                                         double sigmaR, double sigmaI,
                                         const std::string& rule, int maxIter, double tol,
                                         val cb)
{
    auto mat = ptrToSparseCSR(roPtr, ciPtr, vPtr, rows, cols, nnz);
    SparseGenComplexShiftSolve<double> inner(mat);
    ProgressOp<SparseGenComplexShiftSolve<double>> op(inner);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    GenEigsComplexShiftSolver<ProgressOp<SparseGenComplexShiftSolve<double>>> solver(op, nev, ncv, sigmaR, sigmaI);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResultTyped(solver);
}

// 17b. Sparse generalized symmetric eigensolver — Cholesky (CSR)
static val sparseSymGEigsCholeskyCSR(uintptr_t roA, uintptr_t ciA, uintptr_t vA, int nnzA,
                                      uintptr_t roB, uintptr_t ciB, uintptr_t vB, int nnzB,
                                      int rows, int cols,
                                      int nev, int ncv, const std::string& rule,
                                      int maxIter, double tol, val cb)
{
    auto A = ptrToSparseCSR(roA, ciA, vA, rows, cols, nnzA);
    auto B = ptrToSparseCSR(roB, ciB, vB, rows, cols, nnzB);
    SparseSymMatProd<double> innerA(A);
    ProgressOp<SparseSymMatProd<double>> opA(innerA);
    SparseCholesky<double> opB(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsSolver<ProgressOp<SparseSymMatProd<double>>, SparseCholesky<double>, GEigsMode::Cholesky> solver(opA, opB, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResultTyped(solver);
}

// 18b. Sparse generalized symmetric eigensolver — regular inverse (CSR)
static val sparseSymGEigsRegularInverseCSR(uintptr_t roA, uintptr_t ciA, uintptr_t vA, int nnzA,
                                            uintptr_t roB, uintptr_t ciB, uintptr_t vB, int nnzB,
                                            int rows, int cols,
                                            int nev, int ncv, const std::string& rule,
                                            int maxIter, double tol, val cb)
{
    auto A = ptrToSparseCSR(roA, ciA, vA, rows, cols, nnzA);
    auto B = ptrToSparseCSR(roB, ciB, vB, rows, cols, nnzB);
    SparseSymMatProd<double> innerA(A);
    ProgressOp<SparseSymMatProd<double>> opA(innerA);
    SparseRegularInverse<double> opB(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsSolver<ProgressOp<SparseSymMatProd<double>>, SparseRegularInverse<double>, GEigsMode::RegularInverse> solver(opA, opB, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResultTyped(solver);
}

// 19b. Sparse generalized symmetric eigensolver — shift-invert (CSR)
static val sparseSymGEigsShiftInvertCSR(uintptr_t roA, uintptr_t ciA, uintptr_t vA, int nnzA,
                                         uintptr_t roB, uintptr_t ciB, uintptr_t vB, int nnzB,
                                         int rows, int cols,
                                         int nev, int ncv, double sigma,
                                         const std::string& rule, int maxIter, double tol,
                                         val cb)
{
    auto A = ptrToSparseCSR(roA, ciA, vA, rows, cols, nnzA);
    auto B = ptrToSparseCSR(roB, ciB, vB, rows, cols, nnzB);
    using InnerOpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    SparseSymMatProd<double> Bop(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, SparseSymMatProd<double>, GEigsMode::ShiftInvert> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResultTyped(solver);
}

// 20b. Sparse generalized symmetric eigensolver — buckling (CSR)
static val sparseSymGEigsBucklingCSR(uintptr_t roA, uintptr_t ciA, uintptr_t vA, int nnzA,
                                      uintptr_t roB, uintptr_t ciB, uintptr_t vB, int nnzB,
                                      int rows, int cols,
                                      int nev, int ncv, double sigma,
                                      const std::string& rule, int maxIter, double tol,
                                      val cb)
{
    auto A = ptrToSparseCSR(roA, ciA, vA, rows, cols, nnzA);
    auto B = ptrToSparseCSR(roB, ciB, vB, rows, cols, nnzB);
    using InnerOpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    SparseSymMatProd<double> Bop(A);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, SparseSymMatProd<double>, GEigsMode::Buckling> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResultTyped(solver);
}

// 21b. Sparse generalized symmetric eigensolver — Cayley (CSR)
static val sparseSymGEigsCayleyCSR(uintptr_t roA, uintptr_t ciA, uintptr_t vA, int nnzA,
                                    uintptr_t roB, uintptr_t ciB, uintptr_t vB, int nnzB,
                                    int rows, int cols,
                                    int nev, int ncv, double sigma,
                                    const std::string& rule, int maxIter, double tol,
                                    val cb)
{
    auto A = ptrToSparseCSR(roA, ciA, vA, rows, cols, nnzA);
    auto B = ptrToSparseCSR(roB, ciB, vB, rows, cols, nnzB);
    using InnerOpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    InnerOpType innerOp(A, B);
    ProgressOp<InnerOpType> op(innerOp);
    SparseSymMatProd<double> Bop(B);
    ProgressScope scope(cb, ncv * (maxIter + 1), ncv);
    SymGEigsShiftSolver<ProgressOp<InnerOpType>, SparseSymMatProd<double>, GEigsMode::Cayley> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResultTyped(solver);
}

// 22b. Partial SVD — sparse (CSR)
static val sparsePartialSVDCSR(uintptr_t roPtr, uintptr_t ciPtr, uintptr_t vPtr,
                                int rows, int cols, int nnz,
                                int ncomp, int ncv, int maxIter, double tol)
{
    auto mat = ptrToSparseCSR(roPtr, ciPtr, vPtr, rows, cols, nnz);
    PartialSVDSolver<Eigen::SparseMatrix<double>> solver(mat, ncomp, ncv);
    solver.compute(maxIter, tol);

    val result = val::object();
    result.set("singularValues", eigenVecToFloat64Array(solver.singular_values()));
    result.set("matrixU", eigenMatToFloat64Array(solver.matrix_U(ncomp)));
    result.set("matrixV", eigenMatToFloat64Array(solver.matrix_V(ncomp)));
    return result;
}

// 23b. LOBPCG solver (CSR)
static val lobpcgCSR(uintptr_t roPtr, uintptr_t ciPtr, uintptr_t vPtr,
                      int n, int nnz, int nev,
                      int maxIter, double tol)
{
    auto A = ptrToSparseCSR(roPtr, ciPtr, vPtr, n, n, nnz);
    Eigen::SparseMatrix<double> X = makeLobpcgInitial(n, nev);

    LOBPCGSolver<double> solver(A, X);
    solver.compute(maxIter, tol);

    val result = val::object();
    result.set("converged", solver.info() == 0);
    result.set("eigenvalues", eigenVecToFloat64Array(solver.eigenvalues()));
    result.set("eigenvectors", eigenMatToFloat64Array(solver.eigenvectors()));
    result.set("residuals", eigenMatToFloat64Array(solver.residuals()));
    return result;
}

// 24b. LOBPCG generalized solver (CSR)
static val lobpcgGeneralizedCSR(uintptr_t roA, uintptr_t ciA, uintptr_t vA, int nnzA,
                                 uintptr_t roB, uintptr_t ciB, uintptr_t vB, int nnzB,
                                 int n, int nev, int maxIter, double tol)
{
    auto A = ptrToSparseCSR(roA, ciA, vA, n, n, nnzA);
    auto B = ptrToSparseCSR(roB, ciB, vB, n, n, nnzB);
    Eigen::SparseMatrix<double> X = makeLobpcgInitial(n, nev);

    LOBPCGSolver<double> solver(A, X);
    solver.setB(B);
    solver.compute(maxIter, tol);

    val result = val::object();
    result.set("converged", solver.info() == 0);
    result.set("eigenvalues", eigenVecToFloat64Array(solver.eigenvalues()));
    result.set("eigenvectors", eigenMatToFloat64Array(solver.eigenvectors()));
    result.set("residuals", eigenMatToFloat64Array(solver.residuals()));
    return result;
}

// ============================================================
// Embind module
// ============================================================
EMSCRIPTEN_BINDINGS(spectra)
{
    // Utility
    function("estimateOps", &estimateOps);

    // Dense standard eigenvalue solvers
    function("symEigs", &symEigs);
    function("symEigsShift", &symEigsShift);
    function("genEigs", &genEigs);
    function("genEigsRealShift", &genEigsRealShift);
    function("genEigsComplexShift", &genEigsComplexShift);

    // Dense generalized symmetric eigenvalue solvers
    function("symGEigsCholesky", &symGEigsCholesky);
    function("symGEigsShiftInvert", &symGEigsShiftInvert);
    function("symGEigsBuckling", &symGEigsBuckling);
    function("symGEigsCayley", &symGEigsCayley);

    // Davidson solver
    function("davidsonSymEigs", &davidsonSymEigs);

    // Dense partial SVD
    function("partialSVD", &partialSVD);

    // Sparse standard eigenvalue solvers
    function("sparseSymEigs", &sparseSymEigs);
    function("sparseSymEigsShift", &sparseSymEigsShift);
    function("sparseGenEigs", &sparseGenEigs);
    function("sparseGenEigsRealShift", &sparseGenEigsRealShift);
    function("sparseGenEigsComplexShift", &sparseGenEigsComplexShift);

    // Sparse generalized symmetric eigenvalue solvers
    function("sparseSymGEigsCholesky", &sparseSymGEigsCholesky);
    function("sparseSymGEigsRegularInverse", &sparseSymGEigsRegularInverse);
    function("sparseSymGEigsShiftInvert", &sparseSymGEigsShiftInvert);
    function("sparseSymGEigsBuckling", &sparseSymGEigsBuckling);
    function("sparseSymGEigsCayley", &sparseSymGEigsCayley);

    // Sparse partial SVD
    function("sparsePartialSVD", &sparsePartialSVD);

    // LOBPCG solvers
    function("lobpcg", &lobpcg);
    function("lobpcgGeneralized", &lobpcgGeneralized);

    // CSR pointer-based sparse solvers (zero-copy TypedArray input)
    function("sparseSymEigsCSR", &sparseSymEigsCSR);
    function("sparseSymEigsShiftCSR", &sparseSymEigsShiftCSR);
    function("sparseGenEigsCSR", &sparseGenEigsCSR);
    function("sparseGenEigsRealShiftCSR", &sparseGenEigsRealShiftCSR);
    function("sparseGenEigsComplexShiftCSR", &sparseGenEigsComplexShiftCSR);
    function("sparseSymGEigsCholeskyCSR", &sparseSymGEigsCholeskyCSR);
    function("sparseSymGEigsRegularInverseCSR", &sparseSymGEigsRegularInverseCSR);
    function("sparseSymGEigsShiftInvertCSR", &sparseSymGEigsShiftInvertCSR);
    function("sparseSymGEigsBucklingCSR", &sparseSymGEigsBucklingCSR);
    function("sparseSymGEigsCayleyCSR", &sparseSymGEigsCayleyCSR);
    function("sparsePartialSVDCSR", &sparsePartialSVDCSR);
    function("lobpcgCSR", &lobpcgCSR);
    function("lobpcgGeneralizedCSR", &lobpcgGeneralizedCSR);
}
