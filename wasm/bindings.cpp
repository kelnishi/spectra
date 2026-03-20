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

using namespace emscripten;
using namespace Spectra;

// ============================================================
// Helpers
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

// Build a standard result object for real-eigenvalue solvers
template <typename SolverType>
static val makeRealResult(SolverType& solver)
{
    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", solver.num_iterations());
    if (solver.info() == CompInfo::Successful)
    {
        result.set("eigenvalues", eigenVecToJS(solver.eigenvalues()));
        result.set("eigenvectors", eigenMatToJS(solver.eigenvectors()));
    }
    return result;
}

// Build a standard result object for complex-eigenvalue solvers
template <typename SolverType>
static val makeComplexResult(SolverType& solver)
{
    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", solver.num_iterations());
    if (solver.info() == CompInfo::Successful)
    {
        result.set("eigenvalues", eigenComplexVecToJS(solver.eigenvalues()));
        result.set("eigenvectors", eigenComplexMatToJS(solver.eigenvectors()));
    }
    return result;
}

// ============================================================
// 1. Dense symmetric eigenvalue solver
// ============================================================
static val symEigs(const val& matData, int n, int nev, int ncv,
                   const std::string& rule, int maxIter, double tol)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseSymMatProd<double> op(mat);
    SymEigsSolver<DenseSymMatProd<double>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 2. Dense symmetric eigenvalue solver with shift-and-invert
// ============================================================
static val symEigsShift(const val& matData, int n, int nev, int ncv,
                        double sigma, const std::string& rule,
                        int maxIter, double tol)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseSymShiftSolve<double> op(mat);
    SymEigsShiftSolver<DenseSymShiftSolve<double>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 3. Dense general eigenvalue solver
// ============================================================
static val genEigs(const val& matData, int n, int nev, int ncv,
                   const std::string& rule, int maxIter, double tol)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseGenMatProd<double> op(mat);
    GenEigsSolver<DenseGenMatProd<double>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 4. Dense general eigenvalue solver with real shift
// ============================================================
static val genEigsRealShift(const val& matData, int n, int nev, int ncv,
                            double sigma, const std::string& rule,
                            int maxIter, double tol)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseGenRealShiftSolve<double> op(mat);
    GenEigsRealShiftSolver<DenseGenRealShiftSolve<double>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 5. Dense general eigenvalue solver with complex shift
// ============================================================
static val genEigsComplexShift(const val& matData, int n, int nev, int ncv,
                               double sigmaR, double sigmaI,
                               const std::string& rule, int maxIter, double tol)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseGenComplexShiftSolve<double> op(mat);
    GenEigsComplexShiftSolver<DenseGenComplexShiftSolve<double>> solver(op, nev, ncv, sigmaR, sigmaI);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 6. Dense generalized symmetric eigensolver (Cholesky mode)
//    Solves Ax = λBx where B is positive definite
// ============================================================
static val symGEigsCholesky(const val& matA, const val& matB, int n,
                            int nev, int ncv, const std::string& rule,
                            int maxIter, double tol)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    DenseSymMatProd<double> opA(A);
    DenseCholesky<double> opB(B);
    SymGEigsSolver<DenseSymMatProd<double>, DenseCholesky<double>, GEigsMode::Cholesky> solver(opA, opB, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 7. Dense generalized symmetric eigensolver (shift-invert)
//    Solves Ax = λBx near shift sigma
// ============================================================
static val symGEigsShiftInvert(const val& matA, const val& matB, int n,
                               int nev, int ncv, double sigma,
                               const std::string& rule, int maxIter, double tol)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    using OpType = SymShiftInvert<double, Eigen::Dense, Eigen::Dense>;
    using BOpType = DenseSymMatProd<double>;
    OpType op(A, B);
    BOpType Bop(B);
    SymGEigsShiftSolver<OpType, BOpType, GEigsMode::ShiftInvert> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 8. Dense generalized symmetric eigensolver (buckling mode)
// ============================================================
static val symGEigsBuckling(const val& matA, const val& matB, int n,
                            int nev, int ncv, double sigma,
                            const std::string& rule, int maxIter, double tol)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    using OpType = SymShiftInvert<double, Eigen::Dense, Eigen::Dense>;
    using BOpType = DenseSymMatProd<double>;
    OpType op(A, B);
    BOpType Bop(A);  // Buckling mode uses A for the B-operation
    SymGEigsShiftSolver<OpType, BOpType, GEigsMode::Buckling> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 9. Dense generalized symmetric eigensolver (Cayley mode)
// ============================================================
static val symGEigsCayley(const val& matA, const val& matB, int n,
                          int nev, int ncv, double sigma,
                          const std::string& rule, int maxIter, double tol)
{
    Eigen::MatrixXd A = jsArrayToMatrix(matA, n, n);
    Eigen::MatrixXd B = jsArrayToMatrix(matB, n, n);
    using OpType = SymShiftInvert<double, Eigen::Dense, Eigen::Dense>;
    using BOpType = DenseSymMatProd<double>;
    OpType op(A, B);
    BOpType Bop(B);
    SymGEigsShiftSolver<OpType, BOpType, GEigsMode::Cayley> solver(op, Bop, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 10. Davidson symmetric eigensolver (dense)
// ============================================================
static val davidsonSymEigs(const val& matData, int n, int nev,
                           const std::string& rule, int maxIter, double tol)
{
    Eigen::MatrixXd mat = jsArrayToMatrix(matData, n, n);
    DenseSymMatProd<double> op(mat);
    DavidsonSymEigsSolver<DenseSymMatProd<double>> solver(op, nev);
    solver.compute(parseSortRule(rule), maxIter, tol);

    val result = val::object();
    result.set("converged", solver.info() == CompInfo::Successful);
    result.set("info", compInfoToString(solver.info()));
    result.set("numIter", solver.num_iterations());
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
                         int maxIter, double tol)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseSymMatProd<double> op(mat);
    SymEigsSolver<SparseSymMatProd<double>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 13. Sparse symmetric eigenvalue solver with shift-and-invert
// ============================================================
static val sparseSymEigsShift(const val& triplets, int rows, int cols,
                              int nev, int ncv, double sigma,
                              const std::string& rule, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseSymShiftSolve<double> op(mat);
    SymEigsShiftSolver<SparseSymShiftSolve<double>> solver(op, nev, ncv, sigma);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeRealResult(solver);
}

// ============================================================
// 14. Sparse general eigenvalue solver
// ============================================================
static val sparseGenEigs(const val& triplets, int rows, int cols,
                         int nev, int ncv, const std::string& rule,
                         int maxIter, double tol)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseGenMatProd<double> op(mat);
    GenEigsSolver<SparseGenMatProd<double>> solver(op, nev, ncv);
    solver.init();
    solver.compute(parseSortRule(rule), maxIter, tol);
    return makeComplexResult(solver);
}

// ============================================================
// 15. Sparse general eigenvalue solver with real shift
// ============================================================
static val sparseGenEigsRealShift(const val& triplets, int rows, int cols,
                                  int nev, int ncv, double sigma,
                                  const std::string& rule, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseGenRealShiftSolve<double> op(mat);
    GenEigsRealShiftSolver<SparseGenRealShiftSolve<double>> solver(op, nev, ncv, sigma);
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
                                     const std::string& rule, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> mat = jsTripletsToSparse(triplets, rows, cols);
    SparseGenComplexShiftSolve<double> op(mat);
    GenEigsComplexShiftSolver<SparseGenComplexShiftSolve<double>> solver(op, nev, ncv, sigmaR, sigmaI);
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
                                  int maxIter, double tol)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    SparseSymMatProd<double> opA(A);
    SparseCholesky<double> opB(B);
    SymGEigsSolver<SparseSymMatProd<double>, SparseCholesky<double>, GEigsMode::Cholesky> solver(opA, opB, nev, ncv);
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
                                        int maxIter, double tol)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    SparseSymMatProd<double> opA(A);
    SparseRegularInverse<double> opB(B);
    SymGEigsSolver<SparseSymMatProd<double>, SparseRegularInverse<double>, GEigsMode::RegularInverse> solver(opA, opB, nev, ncv);
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
                                     const std::string& rule, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    using OpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = SparseSymMatProd<double>;
    OpType op(A, B);
    BOpType Bop(B);
    SymGEigsShiftSolver<OpType, BOpType, GEigsMode::ShiftInvert> solver(op, Bop, nev, ncv, sigma);
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
                                  const std::string& rule, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    using OpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = SparseSymMatProd<double>;
    OpType op(A, B);
    BOpType Bop(A);
    SymGEigsShiftSolver<OpType, BOpType, GEigsMode::Buckling> solver(op, Bop, nev, ncv, sigma);
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
                                const std::string& rule, int maxIter, double tol)
{
    Eigen::SparseMatrix<double> A = jsTripletsToSparse(tripsA, rows, cols);
    Eigen::SparseMatrix<double> B = jsTripletsToSparse(tripsB, rows, cols);
    using OpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = SparseSymMatProd<double>;
    OpType op(A, B);
    BOpType Bop(B);
    SymGEigsShiftSolver<OpType, BOpType, GEigsMode::Cayley> solver(op, Bop, nev, ncv, sigma);
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

    // Create random initial eigenvector approximations
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X_dense =
        Eigen::MatrixXd::Random(n, nev);
    Eigen::SparseMatrix<double> X(X_dense.sparseView());

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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X_dense =
        Eigen::MatrixXd::Random(n, nev);
    Eigen::SparseMatrix<double> X(X_dense.sparseView());

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
// Embind module
// ============================================================
EMSCRIPTEN_BINDINGS(spectra)
{
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
}
