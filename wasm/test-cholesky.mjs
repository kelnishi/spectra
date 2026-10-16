// Checks for the sparseCholeskyCSR export: solutions, result shapes and
// failure codes.  Run after a build:  node wasm/test-cholesky.mjs
import { createSpectra } from './spectra.mjs';

const spectra = await createSpectra();
const solve = spectra.csr.sparseCholesky;
let failures = 0;

function check(name, ok, detail = '') {
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ` (${detail})` : ''}`);
    if (!ok) failures++;
}

// CSR of a side x side grid Laplacian plus `shift` on the diagonal.
// With lowerOnly, entries above the diagonal are omitted.
function grid(side, shift, lowerOnly = false) {
    const n = side * side;
    const rowOffsets = [0], colIndices = [], values = [];
    for (let i = 0; i < n; i++) {
        const r = Math.floor(i / side), c = i % side;
        const nbrs = [];
        if (r > 0) nbrs.push(i - side);
        if (c > 0) nbrs.push(i - 1);
        if (c + 1 < side) nbrs.push(i + 1);
        if (r + 1 < side) nbrs.push(i + side);
        const entries = nbrs.map(j => [j, -1]);
        entries.push([i, nbrs.length + shift]);
        entries.sort((a, b) => a[0] - b[0]);
        for (const [j, v] of entries) {
            if (lowerOnly && j > i) continue;
            colIndices.push(j);
            values.push(v);
        }
        rowOffsets.push(colIndices.length);
    }
    return { n, rowOffsets: new Int32Array(rowOffsets), colIndices: new Int32Array(colIndices), values: new Float64Array(values) };
}

// Relative residual of column c of the row-major n x k solution.
function residual(A, x, b, k, c) {
    let num = 0, den = 0;
    for (let i = 0; i < A.n; i++) {
        let s = 0;
        for (let p = A.rowOffsets[i]; p < A.rowOffsets[i + 1]; p++) s += A.values[p] * x[A.colIndices[p] * k + c];
        num += (s - b[i * k + c]) ** 2;
        den += b[i * k + c] ** 2;
    }
    return Math.sqrt(num / den);
}

// 1. Small system with a known solution, one right-hand side.
{
    // [4 1 0; 1 3 1; 0 1 2] x = [6 10 8] has x = [1 2 3].
    const r = solve([0, 2, 5, 7], [0, 1, 0, 1, 2, 1, 2], [4, 1, 1, 3, 1, 1, 2], 3, new Float64Array([6, 10, 8]), 1);
    check('3x3 solve', r.success && r.x instanceof Float64Array && r.x.length === 3 &&
        Math.abs(r.x[0] - 1) + Math.abs(r.x[1] - 2) + Math.abs(r.x[2] - 3) < 1e-12, JSON.stringify(Array.from(r.x ?? [])));
}

// 2. Several right-hand sides share one factorization; x is row-major.
const A = grid(40, 0.01);
const k = 5;
const b = new Float64Array(A.n * k).map((_, i) => Math.sin(i));
{
    const r = solve(A.rowOffsets, A.colIndices, A.values, A.n, b, k);
    const shapeOk = r.success && r.x.rows === A.n && r.x.cols === k && r.x.data.length === A.n * k;
    const worst = shapeOk ? Math.max(...Array.from({ length: k }, (_, c) => residual(A, r.x.data, b, k, c))) : Infinity;
    check('grid solve, 5 right-hand sides', shapeOk && worst < 1e-10, `worst residual ${worst.toExponential(2)}`);
    check('stats reported', r.stats && r.stats.factorNnz > 0 && r.stats.minPivot > 0);
}

// 3. Only the lower triangle is read.
{
    const L = grid(40, 0.01, true);
    const full = solve(A.rowOffsets, A.colIndices, A.values, A.n, b, k);
    const lower = solve(L.rowOffsets, L.colIndices, L.values, L.n, b, k);
    let diff = 0;
    for (let i = 0; i < full.x.data.length; i++) diff = Math.max(diff, Math.abs(full.x.data[i] - lower.x.data[i]));
    check('lower triangle alone gives the same solution', lower.success && diff < 1e-12, `max diff ${diff}`);
}

// 4. Failures come back as { success: false, info, error }.
function expectFailure(name, info, run) {
    const r = run();
    check(name, r.success === false && r.info === info && typeof r.error === 'string', `${r.info}: ${r.error}`);
}
const ones = n => new Float64Array(n).fill(1);
{
    const S = grid(20, 0);
    expectFailure('singular Laplacian', 'NotPositiveDefinite',
        () => solve(S.rowOffsets, S.colIndices, S.values, S.n, ones(S.n), 1));
}
expectFailure('indefinite matrix', 'NotPositiveDefinite',
    () => solve([0, 2, 4], [0, 1, 0, 1], [1, 2, 2, 1], 2, ones(2), 1));
expectFailure('NaN entry', 'NotPositiveDefinite',
    () => solve([0, 1, 2], [0, 1], [NaN, 1], 2, ones(2), 1));
expectFailure('column index out of range', 'InvalidInput',
    () => solve([0, 1, 2], [0, 5], [1, 1], 2, ones(2), 1));
expectFailure('row offsets disagree with nnz', 'InvalidInput',
    () => solve([0, 1, 3], [0, 1], [1, 1], 2, ones(2), 1));
expectFailure('zero rows', 'InvalidInput',
    () => solve([0], [], [], 0, ones(1), 1));

// 5. The module still works after failures.
{
    const r = solve(A.rowOffsets, A.colIndices, A.values, A.n, b, k);
    check('solve after failures', r.success && residual(A, r.x.data, b, k, 0) < 1e-10);
}

console.log(failures ? `${failures} check(s) failed` : 'all checks passed');
process.exit(failures ? 1 : 0);
