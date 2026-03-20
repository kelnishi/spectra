// Example: compute eigenvalues of a symmetric matrix using Spectra WASM
// Adjust path to wherever build output lands
import createSpectra from '../build-wasm/wasm/spectra.js';

const spectra = await createSpectra();

// 3x3 symmetric matrix (row-major flat array):
//  [ 2  1  0 ]
//  [ 1  3  1 ]
//  [ 0  1  2 ]
const matrix = [2, 1, 0, 1, 3, 1, 0, 1, 2];
const n = 3;     // matrix dimension
const nev = 2;   // number of eigenvalues to compute
const ncv = 3;   // convergence parameter (must be > nev, typically 2*nev or more)

const result = spectra.symEigs(matrix, n, nev, ncv, 'LargestAlge', 1000, 1e-10);

if (result.converged) {
    console.log('Eigenvalues:', result.eigenvalues);
    console.log('Eigenvectors:', result.eigenvectors);
} else {
    console.log('Did not converge');
}

// Sparse matrix example (same matrix as triplets):
const triplets = [
    { row: 0, col: 0, val: 2 },
    { row: 0, col: 1, val: 1 },
    { row: 1, col: 0, val: 1 },
    { row: 1, col: 1, val: 3 },
    { row: 1, col: 2, val: 1 },
    { row: 2, col: 1, val: 1 },
    { row: 2, col: 2, val: 2 },
];

const sparseResult = spectra.sparseSymEigs(triplets, 3, 3, 2, 3, 'LargestAlge', 1000, 1e-10);
console.log('Sparse eigenvalues:', sparseResult.eigenvalues);
