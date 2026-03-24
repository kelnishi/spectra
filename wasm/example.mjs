// Example: compute eigenvalues with progress reporting
import { createSpectra } from './spectra.mjs';

const spectra = await createSpectra();

// 3x3 symmetric matrix (row-major flat array):
//  [ 2  1  0 ]
//  [ 1  3  1 ]
//  [ 0  1  2 ]
const matrix = [2, 1, 0, 1, 3, 1, 0, 1, 2];
const n = 3;
const nev = 2;
const ncv = 3;

// Without progress callback
const result = spectra.symEigs(matrix, n, nev, ncv, 'LargestAlge', 1000, 1e-10);
console.log('Eigenvalues:', result.eigenvalues);
console.log('Actual ops:', result.numOps);

// With progress callback
console.log('\n--- With progress reporting ---');
const result2 = spectra.symEigs(matrix, n, nev, ncv, 'LargestAlge', 1000, 1e-10,
    (info) => {
        const pct = (info.progress * 100).toFixed(1);
        console.log(`  ${pct}% — ${info.opsCompleted}/${info.estimatedTotalOps} ops`);
    }
);
console.log('Eigenvalues:', result2.eigenvalues);

// Estimate ops before running (useful for UI progress bars)
const est = spectra.estimateOps(ncv, 1000);
console.log('\nEstimated ops for ncv=3, maxIter=1000:', est);

// Sparse matrix example with progress
console.log('\n--- Sparse with progress ---');
const triplets = [
    { row: 0, col: 0, val: 2 },
    { row: 0, col: 1, val: 1 },
    { row: 1, col: 0, val: 1 },
    { row: 1, col: 1, val: 3 },
    { row: 1, col: 2, val: 1 },
    { row: 2, col: 1, val: 1 },
    { row: 2, col: 2, val: 2 },
];

const sparseResult = spectra.sparseSymEigs(triplets, 3, 3, 2, 3, 'LargestAlge', 1000, 1e-10,
    (info) => console.log(`  ${(info.progress * 100).toFixed(1)}%`)
);
console.log('Sparse eigenvalues:', sparseResult.eigenvalues);
