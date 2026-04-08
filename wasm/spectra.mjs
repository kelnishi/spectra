// spectra.mjs — JS wrapper for the Spectra WASM module
// Provides a clean API with optional progress callbacks on all solver functions.
//
// Usage:
//   import { createSpectra } from './spectra.mjs';
//   const spectra = await createSpectra();
//
//   // Without progress
//   const result = spectra.symEigs(matrix, n, nev, ncv, 'LargestAlge', 1000, 1e-10);
//
//   // With progress callback
//   const result = spectra.symEigs(matrix, n, nev, ncv, 'LargestAlge', 1000, 1e-10, (info) => {
//     console.log(`${(info.progress * 100).toFixed(1)}% — ${info.opsCompleted}/${info.estimatedTotalOps} ops`);
//   });
//
//   // Zero-copy CSR input (for large sparse matrices):
//   //   Pass CSR TypedArrays directly — avoids creating millions of {row,col,val} objects.
//   const result = spectra.csr.sparseSymEigs(
//     rowOffsets,  // Int32Array, length = rows + 1
//     colIndices,  // Int32Array, length = nnz
//     values,      // Float64Array, length = nnz
//     rows, cols, nev, ncv, 'LargestAlge', 1000, 1e-10, onProgress
//   );
//   // Result eigenvalues/eigenvectors are Float64Arrays (not plain Arrays).

import createSpectraWasm from '../build-wasm/wasm/spectra.js';

export async function createSpectra(wasmOptions) {
    const wasm = await createSpectraWasm(wasmOptions);

    // Decode a C++ exception into a readable JS Error.
    // With -fwasm-exceptions, caught exceptions are WebAssembly.Exception objects.
    // With legacy exceptions, they're raw pointer integers/bigints.
    // EXPORT_EXCEPTION_HANDLING_HELPERS provides getExceptionMessage() for both.
    function decodeException(err) {
        try {
            // getExceptionMessage handles both WebAssembly.Exception and raw pointers
            const [type, msg] = wasm.getExceptionMessage(err);
            const decoded = new Error(msg ? `${type}: ${msg}` : type);
            decoded.name = type;
            if (err instanceof WebAssembly.Exception && err.stack) {
                decoded.stack = err.stack;
            }
            return decoded;
        } catch {
            // Fallback if decoding fails
            if (err instanceof Error) return err;
            return new Error(`C++ exception: ${err}`);
        }
    }

    // Wrap a WASM function so its last argument is an optional JS progress callback.
    // If omitted or null/undefined, passes undefined to the C++ side (no-op).
    // Also decodes C++ exceptions into readable JS errors.
    function withProgress(wasmFn) {
        return function (...args) {
            const last = args[args.length - 1];
            if (typeof last !== 'function') {
                args.push(undefined);
            }
            try {
                return wasmFn.apply(wasm, args);
            } catch (err) {
                throw decodeException(err);
            }
        };
    }

    // ----------------------------------------------------------------
    // Zero-copy CSR helpers: allocate in WASM heap, bulk-copy once,
    // call the C++ function with raw pointers, then free.
    // ----------------------------------------------------------------
    function allocI32(arr) {
        const ptr = wasm._malloc(arr.byteLength);
        wasm.HEAP32.set(arr, ptr / 4);
        return ptr;
    }

    function allocF64(arr) {
        const ptr = wasm._malloc(arr.byteLength);
        wasm.HEAPF64.set(arr, ptr / 8);
        return ptr;
    }

    // Call a CSR WASM function with one sparse matrix (rowOffsets, colIndices, values)
    // plus arbitrary trailing args.  Handles alloc/free around the call.
    function withCSR(wasmFn, rowOffsets, colIndices, values, ...rest) {
        const roPtr = allocI32(rowOffsets instanceof Int32Array ? rowOffsets : new Int32Array(rowOffsets));
        const ciPtr = allocI32(colIndices instanceof Int32Array ? colIndices : new Int32Array(colIndices));
        const vPtr = allocF64(values instanceof Float64Array ? values : new Float64Array(values));
        try {
            return wasmFn.call(wasm, roPtr, ciPtr, vPtr, ...rest);
        } catch (err) {
            throw decodeException(err);
        } finally {
            wasm._free(roPtr);
            wasm._free(ciPtr);
            wasm._free(vPtr);
        }
    }

    // Call a CSR WASM function with two sparse matrices (A and B in CSR)
    // plus arbitrary trailing args.
    function withCSR2(wasmFn, roA, ciA, vA, roB, ciB, vB, ...rest) {
        const roPtrA = allocI32(roA instanceof Int32Array ? roA : new Int32Array(roA));
        const ciPtrA = allocI32(ciA instanceof Int32Array ? ciA : new Int32Array(ciA));
        const vPtrA = allocF64(vA instanceof Float64Array ? vA : new Float64Array(vA));
        const roPtrB = allocI32(roB instanceof Int32Array ? roB : new Int32Array(roB));
        const ciPtrB = allocI32(ciB instanceof Int32Array ? ciB : new Int32Array(ciB));
        const vPtrB = allocF64(vB instanceof Float64Array ? vB : new Float64Array(vB));
        try {
            return wasmFn.call(wasm,
                roPtrA, ciPtrA, vPtrA, vA.length,
                roPtrB, ciPtrB, vPtrB, vB.length,
                ...rest);
        } catch (err) {
            throw decodeException(err);
        } finally {
            wasm._free(roPtrA);
            wasm._free(ciPtrA);
            wasm._free(vPtrA);
            wasm._free(roPtrB);
            wasm._free(ciPtrB);
            wasm._free(vPtrB);
        }
    }

    // Wrap a single-matrix CSR function with optional progress callback as last arg.
    function csrSingle(wasmFn) {
        return function (rowOffsets, colIndices, values, rows, cols, ...rest) {
            const last = rest[rest.length - 1];
            if (typeof last !== 'function') {
                rest.push(undefined);
            }
            return withCSR(wasmFn, rowOffsets, colIndices, values,
                rows, cols, values.length, ...rest);
        };
    }

    // Wrap a two-matrix CSR function with optional progress callback as last arg.
    function csrDouble(wasmFn) {
        return function (roA, ciA, vA, roB, ciB, vB, rows, cols, ...rest) {
            const last = rest[rest.length - 1];
            if (typeof last !== 'function') {
                rest.push(undefined);
            }
            return withCSR2(wasmFn, roA, ciA, vA, roB, ciB, vB,
                rows, cols, ...rest);
        };
    }

    return {
        // Utility
        estimateOps: wasm.estimateOps.bind(wasm),

        // Dense standard eigenvalue solvers
        //   symEigs(matrix, n, nev, ncv, rule, maxIter, tol, [onProgress])
        symEigs: withProgress(wasm.symEigs),
        //   symEigsShift(matrix, n, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        symEigsShift: withProgress(wasm.symEigsShift),
        //   genEigs(matrix, n, nev, ncv, rule, maxIter, tol, [onProgress])
        genEigs: withProgress(wasm.genEigs),
        //   genEigsRealShift(matrix, n, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        genEigsRealShift: withProgress(wasm.genEigsRealShift),
        //   genEigsComplexShift(matrix, n, nev, ncv, sigmaR, sigmaI, rule, maxIter, tol, [onProgress])
        genEigsComplexShift: withProgress(wasm.genEigsComplexShift),

        // Dense generalized symmetric eigenvalue solvers (Ax = λBx)
        //   symGEigsCholesky(matA, matB, n, nev, ncv, rule, maxIter, tol, [onProgress])
        symGEigsCholesky: withProgress(wasm.symGEigsCholesky),
        //   symGEigsShiftInvert(matA, matB, n, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        symGEigsShiftInvert: withProgress(wasm.symGEigsShiftInvert),
        //   symGEigsBuckling(matA, matB, n, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        symGEigsBuckling: withProgress(wasm.symGEigsBuckling),
        //   symGEigsCayley(matA, matB, n, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        symGEigsCayley: withProgress(wasm.symGEigsCayley),

        // Davidson solver
        //   davidsonSymEigs(matrix, n, nev, rule, maxIter, tol, [onProgress])
        davidsonSymEigs: withProgress(wasm.davidsonSymEigs),

        // Dense partial SVD (no progress — internally creates its own solver)
        //   partialSVD(matrix, rows, cols, ncomp, ncv, maxIter, tol)
        partialSVD: wasm.partialSVD.bind(wasm),

        // Sparse standard eigenvalue solvers (triplet input: [{row, col, val}, ...])
        //   sparseSymEigs(triplets, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
        sparseSymEigs: withProgress(wasm.sparseSymEigs),
        //   sparseSymEigsShift(triplets, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        sparseSymEigsShift: withProgress(wasm.sparseSymEigsShift),
        //   sparseGenEigs(triplets, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
        sparseGenEigs: withProgress(wasm.sparseGenEigs),
        //   sparseGenEigsRealShift(triplets, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        sparseGenEigsRealShift: withProgress(wasm.sparseGenEigsRealShift),
        //   sparseGenEigsComplexShift(triplets, rows, cols, nev, ncv, sigmaR, sigmaI, rule, maxIter, tol, [onProgress])
        sparseGenEigsComplexShift: withProgress(wasm.sparseGenEigsComplexShift),

        // Sparse generalized symmetric eigenvalue solvers
        //   sparseSymGEigsCholesky(tripsA, tripsB, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
        sparseSymGEigsCholesky: withProgress(wasm.sparseSymGEigsCholesky),
        //   sparseSymGEigsRegularInverse(tripsA, tripsB, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
        sparseSymGEigsRegularInverse: withProgress(wasm.sparseSymGEigsRegularInverse),
        //   sparseSymGEigsShiftInvert(tripsA, tripsB, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        sparseSymGEigsShiftInvert: withProgress(wasm.sparseSymGEigsShiftInvert),
        //   sparseSymGEigsBuckling(tripsA, tripsB, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        sparseSymGEigsBuckling: withProgress(wasm.sparseSymGEigsBuckling),
        //   sparseSymGEigsCayley(tripsA, tripsB, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
        sparseSymGEigsCayley: withProgress(wasm.sparseSymGEigsCayley),

        // Sparse partial SVD (no progress)
        //   sparsePartialSVD(triplets, rows, cols, ncomp, ncv, maxIter, tol)
        sparsePartialSVD: wasm.sparsePartialSVD.bind(wasm),

        // LOBPCG solvers (no progress — different iteration model)
        //   lobpcg(tripsA, n, nev, maxIter, tol)
        lobpcg: wasm.lobpcg.bind(wasm),
        //   lobpcgGeneralized(tripsA, tripsB, n, nev, maxIter, tol)
        lobpcgGeneralized: wasm.lobpcgGeneralized.bind(wasm),

        // ============================================================
        // CSR (zero-copy TypedArray) sparse solvers
        // ============================================================
        // Pass CSR-format TypedArrays directly — eliminates GC pressure
        // from millions of {row, col, val} JS objects.
        //
        // All functions accept:
        //   rowOffsets:  Int32Array (length = rows + 1)
        //   colIndices:  Int32Array (length = nnz)
        //   values:      Float64Array (length = nnz)
        //
        // Results use Float64Array for eigenvalues/eigenvectors.
        csr: {
            //   sparseSymEigs(rowOffsets, colIndices, values, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
            sparseSymEigs: csrSingle(wasm.sparseSymEigsCSR),
            //   sparseSymEigsShift(rowOffsets, colIndices, values, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
            sparseSymEigsShift: csrSingle(wasm.sparseSymEigsShiftCSR),
            //   sparseGenEigs(rowOffsets, colIndices, values, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
            sparseGenEigs: csrSingle(wasm.sparseGenEigsCSR),
            //   sparseGenEigsRealShift(rowOffsets, colIndices, values, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
            sparseGenEigsRealShift: csrSingle(wasm.sparseGenEigsRealShiftCSR),
            //   sparseGenEigsComplexShift(rowOffsets, colIndices, values, rows, cols, nev, ncv, sigmaR, sigmaI, rule, maxIter, tol, [onProgress])
            sparseGenEigsComplexShift: csrSingle(wasm.sparseGenEigsComplexShiftCSR),

            //   sparseSymGEigsCholesky(roA, ciA, vA, roB, ciB, vB, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
            sparseSymGEigsCholesky: csrDouble(wasm.sparseSymGEigsCholeskyCSR),
            //   sparseSymGEigsRegularInverse(roA, ciA, vA, roB, ciB, vB, rows, cols, nev, ncv, rule, maxIter, tol, [onProgress])
            sparseSymGEigsRegularInverse: csrDouble(wasm.sparseSymGEigsRegularInverseCSR),
            //   sparseSymGEigsShiftInvert(roA, ciA, vA, roB, ciB, vB, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
            sparseSymGEigsShiftInvert: csrDouble(wasm.sparseSymGEigsShiftInvertCSR),
            //   sparseSymGEigsBuckling(roA, ciA, vA, roB, ciB, vB, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
            sparseSymGEigsBuckling: csrDouble(wasm.sparseSymGEigsBucklingCSR),
            //   sparseSymGEigsCayley(roA, ciA, vA, roB, ciB, vB, rows, cols, nev, ncv, sigma, rule, maxIter, tol, [onProgress])
            sparseSymGEigsCayley: csrDouble(wasm.sparseSymGEigsCayleyCSR),

            //   sparsePartialSVD(rowOffsets, colIndices, values, rows, cols, ncomp, ncv, maxIter, tol)
            sparsePartialSVD(rowOffsets, colIndices, values, rows, cols, ncomp, ncv, maxIter, tol) {
                return withCSR(wasm.sparsePartialSVDCSR, rowOffsets, colIndices, values,
                    rows, cols, values.length, ncomp, ncv, maxIter, tol);
            },

            //   lobpcg(rowOffsets, colIndices, values, n, nev, maxIter, tol)
            lobpcg(rowOffsets, colIndices, values, n, nev, maxIter, tol) {
                return withCSR(wasm.lobpcgCSR, rowOffsets, colIndices, values,
                    n, values.length, nev, maxIter, tol);
            },

            //   lobpcgGeneralized(roA, ciA, vA, roB, ciB, vB, n, nev, maxIter, tol)
            lobpcgGeneralized(roA, ciA, vA, roB, ciB, vB, n, nev, maxIter, tol) {
                return withCSR2(wasm.lobpcgGeneralizedCSR, roA, ciA, vA, roB, ciB, vB,
                    n, nev, maxIter, tol);
            },

            // Standalone Eigen linear solvers
            //   sparseCholesky(rowOffsets, colIndices, values, rows, b, nrhs?)
            //   Solves Ax = b for SPD matrix A.  b is Float64Array (length = rows * nrhs).
            //   nrhs defaults to 1.  Returns { success, x, error? }.
            sparseCholesky(rowOffsets, colIndices, values, rows, b, nrhs) {
                nrhs = nrhs || 1;
                const roPtr = allocI32(rowOffsets instanceof Int32Array ? rowOffsets : new Int32Array(rowOffsets));
                const ciPtr = allocI32(colIndices instanceof Int32Array ? colIndices : new Int32Array(colIndices));
                const vPtr = allocF64(values instanceof Float64Array ? values : new Float64Array(values));
                const bPtr = allocF64(b instanceof Float64Array ? b : new Float64Array(b));
                try {
                    return wasm.sparseCholeskyCSR(roPtr, ciPtr, vPtr, values.length,
                        bPtr, rows, nrhs);
                } catch (err) {
                    throw decodeException(err);
                } finally {
                    wasm._free(roPtr);
                    wasm._free(ciPtr);
                    wasm._free(vPtr);
                    wasm._free(bPtr);
                }
            },
        },

        // Direct access to the underlying WASM module
        _wasm: wasm,
    };
}

export default createSpectra;
