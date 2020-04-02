import ndarray from 'ndarray';
import zeros from 'zeros';
import ops from 'ndarray-ops';
import squeeze from 'ndarray-squeeze';
import gemm from 'ndarray-gemm';
import cholesky from 'ndarray-cholesky-factorization';
import gaussian from 'gaussian';

export default function ndarray_multivariate_gaussian(mu, sigma, N) {
    mu = squeeze(mu);

    if (mu.dimension > 1)
        throw new Error("Input mean is not a vector");
    if (sigma.dimension !== 2)
        throw new Error("Input covariance is not a 2-dimensional matrix");
    if (sigma.shape[0] !== sigma.shape[1])
        throw new Error("Input covariance is not square");
    if (mu.shape[0] !== sigma.shape[0])
        throw new Error("Input mean and covariance dimensions do not match");
    //TODO: Check that sigma is positive semi-definite and symmetric

    // If N is not specified, set it to 1 and output the sample as a vector instead of a matrix
    let outputVector = false;
    if (N === undefined) {
        N = 1;
        outputVector = true;
    }
    const n = mu.shape[0];

    // Sample points from the distribution N(0,I)
    let u = ndarray([], [n, N]);
    let unit_dist = gaussian(0,1);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < N; j++) {
            u.set(i,j,unit_dist.ppf(Math.random()));
        }
    }

    // Calculate cholesky decomposition
    let L = zeros(sigma.shape);
    cholesky(sigma,L);

    // Transform into the output points
    let x = ndarray([], [n, N]);
    gemm(x,L,u);

    // Add mu to each point
    for (let i = 0; i < N; i++) {
        ops.addeq(x.pick(null,i), mu);
    }

    return outputVector ? squeeze(x) : x;
}

