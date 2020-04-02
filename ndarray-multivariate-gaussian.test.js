import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import mvGaussian from './ndarray-multivariate-gaussian.js';

test('generates samples from multivariate gaussian distributions', () => {
    const mu = ndarray([1,2]);
    const sigma = ndarray([2,1,1,4], [2,2]);
    const n = mu.shape[0];
    const N = 1e6;

    const samples = mvGaussian(mu, sigma, N);

    // Calculate experimental mean
    let mu_actual = ndarray([],[n]);
    for (let i = 0; i < n; i++) {
        let mu_i = ops.sum(samples.pick(i)) / N;
        mu_actual.set(i,mu_i);
    }

    // Check mean error
    // TODO: Figure out what the "right" error threshold is
    let mu_error = ndarray([], [n]);
    ops.sub(mu_error, mu, mu_actual);
    ops.abseq(mu_error);
    for (let i = 0; i < n; i++) {
        expect(mu_error.get(i)).toBeLessThan(0.01);
    }

    // Calculate experimental covariance matrix
    let diff = ndarray([], samples.shape);
    for (let i = 0; i < N; i++) {
        ops.sub(diff.pick(null,i), samples.pick(null,i), mu_actual);
    }
    let sigma_actual = ndarray([], [n,n]);
    let temp = ndarray([], [N]);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            ops.mul(temp, diff.pick(i), diff.pick(j));
            let cov_ij = ops.sum(temp) / (N-1);
            sigma_actual.set(i,j,cov_ij);
        }
    }

    // Check covariance error
    // TODO: Figure out what the "right" error threshold is
    let sigma_error = ndarray([], [n,n]);
    ops.sub(sigma_error, sigma, sigma_actual);
    ops.abseq(sigma_error);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            expect(sigma_error.get(i,j)).toBeLessThan(0.01);
        }
    }
});

test('returns single sample as a vector when N is not specified', () => {
    const mu = ndarray([1,2]);
    const sigma = ndarray([2,1,1,4], [2,2]);

    const sample = mvGaussian(mu, sigma);

    expect(sample.dimension).toBe(1);
});

test('throws error if mu is not 1-dimensional', () => {
    const mu1 = ndarray([1,2], [2,1]);
    const mu2 = ndarray([1,2,3,4], [2,2]);
    const sigma = ndarray([1,0,0,1], [2,2]);

    expect(() => mvGaussian(mu1,sigma)).not.toThrow();
    expect(() => mvGaussian(mu2,sigma)).toThrow();
});

test('throws error if sigma is not 2-dimensional', () => {
    const mu = ndarray([1,2]);
    const sigma = ndarray([1,0,0,1]);

    expect(() => mvGaussian(mu,sigma)).toThrow();
});

test('throws error if sigma is not square', () => {
    const mu = ndarray([1,2]);
    const sigma = ndarray([1,0,0,1], [4,1]);

    expect(() => mvGaussian(mu,sigma)).toThrow();
});

test('throws error if dimensions of mu and sigma do not match', () => {
    const mu = ndarray([1,2,3]);
    const sigma = ndarray([1,0,0,1], [2,2]);

    expect(() => mvGaussian(mu,sigma)).toThrow();
});

