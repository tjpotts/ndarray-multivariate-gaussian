# ndarray-multivariate-gaussian [![Build Status](https://travis-ci.com/tjpotts/ndarray-multivariate-gaussian.svg?branch=master)](https://travis-ci.com/tjpotts/ndarray-multivariate-gaussian)

Draw samples from a multivariate gaussian distribution

## Example

```javascript
import ndarray from 'ndarray';
import mvGaussian from 'ndarray-multivariate-gaussian';

const mean = ndarray([1, 2]);
const covariance = ndarray([4, 1, 1, 2], [2,2]);
const N = 10;

// Draw N samples from a 2-dimensional Gaussian distribution
let samples = mvGaussian(mean, covariance, N);
console.log(samples.shape);
// [2, 10]

// Draw a single sample from a 2-dimensional Gaussian distribution
let sample = mvGaussian(mean, covariance);
console.log(sample.shape);
// [2]
```

## Installation

```javascript
$ npm install ndarray-multivariate-gaussian
```

## API

### samples = ndarray-multivariate-gaussian(mu, sigma, [N])
**Arguments**:
- `mu`: n-length mean vector of the distribution
- `sigma`: nxn covariance matrix of the distribution
- `N`: Number of samples to draw. If ommitted, a single sample is drawn

**Returns**: An nxN matrix where each column is a single sample drawn from the distribution. If N is not specified, a single sample is returned as an n-length vector.

## License
&copy; 2020 Timothy Potts. MIT License.

