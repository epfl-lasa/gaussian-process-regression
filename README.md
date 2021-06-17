# Gaussian Process Regression

This is a library implementing standard GP-regression. Currently, it uses only the squared exponential covariance
function but can easily be modified for different covariance functions.

This is a template library - no linking required. Just make sure your code can find `gaussian_processs_regression.h` and
you should be good to go. Install with

```bash
git clone -b agnostic --single-branch https://github.com/epfl-lasa/gaussian-process-regression.git \
  && mkdir -p gaussian-process-regression/build \
  && cd gaussian-process-regression/build \
  && cmake -DCMAKE_BUILD_TYPE=Release .. \
  && make -j \
  && make install
```

## Dependencies

The version here is packaged as a CMake project.

It depends on [Eigen3](http://eigen.tuxfamily.org/).

## Basic Usage

```bash
// input_dim and output_dim are integers specifying the dimensionality of input and output data respectively
GaussianProcessRegression<float> myGPR(input_dim, output_dim);
// the hyperparameters are float values specifying length-scale, signal variance and observation noise variance
myGPR.SetHyperParams(length_scale, sigma_f, sigma_n);
Eigen::VectorXf train_input(input_dim);
Eigen::VectorXf train_output(output_dim);
// add some training data
int n_train = 100;
for(int k=0; k<n_train; k++){
    train_input.setRandom();
    train_output.setRandom();
    myGPR.AddTrainingData(train_input,train_output)
}
// get output at a testing point
Eigen::VectorXf test_input(input_dim);
Eigen::VectorXf test_output(output_dim);
test_input.setRandom();
test_output = myGPR.DoRegression(test_input);
```

## Notes on multidimensional outputs

As seen in the example above, the library supports multiple outputs, but in a kind of "dumb" way. The same covariance
function and inputs are used to model all the outputs.

## TODO

- Add possibility to use user-supplied covariance function.
- Add possibility to get variance information
