#include <boost/python.hpp>
//#include <string>
#include "gaussian_process_regression/gaussian_process_regression.h"

#include <eigen3/Eigen/Dense>
#include <numpy/ndarrayobject.h> // do not use this
#include "boost/numpy/ndarray.hpp" // rather prefer this (require Boost.Numpy (boost_numpy via catkin))
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

namespace bp = boost::python;

typedef float real;

/*minimal example for testing succesful compilation
  of python module*/
// const std::string klas_function(){
//   return "hello klas!";
// }

// functions for converting between numpy arrays and Eigen data types
template <typename R, int rows, int cols>
Eigen::Matrix<R,rows,cols> numpy_to_eigen(){
  
};

// function returning a generic python object
template <typename R, int rows, int cols>
bp::object eigen_to_numpy(Eigen::Matrix<R,rows,cols>){
  
};


// a tiny wrapper class for interfacing nicely with numpy
typedef GaussianProcessRegression<real> GPR;
class NumpyGPR : public GPR{
public:
  NumpyGPR(int inputDim, int outputDim) : GPR(inputDim,outputDim){};
  // methods that will be part of the python interface and should be overridden:
  // void AddTrainingData(bp::numeric::array new_input, bp::object new_output){
  //   new_input.getshape();
  // };

};

BOOST_PYTHON_MODULE(gaussian_process_regression){
  // def("klas",klas_function);

  bp::class_<NumpyGPR>("GaussianProcessRegression",bp::init<int,int>())
    .def("SetHyperParams",&NumpyGPR::SetHyperParams)
    ;
};
