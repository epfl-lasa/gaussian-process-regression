#include <boost/python.hpp>
//#include <string>
#include "gaussian_process_regression/gaussian_process_regression.h"
#include <numpy/ndarrayobject.h>
#include <eigen3/Eigen/Dense>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

using namespace boost::python;

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
object eigen_to_numpy(Eigen::Matrix<R,rows,cols>){
  
};


// a tiny wrapper class for interfacing nicely with numpy
typedef GaussianProcessRegression<real> GPR;
class NumpyGPR : public GPR{
public:
  NumpyGPR(int inputDim, int outputDim) : GPR(inputDim,outputDim){};
  // methods that will be part of the python interface and should be overridden:
  AddTrainingData()
};

BOOST_PYTHON_MODULE(gaussian_process_regression){
  // def("klas",klas_function);

  class_<NumpyGPR>("GaussianProcessRegression",init<int,int>())
    .def("SetHyperParams",&NumpyGPR::SetHyperParams)
    ;
};
