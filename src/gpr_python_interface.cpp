#include <boost/python.hpp>
//#include <string>
#include "gaussian_process_regression/gaussian_process_regression.h"

#include <eigen3/Eigen/Dense>
#include <numpy/ndarrayobject.h> // 

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

namespace bp = boost::python;

typedef double real;

/*minimal example for testing succesful compilation
  of python module*/
// const std::string klas_function(){
//   return "hello klas!";
// }

// functions for converting between numpy arrays and Eigen data types
// this one is tested and working correctly for 1d and 2d ndarrays
Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic> numpy_to_eigen(const bp::object & in){
  // get the shape
  auto shape = in.attr("shape");
  int rows = bp::extract<int>(shape[0]);
  int cols;
  bool b_one_d_array = false;
  // determine if we are dealing with a 1d array
  if(bp::len(shape) == 1){
    cols = 1;
    b_one_d_array = true;
  }
  else{
    cols = bp::extract<int>(shape[1]);
  }
  // initialize return matrix
  Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic> out(rows,cols);
  for (int row = 0; row < rows; ++row)
    {
      // if we have a 1d array, we must reference its memory correctly
      if(b_one_d_array){
	  out(row,0) = bp::extract<float>(in[row]);
      }
      else{
      for (int col = 0; col < cols; ++col)
  	{
	  // this is for the 2d array
	  out(row,col) = bp::extract<float>(in[row][col]);
  	}
      }
    }
  return out;
};

// todo: a numpy_to_eigen without copying data
// could be useful at testing time (need to care about lifetime of underlyng data!)

// function returning a generic python object
bp::object eigen_to_numpy(const Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic>& in){
  //npy_intp dims[2] = {1,2};
  npy_intp dims[2] = {static_cast<npy_intp>(in.rows()),static_cast<npy_intp>(in.cols())};
  PyObject * pyObj = PyArray_SimpleNewFromData( 2,dims, NPY_DOUBLE, const_cast<real*>(in.data()) );
  bp::handle<> handle( pyObj );
  bp::numeric::array arr( handle );
  // need to copy or else object will be destroyed here
  return arr.copy();
};


// a tiny wrapper class for interfacing nicely with numpy
typedef GaussianProcessRegression<real> GPR;
class NumpyGPR : public GPR{
public:
  NumpyGPR(int inputDim, int outputDim) : GPR(inputDim,outputDim){};
  // methods that will be part of the python interface and should be overridden:
  void AddTrainingData(const bp::numeric::array & new_input,const bp::numeric::array & new_output){
    //GPR::AddTrainingData(numpy_to_eigen(new_input),numpy_to_eigen(new_output));
    //numpy_to_eigen(new_input);
  };

  bp::object simple_test(const bp::numeric::array & in){
    auto a = numpy_to_eigen(in);
    std::cout<<a<<std::endl;
    return eigen_to_numpy(a);
  };

};

void simple_test2(const bp::numeric::array & in){
    std::cout<<"hello, world"<<std::endl;    
  };


BOOST_PYTHON_MODULE(gaussian_process_regression){
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array(); // need this otherwise creating arrays will cause segfault
  //def("klas",klas_function);
  bp::def("simple_test2",simple_test2);
  bp::class_<NumpyGPR>("GaussianProcessRegression",bp::init<int,int>())
    .def("SetHyperParams",&NumpyGPR::SetHyperParams)
    .def("AddTrainingData",&NumpyGPR::AddTrainingData)
    .def("simple_test",&NumpyGPR::simple_test)
    ;
};
