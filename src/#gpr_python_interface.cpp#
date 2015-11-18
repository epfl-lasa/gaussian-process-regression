#include <boost/python.hpp>
//#include <string>
#include "gaussian_process_regression/gaussian_process_regression.h"

#include <eigen3/Eigen/Dense>
#include <numpy/ndarrayobject.h> // 

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


namespace bp = boost::python;

typedef double real;


inline void DEBUG(std::string message){
  std::cout<<message<<std::endl;
}
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
  // TODO: change this to copy the data in one go instead of looping!
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

//template <typename EigenType>
typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> MatrixXr;
typedef MatrixXr EigenType;
struct EigenToNumpyConverter{
  // function returning a python object containing the numpy array. 
  static PyObject* convert(EigenType in){


    // //    dimensionality of array:
    // DEBUG("I am here");
    // npy_intp dims[2] = {static_cast<npy_intp>(in.rows()),static_cast<npy_intp>(in.cols())};
    // // create the raw python array and set its data to point to the data of in
    // DEBUG("I am here");
    // PyObject * pyObj = PyArray_SimpleNewFromData( 2,dims, NPY_DOUBLE, const_cast<real*>(in.data()) );
    // // get a handle to the created object. Not sure what this does.. something to do with reference counting?
    // DEBUG("I am here");
    // bp::handle<> handle( pyObj );
    // // interpret the object as a numpy array
    // DEBUG("I am here 4");

    // bp::numeric::array arr( handle );
    // // need to return a copy or else object will be destroyed here
    // //  return arr.copy();
    // // instead of copying, trying with incref here, increasing the reference count! Very python!
    
    // auto a = bp::incref(arr.ptr());

    npy_intp dim = 3;
    auto p = PyArray_SimpleNew(1,&dim,NPY_FLOAT);
    DEBUG("I am here 5");
    delete p;
    //return pyObj;
    DEBUG("I am here 6");
    return bp::incref(p);
  }
};
// todo: a numpy_to_eigen without copying data
// could be useful at testing time (need to care about lifetime of underlyng data!)

// instatiate and register converter for Vector and Matrix
//bp::to_python_converter<MatrixXr,EigenToNumpyConverter>();



// a tiny wrapper class for interfacing nicely with numpy
typedef GaussianProcessRegression<real> GPR;
class NumpyGPR : public GPR{
public:
  NumpyGPR(int inputDim, int outputDim) : GPR(inputDim,outputDim){};
  // methods that will be part of the python interface and should be overridden:
  void AddTrainingData(const bp::numeric::array & new_input,const bp::numeric::array & new_output){
    //GPR::AddTrainingData(numpy_to_eigen(new_input),numpy_to_eigen(new_output));
    //numpy_to_eigen(new_input);
    auto ni = numpy_to_eigen(new_input);
    auto no = numpy_to_eigen(new_output);
    if(ni.cols()<1){
      GPR::AddTrainingData(ni, no); // single add
    }else{
      GPR::AddTrainingDataBatch(ni,no); // batch add
    }
  };

  bp::object get_input_data(){
    //    return eigen_to_numpy(GPR::get_input_data());
  }

  bp::object get_output_data(){
    //    return eigen_to_numpy(GPR::get_output_data());
  }
  void test_2(){
        npy_intp dim = 3;
        auto p = PyArray_SimpleNew(1,&dim,NPY_FLOAT);
	delete p;
  }
  
  Eigen::MatrixXd test_eigen_to_numpy(){
    Eigen::MatrixXd a(2,2);
    a(1,2) = 1.2;
    a(2,1) = 2.1;
    EigenToNumpyConverter c;
    auto p = c.convert(a);
    //delete p;
    DEBUG("back here");
    return a;
    
  }
  

  // just a function to test the eignen <-> numpy interface
  // bp::object simple_test(const bp::numeric::array & in){
  //   auto a = numpy_to_eigen(in);
  //   std::cout<<a<<std::endl;
  //   return eigen_to_numpy(a);
  // };

};



BOOST_PYTHON_MODULE(gaussian_process_regression){

  bp::to_python_converter<
  MatrixXr,
  EigenToNumpyConverter>();

  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array(); // need this otherwise creating arrays will cause segfault
  bp::class_<NumpyGPR>("GaussianProcessRegression",bp::init<int,int>())
    .def("SetHyperParams",&NumpyGPR::SetHyperParams)
    .def("AddTrainingData",&NumpyGPR::AddTrainingData)
    .def("get_input_data",&NumpyGPR::get_input_data)
    .def("get_output_data",&NumpyGPR::get_output_data)
    .def("test_eigen_to_numpy",&NumpyGPR::test_eigen_to_numpy)
    .def("test_2",&NumpyGPR::test_2)
    ;
  //  bp::to_python_value<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> >()

};
