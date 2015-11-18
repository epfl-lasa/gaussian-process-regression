#include <boost/python.hpp>
#include <numpy/ndarrayobject.h> // 
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace bp = boost::python;

inline void DEBUG(std::string message){
  std::cout<<message<<std::endl;
}

struct dummy_struct{
  double a;
  double b;
};

typedef double real;
struct MyConverter{
  static PyObject * convert(const Eigen::MatrixXd& in){

        //    dimensionality of array:
    DEBUG("I am here");
    npy_intp dims[2] = {static_cast<npy_intp>(in.rows()),static_cast<npy_intp>(in.cols())};
    // create the raw python array and set its data to point to the data of in
    DEBUG("I am here");
    PyObject * pyObj = PyArray_SimpleNewFromData( 2,dims, NPY_DOUBLE, const_cast<real*>(in.data()) );
    // get a handle to the created object. Not sure what this does.. something to do with reference counting?
    DEBUG("I am here");
    bp::handle<> handle( pyObj );
    // interpret the object as a numpy array
    DEBUG("I am here 4");
    bp::numeric::array arr( handle );

    return bp::incref(arr.ptr());
  }
};

Eigen::MatrixXd test_convert(){
  Eigen::MatrixXd test(2,2);
  test.setIdentity();
  return test;
}

BOOST_PYTHON_MODULE(converter_dummy){
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array(); // need this otherwise creating arrays will cause segfault

  bp::to_python_converter<
    Eigen::MatrixXd,
  MyConverter>();

  bp::def("test_convert",&test_convert);
  //  bp::to_python_value<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> >()

};
