#include <boost/python.hpp>
#include <numpy/ndarrayobject.h> // 

#include <iostream>

namespace bp = boost::python;

inline void DEBUG(std::string message){
  std::cout<<message<<std::endl;
}

struct dummy_struct{
  double a;
  double b;
};

struct MyConverter{
  static PyObject * convert(dummy_struct d){
    //    double a = 42.0;
    npy_intp dim = 3;
    auto p = PyArray_SimpleNew(1,&dim,NPY_FLOAT);
    DEBUG("IN CONVERTER");
    return bp::incref(p);
  }
};

dummy_struct test_convert(){
  dummy_struct hej;
  return hej;
}

BOOST_PYTHON_MODULE(converter_dummy){
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array(); // need this otherwise creating arrays will cause segfault

  bp::to_python_converter<
  dummy_struct,
  MyConverter>();

  bp::def("test_convert",&test_convert);
  //  bp::to_python_value<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> >()

};
