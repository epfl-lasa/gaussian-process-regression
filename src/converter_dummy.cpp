#include <boost/python.hpp>
#include <numpy/ndarrayobject.h> // 
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace bp = boost::python;

inline void DEBUG(std::string message){
  std::cout<<message<<std::endl;
}

// this is some trick to convert from c-type to numpy-type (represented by an enum) at compile time
// first declare a struct template
template<typename scalar_type>
struct c_scalar_to_numpy;
// then define it for float 
template<>
struct c_scalar_to_numpy<float>{
  static const int npy = NPY_FLOAT;
};
// .. and double
template<>
struct c_scalar_to_numpy<double>{
  static const int npy = NPY_DOUBLE;
};
// not the prettiest solution but it works..


// Converter from Eigen to numpy. Follows boost::python way of creating a converter
template <typename EigenType>
struct EigenToNumpyConverter{
  static PyObject * convert(const EigenType& in){
    auto numpy_type = c_scalar_to_numpy<typename EigenType::Scalar >::npy;
    
    //    dimensionality of array:
    npy_intp dims[2] = {static_cast<npy_intp>(in.rows()),static_cast<npy_intp>(in.cols())};
    // create the raw python array and set its data to point to the data of in
    PyObject * pyObj = PyArray_SimpleNewFromData( 2,dims, numpy_type, const_cast<typename EigenType::Scalar* >(in.data()) );
    // get a handle to the created object. Not sure what this does.. something to do with reference counting?
    bp::handle<> handle( pyObj );
    // interpret the object as a numpy array
    bp::numeric::array arr( handle );
    return bp::incref(arr.ptr());
  };
  // this needs to be called in your BOOST_PYTHON_MODULE and will register the converter
  static void register_converter(){
    bp::to_python_converter<EigenType, EigenToNumpyConverter<EigenType> >();
  };
};

// test 2d dynamicaly sized
template <typename MatrixType>
MatrixType test_convert_dyn_mat(){
  MatrixType test(2,2);
  test.setIdentity();
  return test;
}

// test 1d dyn size
template <typename VectorType>
VectorType test_convert_dyn_vec(){
  int len = 7;
  VectorType test(7);
  for (int i = 0; i < len; ++i)
    test(i) = 1.1*i;
  return test;
}


BOOST_PYTHON_MODULE(converter_dummy){
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array(); // need this otherwise creating arrays will cause segfault

  // we need to register this for each eigen type we wish to support, e.g. matrices and vectors of floats and doubles

  EigenToNumpyConverter<Eigen::MatrixXd>().register_converter();
  EigenToNumpyConverter<Eigen::MatrixXf>().register_converter();
  EigenToNumpyConverter<Eigen::VectorXd>().register_converter();
  EigenToNumpyConverter<Eigen::VectorXf>().register_converter();

  bp::def("test_convert_dyn_mat",&test_convert_dyn_mat<Eigen::MatrixXd>);
  bp::def("test_convert_dyn_mat_float",&test_convert_dyn_mat<Eigen::MatrixXf>);
  bp::def("test_convert_dyn_vec",&test_convert_dyn_vec<Eigen::VectorXd>);
  bp::def("test_convert_dyn_vec_float",&test_convert_dyn_vec<Eigen::VectorXf>);
  //  bp::to_python_value<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> >()

};
