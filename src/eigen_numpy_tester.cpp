#include <boost/python.hpp>
#include "gaussian_process_regression/eigen_numpy_conversion.h"

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
