#include <gtest/gtest.h>
#include "gaussian_process_regression/gaussian_process_regression.h"
#include <fstream>
#include <vector>
#include <string>
#include <sstream>



using namespace std;
typedef Eigen::Matrix<float,3,1> Vector3r;
// this is the current accuracy level. It can probably be vastly improved by some optimizations here and there.
// for now, it is sufficient
static double COMPARISON_THRESHOLD_FLOAT = 1.0e-5;
static double COMPARISON_THRESHOLD_DOUBLE = 1.0e-5;
 

template<typename input_type, typename output_type>
void load_data(const char *fname, vector<input_type> &inputs, vector<output_type> &outputs, int input_dim, int output_dim) {
  input_type inp,tinp;
  output_type outp,toutp;
  ifstream myfile(fname);
  ASSERT_TRUE(myfile);
  string line;
  while(getline(myfile,line)){
    istringstream line_stream(line);
    for(size_t k = 0; k < input_dim; k++)
      line_stream>>inp(k);
    for(size_t k = 0; k < output_dim; k++)
      line_stream>>outp(k);
    inputs.push_back(inp);
    outputs.push_back(outp);
  }
}

template<typename R>
void set_hyperparameters_from_file(const char *fname, GaussianProcessRegression<R> & gpr) {
  ifstream myfile;
  myfile.open(fname);
  ASSERT_TRUE(myfile);
  R l, f, n;
  myfile>>l>>f>>n;
  myfile.close();
  gpr.SetHyperParams(l,f,n);
}

TEST(FindTestData,HyperParams){
  ifstream myfile("hyperparams.txt");
  ASSERT_TRUE(myfile);
}

TEST(FindTestData,SISO_train_data){
  ifstream myfile("siso_train_data.txt");
  ASSERT_TRUE(myfile);
}

TEST(FindTestData,SISO_test_data){
  ifstream myfile("siso_test_data.txt");
  ASSERT_TRUE(myfile);
}

TEST(Instantiate,Constructors){
  GaussianProcessRegression<float> g1;
  GaussianProcessRegression<double> g2;
  GaussianProcessRegression<float> g3(3,3);
  GaussianProcessRegression<double> g4(3,3);
}

TEST(OutputSize,MIMO){
  const size_t input_dim(20), output_dim(10);

  GaussianProcessRegression<float> gpr(input_dim, output_dim);
  gpr.SetHyperParams(1.1, 1.0, 0.4);
  Eigen::Matrix<float,input_dim,1> train_input, test_input;
  train_input.setRandom();
  test_input.setRandom();
  Eigen::Matrix<float,output_dim,1> train_output, test_output;
  train_output.setRandom();
  test_output.setRandom();
  gpr.AddTrainingData(train_input,train_output);
  auto outp = gpr.DoRegression2(test_input);
  ASSERT_EQ(outp.rows(), output_dim);
}


template<typename R>
void test_do_regression_siso(R threshold){
  GaussianProcessRegression<R> gpr(1,1);
  set_hyperparameters_from_file("hyperparams.txt", gpr);
  typedef Eigen::Matrix<R,1,1> Vec1;
  vector<Vec1> train_inputs, train_outputs, test_inputs, test_outputs;
  load_data("siso_train_data.txt",train_inputs,train_outputs,1,1);
  for(size_t k=0; k<train_inputs.size(); k++){
    gpr.AddTrainingData(train_inputs[k], train_outputs[k]);
  }
  load_data("siso_test_data.txt",test_inputs,test_outputs,1,1);

  for(size_t k=0; k<test_inputs.size(); k++){
    Vec1 outp = gpr.DoRegression2(test_inputs[k]);
    ASSERT_NEAR(test_outputs[k](0), outp(0),threshold);
  }
}

template<typename R>
void test_do_regression_miso(R threshold){
  GaussianProcessRegression<R> gpr(4,1);
  set_hyperparameters_from_file("hyperparams.txt", gpr);
  typedef Eigen::Matrix<R,4,1> input_type;
  typedef Eigen::Matrix<R,1,1> output_type;
  vector<input_type> train_inputs, test_inputs;
  vector<output_type> train_outputs, test_outputs;
  load_data("miso_train_data.txt",train_inputs,train_outputs,4,1);
  for(size_t k=0; k<train_inputs.size(); k++){
    gpr.AddTrainingData(train_inputs[k], train_outputs[k]);
  }
  load_data("miso_test_data.txt",test_inputs,test_outputs,4,1);

  for(size_t k=0; k<test_inputs.size(); k++){
    auto outp = gpr.DoRegression2(test_inputs[k]);
    ASSERT_NEAR(test_outputs[k](0), outp(0),threshold);
  }
}

template<typename R>
void test_do_regression_mimo(R threshold){
  GaussianProcessRegression<R> gpr(4,3);
  set_hyperparameters_from_file("hyperparams.txt", gpr);
  typedef Eigen::Matrix<R,4,1> input_type;
  typedef Eigen::Matrix<R,3,1> output_type;
  vector<input_type> train_inputs, test_inputs;
  vector<output_type> train_outputs, test_outputs;
  load_data("mimo_train_data.txt",train_inputs,train_outputs,4,3);
  for(size_t k=0; k<train_inputs.size(); k++){
    gpr.AddTrainingData(train_inputs[k], train_outputs[k]);
  }
  load_data("mimo_test_data.txt",test_inputs,test_outputs,4,3);

  for(size_t k=0; k<test_inputs.size(); k++){
    auto outp = gpr.DoRegression2(test_inputs[k]);
    for (size_t l=0; l < outp.rows(); ++l)
      {
	ASSERT_NEAR(test_outputs[k](l), outp(l), threshold);
      }
  }
}


TEST(DoRegression,SISO){
  test_do_regression_siso<float>(COMPARISON_THRESHOLD_FLOAT);
}


TEST(DoRegression,SISO_DOBLE){
  test_do_regression_siso<double>(COMPARISON_THRESHOLD_DOUBLE);
}

TEST(DoRegression,MISO){
  test_do_regression_miso<float>(COMPARISON_THRESHOLD_FLOAT);
}

TEST(DoRegression,MIMO){
  test_do_regression_mimo<float>(COMPARISON_THRESHOLD_FLOAT);
}



int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



