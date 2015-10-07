#include <gtest/gtest.h>
#include "gaussian_process_regression/gaussian_process_regression.h"
#include <fstream>
#include <vector>

using namespace std;
typedef Eigen::Matrix<float,3,1> Vector3r;

void load_training_data(const char *fname, vector<Vector3r> &pos, vector<Vector3r> &vel) {
  Vector3r tempPos;
  Vector3r tempVel;
  ifstream myfile;
  myfile.open(fname);
  while(myfile >> tempPos(0)){
    myfile>>tempPos(1)>>tempPos(2)>>tempVel(0)>>tempVel(1)>>tempVel(2);
    pos.push_back(tempPos);
    vel.push_back(tempVel);
  }
  myfile.close();
}

template<typename R>
void set_hyperparameters_from_file(const char *fname, GaussianProcessRegression<R> & gpr) {
  ifstream myfile;
  myfile.open(fname);
  R l, f, n;
  myfile>>l>>f>>n;
  myfile.close();
  gpr.SetHyperParams(l,f,n);
}


TEST(Instantiate,Constructors){
  GaussianProcessRegression<float> g1;
  GaussianProcessRegression<double> g2;
  GaussianProcessRegression<float> g3(3,3);
  GaussianProcessRegression<double> g4(3,3);
}

TEST(DoRegression,OneDOutputFloat){
  GaussianProcessRegression<float> gpr;
  set_hyperparameters_from_file("hyperparams.txt", gpr);
  vector<Vector3r> train_inputs, train_outputs, test_inputs, test_outputs;
  load_training_data("testdata1d.txt", );
  for(size_t k=0; k<inputs.size(); k++){
    gpr.AddTrainingData(inputs[k], outputs[k]);
  }
  
}

TEST(DoRegression,TwoDOutput){
  
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


