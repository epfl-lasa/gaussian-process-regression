#include "gaussian_process_regression/gaussian_process_regression.h"

template<typename R>
GaussianProcessRegression<R>::GaussianProcessRegression(int inputDim,int outputDim)
{
  input_data_.resize(inputDim,0);
  output_data_.resize(outputDim,0);
  n_data_ = 0;
}


template<typename R>
void GaussianProcessRegression<R>::AddTrainingData(VectorXr newInput, VectorXr newOutputs)
{
  n_data_++;
  if(n_data_>=input_data_.cols()){
    input_data_.conservativeResize(input_data_.rows(),n_data_);
    output_data_.conservativeResize(output_data_.rows(),n_data_);
  }

  //cout<<input_data_<<endl<<newInput<<endl;
  input_data_.col(n_data_-1) = newInput;
  output_data_.col(n_data_-1) = newOutputs;
  b_need_prepare_ = true;
}


template<typename R>
R GaussianProcessRegression<R>::SQEcovFuncD(VectorXr x1, VectorXr x2)
{
  dist = x1-x2;
  //cout<<dist<<endl;
  double d = dist.dot(dist);
  d = sigma_f_*sigma_f_*exp(-1/l_scale_/l_scale_/2*d);
  return d;
}

template<typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::SQEcovFunc(MatrixXr x1, VectorXr x2){
  int nCol = x1.cols();
  VectorXr KXx(nCol);
  for(int i=0;i<nCol;i++){
    KXx(i)=SQEcovFuncD(x1.col(i),x2);
  }
  return KXx;
}

// This is a slow process that should be replaced by linear solve at some point
template<typename R>
void GaussianProcessRegression<R>::PrepareRegression(bool force_prepare)
{
  if(!b_need_prepare_ & !force_prepare)
    return;

  KXX = SQEcovFunc(input_data_);
  KXX_ = KXX;
  // add measurement noise
  for(int i=0;i<KXX.cols();i++)
    KXX_(i,i) += sigma_n_*sigma_n_;

  // this is a time theif:
  KXX_ = KXX_.inverse();
  b_need_prepare_ = false;
}


template <typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::DoRegression(const VectorXr& inp,bool prepare){
  if(prepare || b_need_prepare_){
    PrepareRegression();
  }
  VectorXr outp(output_data_.rows());
  outp.setZero();
  KXx = SQEcovFunc(input_data_,inp);
  KxX = SQEcovFunc(input_data_,inp).transpose();
  VectorXr tmp(input_data_.cols());
  // this line is the slow one, hard to speed up further?
  tmp = KXX_*KXx;
  // the rest is noise in comparison with the above line.
  for(int i=0;i<output_data_.rows();i++){
    outp(i)=tmp.dot(output_data_.row(i));
  }
  return outp;
}

template<typename R>
void GaussianProcessRegression<R>::ClearTrainingData()
{
    input_data_.resize(input_data_.rows(),0);
    output_data_.resize(output_data_.rows(),0);
    b_need_prepare_ = true;
    n_data_ = 0;
}

template<typename R>
typename GaussianProcessRegression<R>::MatrixXr GaussianProcessRegression<R>::SQEcovFunc(MatrixXr x1){
  int nCol = x1.cols();
  MatrixXr retMat(nCol,nCol);
  for(int i=0;i<nCol;i++){
    for(int j=i;j<nCol;j++){
      retMat(i,j)=SQEcovFuncD(x1.col(i),x1.col(j));
      retMat(j,i)=retMat(i,j);
    }
  }
  return retMat;
}

template<typename R>
void GaussianProcessRegression<R>::Debug()
{
  std::cout<<"input data \n"<<input_data_<<std::endl;
  std::cout<<"output data \n"<<output_data_<<std::endl;
}

