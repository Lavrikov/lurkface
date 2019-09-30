#include "caffe/common.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"

namespace caffe
{
    extern INSTANTIATE_CLASS(InputLayer);
    extern INSTANTIATE_CLASS(ConvolutionLayer);
    extern INSTANTIATE_CLASS(PoolingLayer);
    extern INSTANTIATE_CLASS(ReLULayer);
    extern INSTANTIATE_CLASS(InnerProductLayer);
    extern INSTANTIATE_CLASS(SoftmaxLayer);
    extern INSTANTIATE_CLASS(LRNLayer);
    extern INSTANTIATE_CLASS(ConcatLayer);
    extern INSTANTIATE_CLASS(DropoutLayer);

    REGISTER_LAYER_CLASS(Convolution);
    REGISTER_LAYER_CLASS(ReLU);
    REGISTER_LAYER_CLASS(Pooling);
    REGISTER_LAYER_CLASS(Softmax);
    REGISTER_LAYER_CLASS(LRN);
}