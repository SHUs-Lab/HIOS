#include <cstdio>
#include <sstream>
#include <fstream>
#include <map>
#include <iostream>
#include <string>
#include <cassert>
#include <cstdint>

#include <unistd.h>
#include <pthread.h>
#include <cudnn.h>
#include <stdlib.h>
#include "ops.h"
#include "json.h"
#include "utils.h"
#include <mpi.h>
#if defined(USE_FLOAT16)
typedef __half bias_data_type;
cudnnDataType_t cudnn_bias_data_type = CUDNN_DATA_HALF;
cudnnDataType_t cudnn_data_type = CUDNN_DATA_HALF;
cudnnDataType_t cudnn_conv_data_type = CUDNN_DATA_FLOAT;
cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NHWC;
#elif defined(USE_INT8)
typedef float bias_data_type;
cudnnDataType_t cudnn_bias_data_type = CUDNN_DATA_FLOAT;
cudnnDataType_t cudnn_data_type = CUDNN_DATA_INT8;
cudnnDataType_t cudnn_conv_data_type = CUDNN_DATA_INT32;
cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NHWC;
#else // USE_FLOAT32
typedef float bias_data_type;
cudnnDataType_t cudnn_bias_data_type = CUDNN_DATA_FLOAT;
cudnnDataType_t cudnn_data_type = CUDNN_DATA_FLOAT;
cudnnDataType_t cudnn_conv_data_type = CUDNN_DATA_FLOAT;
cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NCHW;
#endif

#if defined(USE_TENSOR_CORE)
cudnnMathType_t cudnn_math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#else
cudnnMathType_t cudnn_math_type = CUDNN_DEFAULT_MATH;
#endif

#define CONTEXT_WORKSPACE_SIZE 512 * 1024 * 1024  // 256 MB
#define MAX_NUM_GROUPS 10
#define MAX_PROCESS 10
#define MAX_NUM_VALUES 25
#define MAX_NUM_TERMS 25
#define MAX_NUM_NODES 1000
#define MAX_GROUP_SIZE 40
#define MAX_STREAM 10
using std::vector;
using std::map;
using std::stringstream;
using std::string;
using std::size_t;



struct RankInfo{
	int rank;
	int tag;
};

static void print(const char *str, data_type *device_data, int cnt) {
	size_t size = sizeof(data_type) * cnt;
	auto * host_data = (data_type*)malloc(size);
	checkCUDA(cudaMemcpy(host_data, device_data, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	printf("%s ", str);
	for(int i = 0; i < cnt; i++)
		printf("%+2.3f ", (double)host_data[i]);
	printf("\n");
	free(host_data);
}


struct ConvKey {
	int attrs[12];
	ConvKey() {}
	ConvKey(int batch_size, int in_channels, int input_h, int input_w, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups) {
		attrs[0] = batch_size;
		attrs[1] = in_channels;
		attrs[2] = input_h;
		attrs[3] = input_w;
		attrs[4] = out_channels;
		attrs[5] = kernel_h;
		attrs[6] = kernel_w;
		attrs[7] = stride_h;
		attrs[8] = kernel_w;
		attrs[9] = padding_h;
		attrs[10] = padding_w;
		attrs[11] = groups;
	}
	void print() {
		fprintf(stderr, "input_shape: %d %d %d %d  out_channels: %d  kernel, stride, padding: %d %d, %d %d, %d %d  groups: %d\n",
				attrs[0], attrs[1], attrs[2], attrs[3], attrs[4], attrs[5], attrs[6], attrs[7], attrs[8], attrs[9], attrs[10], attrs[11]);
	}
};

bool operator<(const ConvKey &lhs, const ConvKey &rhs) {
	const int n = sizeof(lhs.attrs) / sizeof(lhs.attrs[0]);
	for(int i = 0; i < n; i++) {
		if(lhs.attrs[i] != rhs.attrs[i])
			return lhs.attrs[i] < rhs.attrs[i];
	}
	return false;
}



struct CudnnContext {
	cudnnHandle_t dnn;
	cudaStream_t stream;
	size_t max_size;
	data_type *space;
	int deviceNum;
	CudnnContext(int deviceNum, size_t max_size = CONTEXT_WORKSPACE_SIZE) {

		this->deviceNum = deviceNum;

		this->max_size = max_size;
		cudaSetDevice(deviceNum);
		checkCUDNN(cudnnCreate(&dnn));
		checkCUDA(cudaStreamCreate(&stream));
		checkCUDA(cudaMalloc(&space, max_size));
		checkCUDNN(cudnnSetStream(dnn, stream));
	}
	~CudnnContext() {
		checkCUDNN(cudnnDestroy(dnn));
		checkCUDA(cudaFree(space));
		checkCUDA(cudaStreamDestroy(stream));
	}
};


CudnnContext* contexts[MAX_STREAM];
cudnnConvolutionFwdAlgo_t get_conv_alg(int batch_size, int in_channels, int input_h, int input_w, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups) {
	ConvKey key(batch_size, in_channels, input_h, input_w, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups);

	data_type *input_data, *filter_data, *output_data;
	int output_h = 1 + (input_h - kernel_h + 2 * padding_h) / stride_h;
	int output_w = 1 + (input_w - kernel_w + 2 * padding_w) / stride_w;
	cudnnTensorDescriptor_t inputTensor, outputTensor;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnFilterDescriptor_t filterDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, cudnn_data_format, cudnn_data_type, batch_size, in_channels, input_h, input_w));
	assert(in_channels % groups == 0);
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, cudnn_data_type, cudnn_data_format, out_channels, in_channels / groups, kernel_h, kernel_w));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/, CUDNN_CROSS_CORRELATION, cudnn_conv_data_type));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc, cudnn_math_type));
	checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, groups));
	int n, c, h, w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
	assert(n == batch_size);
	assert(c == out_channels);
	assert(h == output_h);
	assert(w == output_w);
	checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, cudnn_data_format, cudnn_data_type, n, c, h, w));

	size_t input_size = sizeof(data_type) * batch_size * in_channels * input_h * input_w;
	size_t filter_size = sizeof(data_type) * out_channels * (in_channels / groups)* kernel_h * kernel_w;
	size_t output_size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
	checkCUDA(cudaMalloc(&input_data, input_size));
	checkCUDA(cudaMalloc(&filter_data, filter_size));
	checkCUDA(cudaMalloc(&output_data, output_size));


	cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
	int returned;

	checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
				contexts[0]->dnn, inputTensor, input_data, filterDesc,
				filter_data, convDesc, outputTensor, output_data,
				CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf,
				contexts[0]->space, contexts[0]->max_size));
	assert(returned == CUDNN_CONVOLUTION_FWD_ALGO_COUNT);


	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	checkCUDA(cudaFree(input_data));
	checkCUDA(cudaFree(output_data));
	checkCUDA(cudaFree(filter_data));
	return perf[0].algo;
}

struct ConvOP {
	int batch_size;
	int in_channels;
	int out_channels;
	int input_h;
	int input_w;
	int kernel_h;
	int kernel_w;
	int stride_h;
	int stride_w;
	int padding_h;
	int padding_w;
	int groups;
	string act;
	bool has_act;

	int output_h;
	int output_w;

	CudnnContext *context;

	cudnnConvolutionFwdAlgo_t conv_alg;
	cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnActivationDescriptor_t actiDesc;

	data_type *input_data;
	data_type *output_data;
	data_type *filter_data;
	data_type *bias_data;

	void init(int batch_size, int in_channels, int input_h, int input_w, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups, string act) {
		this->batch_size = batch_size;
		this->in_channels = in_channels;
		this->input_h = input_h;
		this->input_w = input_w;
		this->out_channels = out_channels;
		this->kernel_h = kernel_h;
		this->kernel_w = kernel_w;
		this->stride_h = stride_h;
		this->stride_w = stride_w;
		this->padding_h = padding_h;
		this->padding_w = padding_w;
		this->groups = groups;
		this->act = act;
		this->has_act = (act != "identity");
		this->output_h = 1 + (input_h - kernel_h + 2 * padding_h) / stride_h;
		this->output_w = 1 + (input_w - kernel_w + 2 * padding_w) / stride_w;
	}
	size_t get_filter_size() {
		return sizeof(data_type) * out_channels * (in_channels / groups) * kernel_h * kernel_w;
	}
	size_t get_bias_size() {
		return sizeof(bias_data_type) * out_channels;
	}
	void map(data_type *input_data, CudnnContext *context) {
		this->input_data = input_data;
		this->context = context;


		checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
		checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


		checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, cudnn_data_format, cudnn_data_type, batch_size, in_channels, input_h, input_w));
		checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor, cudnn_data_format, cudnn_bias_data_type, 1, out_channels, 1, 1));
		assert(in_channels % groups == 0);
		checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, cudnn_data_type, cudnn_data_format, out_channels, in_channels / groups, kernel_h, kernel_w));
		checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/, CUDNN_CROSS_CORRELATION, cudnn_conv_data_type));
		checkCUDNN(cudnnSetConvolutionMathType(convDesc, cudnn_math_type));
		checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, groups));
		int n, c, h, w;
		checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
		assert(n == batch_size);
		assert(c == out_channels);
		assert(h == output_h);
		assert(w == output_w);
		checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, cudnn_data_format, cudnn_data_type, n, c, h, w));
		if (has_act) {
			cudnnActivationMode_t act_mode;
			if(act == "relu") {
				act_mode = CUDNN_ACTIVATION_RELU;
			} else if(act == "tanh") {
				act_mode = CUDNN_ACTIVATION_TANH;
			} else if(act == "sigmoid") {
				act_mode = CUDNN_ACTIVATION_SIGMOID;
			} else {
				FatalError("Wrong activation mode " + act);
			}
			checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
			checkCUDNN(cudnnSetActivationDescriptor(actiDesc, act_mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
		}

		size_t filter_size = get_filter_size();
		size_t bias_size = get_bias_size();
		size_t output_size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;

		checkCUDA(cudaMalloc(&filter_data, filter_size));
		checkCUDA(cudaMalloc(&bias_data, bias_size));
		checkCUDA(cudaMalloc(&output_data, output_size));
		this->conv_alg = get_conv_alg(batch_size, in_channels, input_h, input_w, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups);
	}
	void forward() {
		assert(0 != batch_size);
		const float alpha = 1.0f;
		const float beta = 0.0f;
		if(has_act) {
#if defined(USE_TENSOR_CORE)
			checkCUDNN(cudnnConvolutionForward(
						context->dnn, &alpha, inputTensor, input_data, filterDesc, filter_data,
						convDesc, conv_alg, context->space, context->max_size,
						&beta, outputTensor, output_data));
			checkCUDNN(cudnnAddTensor(context->dnn, &alpha, biasTensor, bias_data, &alpha, outputTensor, output_data));
			checkCUDNN(cudnnActivationForward(context->dnn, actiDesc, &alpha, outputTensor, output_data, &beta, outputTensor, output_data));
#else
			checkCUDNN(cudnnConvolutionBiasActivationForward(
						context->dnn, &alpha, inputTensor, input_data, filterDesc, filter_data,
						convDesc,  conv_alg , context->space, context->max_size,
						&beta, outputTensor, output_data, biasTensor, bias_data, actiDesc,
						outputTensor, output_data));


#endif
		} else {
			checkCUDNN(cudnnConvolutionForward(
						context->dnn, &alpha, inputTensor, input_data, filterDesc, filter_data,
						convDesc, conv_alg, context->space, context->max_size,
						&beta, outputTensor, output_data));
			checkCUDNN(cudnnAddTensor(context->dnn, &alpha, biasTensor, bias_data, &alpha, outputTensor, output_data));
		}


	}
	void unmap() {
		checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
		checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
		if (has_act) {
			checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
		}

		checkCUDA(cudaFree(filter_data));
		checkCUDA(cudaFree(bias_data));
		checkCUDA(cudaFree(output_data));
	}
};

struct ActivationOP {
	static const int RELU = 1;
	static const int TANH = 2;
	static const int SIGMOID = 3;

	int batch_size;
	int in_channels;
	int input_h;
	int input_w;
	int out_channels;
	int act_type;
	bool inplace;

	int output_h;
	int output_w;


	cudnnTensorDescriptor_t inputTensor;
	cudnnActivationDescriptor_t actiDesc;

	data_type *input_data;
	data_type *output_data;
	CudnnContext *context;

	void init(int batch_size, int in_channels, int input_h, int input_w, int act_type, bool inplace) {
		this->batch_size = batch_size;
		this->in_channels = in_channels;
		this->input_h = input_h;
		this->input_w = input_w;
		this->act_type = act_type;
		this->inplace = inplace;
		this->out_channels = in_channels;
		this->output_h = input_h;
		this->output_w = input_w;
		this->context = nullptr;
	}
	void map(data_type *input_data, CudnnContext *context) {

		assert(input_data != nullptr);
		this->input_data = input_data;
		this->context = context;
		checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
		checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, cudnn_data_format, cudnn_data_type, batch_size, in_channels, input_h, input_w));
		checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
		cudnnActivationMode_t mode;
		switch (act_type) {
			case RELU:
				mode = CUDNN_ACTIVATION_RELU;
				break;
			case SIGMOID:
				mode = CUDNN_ACTIVATION_SIGMOID;
				break;
			case TANH:
				mode = CUDNN_ACTIVATION_TANH;
				break;
			default:
				assert(false);
		}
		checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
		if (!inplace) {
			size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
			checkCUDA(cudaMalloc(&output_data, size));
		} else {
			this->output_data = input_data;
		}
		assert(output_data != nullptr);
	}

	void unmap() {
		checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
		checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
		if (!inplace) {
			checkCUDA(cudaFree(output_data));
		}
	}

	void forward() {
		const float alpha = 1.0f;
		const float beta = 0.0f;
		assert(output_data != nullptr);
		assert(input_data != nullptr);
		checkCUDNN(cudnnActivationForward(context->dnn, actiDesc, &alpha, inputTensor, input_data, &beta, inputTensor, output_data));
	}
};

struct ElementOP {
	static const int MUL = 1;
	static const int ADD = 2;

	int batch_size;
	int channels;
	int h;
	int w;
	int op_type;

	cudnnTensorDescriptor_t inputTensor;
	cudnnOpTensorDescriptor_t opDesc;

	data_type *input_a;
	data_type *input_b;
	data_type *output_data;
	CudnnContext *context;

	void init(int batch_size, int channels, int h, int w, int op_type) {
		this->batch_size = batch_size;
		this->channels = channels;
		this->h = h;
		this->w = w;
		this->op_type = op_type;
		this->context = nullptr;
	}
	void map(data_type *input_a, data_type *input_b, CudnnContext *context) {
		this->input_a = input_a;
		this->input_b = input_b;
		size_t size = sizeof(data_type) * batch_size * channels * h * w;
		this->context = context;
		checkCUDA(cudaMalloc(&output_data, size));

		checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
		checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
		checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, cudnn_data_format, cudnn_data_type, batch_size, channels, h, w));
		cudnnOpTensorOp_t opType;
		if(op_type == MUL)
			opType = CUDNN_OP_TENSOR_MUL;
		else if(op_type == ADD)
			opType = CUDNN_OP_TENSOR_ADD;
		else
			FatalError("not supported elementwise op");
		checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, cudnn_data_type, CUDNN_NOT_PROPAGATE_NAN));
	}
	void forward() {
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(cudnnOpTensor(context->dnn, opDesc, &alpha, inputTensor, input_a, &alpha, inputTensor, input_b, &beta, inputTensor, output_data));
	}
	void unmap() {
		checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
		checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
		checkCUDA(cudaFree(output_data));
	}
};

struct PoolOP {
	static const int MAX_POOL = 1;
	static const int AVG_POOL = 2;
	std::string name;

	int batch_size;
	int in_channels;
	int input_h;
	int input_w;
	int kernel_h;
	int kernel_w;
	int stride_h;
	int stride_w;
	int padding_h;
	int padding_w;
	int out_channels;
	int output_h;
	int output_w;
	int pool_type;

	cudnnTensorDescriptor_t inputTensor, outputTensor;
	cudnnPoolingDescriptor_t poolDesc;

	CudnnContext *context;

	data_type *input_data;
	data_type *output_data;

	void init(int batch_size, int in_channels, int input_h, int input_w, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int pool_type) {
		this->batch_size = batch_size;
		this->in_channels = in_channels;
		this->input_h = input_h;
		this->input_w = input_w;
		this->kernel_h = kernel_h;
		this->kernel_w = kernel_w;
		this->stride_h = stride_h;
		this->stride_w = stride_w;
		this->padding_h = padding_h;
		this->padding_w = padding_w;
		this->pool_type = pool_type;

		this->out_channels = in_channels;
		this->output_h = 1 + (input_h - kernel_h + 2 * padding_h) / stride_h;
		this->output_w = 1 + (input_w - kernel_w + 2 * padding_w) / stride_w;
	}
	void map(data_type *input_data, CudnnContext *context) {
		this->input_data = input_data;
		this->context = context;
		checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
		checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
		// set descriptors
		checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, cudnn_data_format,
					cudnn_data_type, batch_size, in_channels, input_h, input_w));
		cudnnPoolingMode_t mode;
		if(pool_type == MAX_POOL) {
			mode = CUDNN_POOLING_MAX;
		} else if(pool_type == AVG_POOL) {
			mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
		} else {
			FatalError("unrecognized pooling type");
		}
		checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w));
		int n, c, h, w;
		checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, inputTensor, &n, &c, &h, &w));
		assert(n == batch_size);
		assert(c == out_channels);
		assert(h == output_h);
		assert(w == output_w);
		checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, cudnn_data_format, cudnn_data_type, n, c, h, w));

		size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
		checkCUDA(cudaMalloc(&output_data, size));
	}
	void forward() {
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(cudnnPoolingForward(context->dnn, poolDesc, &alpha, inputTensor, input_data, &beta, outputTensor, output_data));
	}


	void unmap() {
		checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
		checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
		checkCUDA(cudaFree(output_data));
	}

};


struct NodeBase {
	string name;
	int myrank;
	int op_rank;
	CudnnContext *context = nullptr;
	int batch_size;
	int out_channels;
	int output_h;
	int output_w;
	int test;
	data_type *output_data;
	int waitcount;


	void print_shape() {
		fprintf(stdout, "%s %d %d %d %d\n", name.c_str(), batch_size, out_channels, output_h, output_w);
	}
	void set_context(CudnnContext *context_) {
		this->context = context_;
	}


	virtual void map() = 0;
	virtual void forward() = 0;
	virtual void unmap() = 0;
	virtual void WaitIsend() = 0;
};


struct Placeholder: NodeBase {
	void init(string name, int batch_size, int out_channels, int output_h, int output_w) {
		this->name = name;
		this->batch_size = batch_size;
		this->out_channels = out_channels;
		this->output_h = output_h;
		this->output_w = output_w;

	}
	void init(int batch_size, const Json::Value &value_config) {
		this->name = value_config["name"].asString();
		this->batch_size = batch_size;
		this->out_channels = value_config["output_shape"][0].asInt();
		this->output_h = value_config["output_shape"][1].asInt();
		this->output_w = value_config["output_shape"][2].asInt();
		this->output_data = nullptr;
		this->op_rank = value_config["rank"].asInt();


	}
	void map() override {
		size_t size = sizeof(data_type) * this->batch_size * this->out_channels * this->output_h * this->output_w;
		checkCUDA(cudaMalloc(&output_data, size));

	}
	void forward() override {

	}
	void WaitIsend()override 
	{
		waitcount = 0;

	}

	void unmap() override {
		checkCUDA(cudaFree(output_data));
	}
};

struct Value: NodeBase {
	NodeBase *node;
	int begin;
	int end;
	int source_rank;
	int node_not_found;
	int tag;
	int waitcount;


	void init(const Json::Value &value_config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		this->op_rank = op_rank;
		this->source_rank = value_config[5].asInt();
		this->tag = value_config[6].asInt();
		MPI_Comm_rank(MPI_COMM_WORLD, &this->myrank);
		this->node_not_found = 0;
		if(this->source_rank != this->myrank) {			

			if(node_map.size() == 0)
			{
				this->node_not_found = 1;


			}			

			else if( node_map.find(value_config[0].asString()) != node_map.end()) {
				this->node = node_map.at(value_config[0].asString());
			}else{
				this->node_not_found = 1;
			}
			this->batch_size = batch_size; 
			this->begin = value_config[1].asInt();
			this->end = value_config[2].asInt();
			this->output_h = value_config[3].asInt();
			this->output_w = value_config[4].asInt();
			this->out_channels = end - begin;
			this->context = nullptr;
			this->output_data = nullptr;

		}
		else if(this->source_rank == this->myrank)
		{
			this->node = node_map.at(value_config[0].asString());
			this->begin = value_config[1].asInt();
			this->end = value_config[2].asInt();
			this->output_h = value_config[3].asInt();
			this->output_w = value_config[4].asInt();
			this->out_channels = end - begin;
			this->context = nullptr;
			this->output_data = nullptr;
			this->output_h = node->output_h;
			this->output_w = node->output_w;
		}
		this->batch_size = batch_size;


	} 
	void map() override {

		if(this->node_not_found == 0 )
		{


			if(batch_size == 1) {

				output_data = node->output_data + begin * node->output_h * node->output_w;
			} else {
				if (begin == 0 && end == node->out_channels) {
					output_data = node->output_data;
				} else {
					size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
					checkCUDA(cudaMalloc(&output_data, size));
				}
			}

			assert(output_data != nullptr);
		}
		if(this->node_not_found == 1 and this->source_rank != this->myrank)
		{
			size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
			checkCUDA(cudaMalloc(&output_data, size));

			assert(output_data != nullptr);
		}
		assert(output_data != nullptr);



	}



	void unmap() override {

		if(this->source_rank == this->myrank)
		{

			if(batch_size == 1) {
				return;
			} else {
				if (begin == 0 && end == node->out_channels)
					return;
				checkCUDA(cudaFree(output_data));
			}
		}
		if(this->node_not_found == 1 and this->source_rank != this->myrank){
			checkCUDA(cudaFree(output_data));
		}

	}
	void forward() override {

		int myrank;
		waitcount = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		if(this->op_rank < 0)
		{
			return;
		}
		else if(this->node_not_found == 1 and this->op_rank >= 0   and this->source_rank != this->myrank)
		{				
			MPI_Recv(output_data, batch_size * out_channels * output_h * output_w, MPI_FLOAT, this->source_rank, this->tag, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			return;
		}

		if(batch_size == 1 || (begin == 0 && end == node->out_channels))
			return;

		int num_blocks = batch_size;
		int src_blk_size = node->out_channels * node->output_h * node->output_w;
		int dst_blk_size = out_channels * output_h * output_w;
		int offset = begin * output_h * output_w;
		int n = num_blocks * dst_blk_size;

		assign_with_stride_dst_call(output_data, node->output_data + offset, n,dst_blk_size, src_blk_size,context->stream);
	}

	void WaitIsend()override 
	{
		waitcount = 0;

	}

};



struct Term: NodeBase {
	Value values[MAX_NUM_VALUES];
	int num_values;
	int source_rank;
	int flag;
	int num_terms_inInput;

	cudnnTensorDescriptor_t inputTensor;
	cudnnOpTensorDescriptor_t opDesc;
	data_type *local_data;



	void init(const Json::Value &term_config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		this->op_rank = op_rank;
		this->num_values = (int)term_config.size();
		this->num_terms_inInput = 0;

		this->context = context;
		this->flag = 0;
		for(Json::ArrayIndex i = 0; i < term_config.size(); i++) {			

			values[i].init(term_config[i], node_map, op_rank, batch_size);
			this->source_rank = values[i].source_rank;
		}


		this->batch_size = values[0].batch_size;
		this->out_channels = values[0].out_channels;
		this->output_h = values[0].output_h;
		this->output_w = values[0].output_w;
		for(int i = 1; i < num_values; i++) {
			assert(values[i].batch_size == batch_size);
			assert(values[i].out_channels == out_channels);
			assert(values[i].output_h == output_h);
			assert(values[i].output_w == output_w);
		}
		this->context = nullptr;
		this->output_data = nullptr;
		this->batch_size = batch_size;

	}
	void update_batch_size( int batch_size)
	{
		this->batch_size = batch_size;
		for(int i =0; i < this->num_values; i++) {
			values[i].batch_size = batch_size;
		}


	}
	void map() override {
		for(int i = 0; i < num_values; i++) {
			values[i].set_context(context);

			values[i].map();
		}
		if(num_values == 1) {
			this->output_data = values[0].output_data;

		} else {

			checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
			checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
			checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, cudnn_data_format, cudnn_data_type,
						values[0].batch_size, values[0].out_channels, values[0].output_h, values[0].output_w));

			checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_ADD, cudnn_data_type, CUDNN_NOT_PROPAGATE_NAN));

			size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
			checkCUDA(cudaMalloc(&output_data, size));
		}
		if(num_values > 1 and this->op_rank < 0)
		{
			size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
			checkCUDA(cudaMalloc(&local_data, size));
			cudaMemset(local_data, 0, size);
		}

	}
	void unmap() override {
		for(int i = 0; i < num_values; i++)
			values[i].unmap();
		if(num_values > 1) {
			checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
			checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
			checkCUDA(cudaFree(output_data));
		}
		if(num_values > 1 and this->op_rank < 0)
		{
			checkCUDA(cudaFree(local_data));
		}

	}

	void WaitIsend()override 
	{
		waitcount = 0;

		for(int i = 0; i < num_values; i++) {
			values[i].WaitIsend();
		}
		this->flag = 0;
	}



	void do_reduce(){

		assert(this->local_data != nullptr);

		for(int i = 0; i < num_values; i++) {
			if (values[i].node_not_found == 0)
			{
				assert(values[i].output_data != nullptr);
				const float alpha = 1.0;
				const float beta = 1.0;
				checkCUDNN(cudnnAddTensor(context->dnn, &alpha, inputTensor, values[i].output_data, &beta, inputTensor, this->local_data));


			}
		}



		MPI_Request request;
		MPI_Status status;
		int myrank;
		waitcount = 0;			
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);




		cudaStreamSynchronize(context->stream);


		if(myrank == 0){

			int tag = batch_size * out_channels * output_h * output_w;
			MPI_Isend(this->local_data , batch_size * out_channels * output_h * output_w , MPI_FLOAT, 1, tag, MPI_COMM_WORLD,&request);
			MPI_Recv(output_data, batch_size * out_channels * output_h * output_w, MPI_FLOAT, 1,tag, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		}else if (myrank == 1){

			int tag = batch_size * out_channels * output_h * output_w;
			MPI_Isend(this->local_data , batch_size * out_channels * output_h * output_w , MPI_FLOAT, 0, tag, MPI_COMM_WORLD,&request);
			MPI_Recv(output_data, batch_size * out_channels * output_h * output_w, MPI_FLOAT, 0,tag, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		}
		MPI_Wait(&request, &status);

		const float alpha = 1.0;
		const float beta = 1.0;
		checkCUDNN(cudnnAddTensor(context->dnn, &alpha, inputTensor, local_data, &beta, inputTensor, output_data));





	}

	void forward() override {
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		for(int i = 0; i < num_values; i++) {

			values[i].forward();
		}

		if(this->op_rank < 0 and num_values > 1 and this->num_terms_inInput > 1){
			int needreduce = 0;
			for(int i = 0; i < num_values; i++) {
				if (values[i].node_not_found == 1)
				{
					needreduce++;
				}
			}
			if (needreduce == num_values){

			}	
			else if(needreduce > 0){

				do_reduce();

				return;
			}


		}

		if (this->op_rank < 0 and this->num_terms_inInput == 1)
		{

			return;
		}



		if(num_values > 1) {
			int n = batch_size * out_channels * output_h * output_w;

			switch(num_values) {
				case 2:

					accumulate_sum_2_call(output_data, values[0].output_data, values[1].output_data, n, context->stream);
					break;
				case 3:

					accumulate_sum_3_call(output_data, values[0].output_data, values[1].output_data, values[2].output_data, n, context->stream);
					break;
				case 4:

					accumulate_sum_4_call(output_data, values[0].output_data, values[1].output_data, values[2].output_data, values[3].output_data, n, context->stream);
					break;
				case 5:

					accumulate_sum_5_call(output_data, values[0].output_data, values[1].output_data, values[2].output_data, values[3].output_data, values[4].output_data, n, context->stream);
					break;
				default:
					for(int i = 0; i < num_values; i++) {
						const float alpha = 1.0;
						const float beta = (i == 0 ? 0.0f : 1.0f);
						checkCUDNN(cudnnAddTensor(context->dnn, &alpha, inputTensor, values[i].output_data, &beta, inputTensor, output_data));
					}
			}
		}
	}
};

struct Input: NodeBase {
	int num_terms;
	Term terms[MAX_NUM_TERMS];
	int term_in_pro[MAX_PROCESS];
	int max_term_per_process;
	int num_process;
	int input_size;
	int receive_cnt[MAX_PROCESS][MAX_PROCESS];
	int receive_disp[MAX_PROCESS][MAX_PROCESS];
	int myrank;
	void init(const Json::Value &input_config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		this->op_rank = op_rank;
		this->batch_size = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &this->num_process);
		MPI_Comm_rank(MPI_COMM_WORLD, &this->myrank);	

		for(int i = 0; i < this->num_process;i++) {
			term_in_pro[i] = 0;
		}
		this->max_term_per_process = 0;	
		this->num_terms = (int)input_config.size();
		this->context = context;
		int sum = 0;
		this->input_size = input_config.size();
		for(Json::ArrayIndex i = 0; i < input_config.size(); i++) {
			terms[i].init(input_config[i], node_map, op_rank, batch_size);
			terms[i].num_terms_inInput = this->num_terms;
			sum += terms[i].out_channels;
			term_in_pro[terms[i].source_rank]++;

			if((max_term_per_process < term_in_pro[terms[i].source_rank]) and input_config.size() > 1) {
				this->max_term_per_process = term_in_pro[terms[i].source_rank];

			}

		}



		this->out_channels = sum;
		this->output_h = terms[0].output_h;
		this->output_w = terms[0].output_w;
		this->output_data = nullptr;
		this->context = nullptr;
		this->op_rank = op_rank;
		this->batch_size = batch_size;


	}
	void set_op_rank(int op_rank){
		this->op_rank = op_rank;
	}

	void populate_gather(int op_rank)
	{


		assert(this->output_data != nullptr);

		struct data_position_info{
			int reveive_cnt;
			int receive_disp;
			int term_id;
		}data_position[MAX_PROCESS];

		int disp[MAX_PROCESS] = {0};
		int cnt[MAX_PROCESS] = {0};
		MPI_Comm_rank(MPI_COMM_WORLD, &this->myrank);



		for(int i = 0; i< this->max_term_per_process ; i++) {
			for( int j = 0; j < this->num_process; j++)
			{
				data_position[j].receive_disp = 0;
				data_position[j].reveive_cnt = 0;
				int found_term[MAX_PROCESS] = {0};
				int offset = 0;
				for(int k = 0; k < this->input_size; k++){
					int src_blk_size = terms[k].out_channels * output_h * output_w*batch_size;
					if((this->terms[k].source_rank ==j) and  (this->terms[k].flag == 0))
					{
						this->terms[k].flag = 1;
						if(found_term[j] == 0){
							data_position[j].receive_disp = offset;
							data_position[j].reveive_cnt = src_blk_size;
							if(j == this->myrank){
								data_position[j].term_id = k;
							}	
							found_term[j] = 1;
							break;
						}
					}
					offset = offset + src_blk_size;
				}

			}
			for( int m = 0; m < this->num_process; m++)
			{

				disp[m] = data_position[m].receive_disp;
				cnt[m] = data_position[m].reveive_cnt;

			}


			if( op_rank == -3){
				int sendcount = 0;
				if (cnt[this->myrank] != 0){
					sendcount = terms[data_position[this->myrank].term_id].out_channels * terms[data_position[this->myrank].term_id].output_h * terms[data_position[this->myrank].term_id].output_w*terms[data_position[this->myrank].term_id].batch_size;
					MPI_Gatherv(terms[data_position[this->myrank].term_id].output_data, sendcount, MPI_FLOAT, this->output_data, cnt, disp, MPI_FLOAT,0, MPI_COMM_WORLD);
				}else{
					sendcount = 0;
					MPI_Gatherv("", sendcount, MPI_FLOAT, this->output_data, cnt, disp, MPI_FLOAT,0, MPI_COMM_WORLD);
				}



			}else if (op_rank == -2){
				int sendcount = 0;

				if (cnt[this->myrank] != 0){

					sendcount = terms[data_position[this->myrank].term_id].out_channels * terms[data_position[this->myrank].term_id].output_h * terms[data_position[this->myrank].term_id].output_w*terms[data_position[this->myrank].term_id].batch_size;

					MPI_Allgatherv(terms[data_position[this->myrank].term_id].output_data, sendcount, MPI_FLOAT, this->output_data, cnt, disp, MPI_FLOAT, MPI_COMM_WORLD);


				}else{
					sendcount = 0;
					MPI_Allgatherv("", sendcount, MPI_FLOAT, this->output_data, cnt, disp, MPI_FLOAT, MPI_COMM_WORLD);
				}

			}

		}			

	}




	void map() override {

		for(int i = 0; i < num_terms; i++) {


			terms[i].set_context(context);
			terms[i].map();
		}
		if(num_terms == 1) {
			this->output_data = terms[0].output_data;
		} else {

			if( op_rank == -2 and this->myrank ==0){
				size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
				checkCUDA(cudaMalloc(&output_data, size));
			}else{
				size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
				checkCUDA(cudaMalloc(&output_data, size));

			}
		}
	}
	void unmap() override {
		for(int i = 0; i < num_terms; i++)
			terms[i].unmap();
		if(num_terms > 1) {
			if (op_rank == -3 and this->myrank !=0){
			}else{
				checkCUDA(cudaFree(output_data));
			}

		}
	}

	void WaitIsend()override 
	{
		waitcount = 0;

		for(int i = 0; i < num_terms; i++){
			terms[i].WaitIsend();
		}
	}




	void forward() override {
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		assert(output_data != nullptr);

		for(int i = 0; i < num_terms; i++) {

			terms[i].forward();

		}

		if(this->op_rank == -1 and num_terms == 1) {
			checkCUDA(cudaDeviceSynchronize());
			MPI_Bcast(this->output_data, this->batch_size* out_channels * output_h * output_w, MPI_FLOAT, 0, MPI_COMM_WORLD);        	
			//checkCUDA(cudaDeviceSynchronize());

			return;
		}

		if((num_terms > 1) and (this->op_rank == this->myrank)) {

			int offset = 0;
			for(int i = 0; i < num_terms; i++) {


				int src_blk_size = terms[i].out_channels * output_h * output_w;
				int dst_blk_size = out_channels * output_h * output_w;
				int num_blocks = batch_size;
				int n = num_blocks * src_blk_size;

				assign_with_stride_src_call(output_data + offset, terms[i].output_data, n, dst_blk_size, src_blk_size,context->stream);
				offset += src_blk_size;

			}
		}



		if((num_terms == 1) and (terms[0].num_values > 1) and this->op_rank < -1) {
			terms[0].do_reduce();


		}

		if((num_terms > 1) and (this->op_rank == -2)) {

			checkCUDA(cudaDeviceSynchronize());
			populate_gather(-2);
			//checkCUDA(cudaDeviceSynchronize());
		}else if((num_terms > 1) and (this->op_rank == -3)) {

			checkCUDA(cudaDeviceSynchronize());
			populate_gather(-3);
			//checkCUDA(cudaDeviceSynchronize());

		}



	}
};


struct MPI_send_recv: NodeBase {
	RankInfo rInfo[MAX_NUM_TERMS];
	int rInfo_len;
	Input input;
	int com_type;
	MPI_Request request;
	MPI_Status  status;
	int waitcount;
	void init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		this->name = conv_config["name"].asString();
		this->com_type = conv_config["com_type"].asInt();

		this->input.init(conv_config["inputs"], node_map, op_rank,batch_size);
		this->batch_size = batch_size;
		this->out_channels = input.out_channels;
		this->output_h = input.output_h;
		this->output_w = input.output_w;
		this->context = nullptr;
		this->output_data = nullptr;
		const Json::Value &out_config = conv_config["outputs"];
		rInfo_len = out_config.size();

		for(int i = 0; i < rInfo_len; i++) {
			rInfo[i].rank = out_config[i][0].asInt();
			rInfo[i].tag = out_config[i][1].asInt();

		}



	}
	void map() override {

		input.set_context(context);
		input.map();

		this->output_data = input.output_data;

	}

	void send_data()
	{	
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


		if(this->com_type == 1) {

			for(int i = 0; i < rInfo_len; i++) {

				if( rInfo[i].rank != this->myrank){


					MPI_Ssend(output_data, batch_size * out_channels * output_h * output_w , MPI_FLOAT, rInfo[i].rank, rInfo[i].tag, MPI_COMM_WORLD);


				}
			}
		}

	}
	void wait()
	{

		MPI_Wait(&request, &status);
	}

	void WaitIsend()override 
	{
		if(this->com_type != 1)
		{

			waitcount = 0;
			input.WaitIsend();
		}
	}

	void forward() override {

		input.forward();
		send_data();

	}
	void unmap() override {
		input.unmap();
	}



};



struct Conv: NodeBase {
	Input input;
	ConvOP conv_op;
	RankInfo rInfo[MAX_NUM_TERMS];
	int rInfo_len;
	int myrank;
	MPI_Request request;
	MPI_Status  status;
	int waitcount;
	void init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		this->name = conv_config["name"].asString();
		this->input.init(conv_config["inputs"], node_map, op_rank,batch_size);
		conv_op.init(batch_size, input.out_channels, input.output_h, input.output_w,
				conv_config["out_channels"].asInt(),
				conv_config["kernel"][0].asInt(),
				conv_config["kernel"][1].asInt(),
				conv_config["stride"][0].asInt(),
				conv_config["stride"][1].asInt(),
				conv_config["padding"][0].asInt(),
				conv_config["padding"][1].asInt(),
				conv_config["groups"].asInt(),
				conv_config["act"].asString());
		this->batch_size = batch_size;
		this->out_channels = conv_op.out_channels;
		this->output_h = conv_op.output_h;
		this->output_w = conv_op.output_w;
		this->context = nullptr;
		this->output_data = nullptr;
		this->test = 5;
		const Json::Value &out_config = conv_config["outputs"];
		rInfo_len = out_config.size();

		for(int i = 0; i < rInfo_len; i++) {
			rInfo[i].rank = out_config[i][0].asInt();
			rInfo[i].tag = out_config[i][1].asInt();

		}

	}
	void map() override {
		input.set_context(context);
		input.map();


		conv_op.map(input.output_data, context);
		this->output_data = conv_op.output_data;
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		if( myrank == 1){
			assert(output_data != nullptr);               
		}



	}

	void send_data()
	{
		int myrank;
		waitcount = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		assert(output_data != nullptr);


		int isendcounter = 0;
		for(int i = 0; i < rInfo_len; i++) {
			if( rInfo[i].rank != myrank)
				isendcounter++;
		}

		if (isendcounter >= 1 and rInfo_len >= 1)
		{

			cudaStreamSynchronize(context->stream);
		}


		for(int i = 0; i < rInfo_len; i++) {

			if( rInfo[i].rank != myrank){															
				MPI_Isend(output_data, batch_size * out_channels * output_h * output_w , MPI_FLOAT, rInfo[i].rank, rInfo[i].tag, MPI_COMM_WORLD,&request);

				waitcount++;

			}

		}

	}
	void wait()
	{
		MPI_Wait(&request, &status);
	}

	void WaitIsend()override 
	{
		if(waitcount > 0){
			MPI_Wait(&request, &status);
		}
		waitcount = 0;					

		input.WaitIsend();


	}

	void forward() override {
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &this->myrank);

		input.forward();

		assert(output_data != nullptr);
		conv_op.forward();
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		send_data();

	}
	void unmap() override {

		input.unmap();
		conv_op.unmap();
	}

};

struct Element: NodeBase {
	Value inputs[2];
	ElementOP elem_op;

	NodeBase * get_input(Json::Value inputs_config, const std::map<string,NodeBase*> &node_map, int idx) {
		Json::Value cinput = inputs_config[0];
		assert(cinput.size() == 2);
		assert(0 <= idx && idx < 2);
		cinput = cinput[idx];
		NodeBase *node = node_map.at(cinput[0].asString());
		int begin = cinput[1].asInt();
		int end = cinput[2].asInt();
		assert(begin == 0 && end == node->out_channels);
		return node;
	}

	void init(const Json::Value &elem_config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		name = elem_config["name"].asString();
		Json::Value inputs_config = elem_config["inputs"];
		assert(inputs_config.size() == 1);
		int op_type;
		if(elem_config["op_type"].asString() == "mul") {
			op_type = ElementOP::MUL;
		} else if(elem_config["op_type"].asString() == "add") {
			op_type = ElementOP::ADD;
		} else {
			FatalError("");
		}
		this->inputs[0].init(inputs_config[0][0], node_map, op_rank, batch_size);
		this->inputs[1].init(inputs_config[0][1], node_map, op_rank, batch_size);
		elem_op.init(inputs[0].batch_size, inputs[0].out_channels, inputs[0].output_h, inputs[0].output_w, op_type);
		this->context = nullptr;
		this->output_data = nullptr;
		this->batch_size = batch_size;
		this->out_channels = inputs[0].out_channels;
		this->output_h = inputs[0].output_h;
		this->output_w = inputs[0].output_w;
	}
	void map() override {
		inputs[0].map();
		inputs[1].map();
		elem_op.map(inputs[0].output_data, inputs[1].output_data, this->context);
		this->output_data = elem_op.output_data;
	}
	void forward() override {
		inputs[0].forward();
		inputs[1].forward();
		elem_op.forward();
	}
	void WaitIsend()override 
	{


		inputs[0].WaitIsend();
		inputs[1].WaitIsend();
		waitcount = 0;

	}

	void unmap() override {
		inputs[0].unmap();
		inputs[1].unmap();
		elem_op.unmap();
	}

};


struct Graph;
struct Sequential: NodeBase {
	std::vector<NodeBase*> nodes;
	RankInfo rInfo[MAX_NUM_TERMS];
	int rInfo_len;
	int batch_size;
	MPI_Request request[MAX_STREAM];
	MPI_Status  status[MAX_STREAM];
	int waitcount;

	void init(const Json::Value &config, std::map<string,NodeBase*> &node_map, Graph *graph, int batch_size);


	void send_data()
	{
		int myrank;
		waitcount = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		assert(output_data != nullptr);

		int isendcounter = 0;
		for(int i = 0; i < rInfo_len; i++) {
			if( rInfo[i].rank != myrank)
				isendcounter++;
		}

		if (isendcounter >= 1 and rInfo_len >= 1)
		{

			cudaStreamSynchronize(context->stream);
		}
		for(int i = 0; i < rInfo_len; i++) {

			if( rInfo[i].rank != myrank){

				assert(output_data != nullptr);


				MPI_Isend(output_data, batch_size * out_channels * output_h * output_w , MPI_FLOAT, rInfo[i].rank, rInfo[i].tag, MPI_COMM_WORLD,&request[waitcount]);

				waitcount++;

			}
		}

	}

	void map() {
		for(auto node : nodes) {
			node->context = context;
			node->map();
		}
		this->output_data = nodes.back()->output_data;


	}

	void forward() {
		for(auto node : nodes) {
			node->forward();
		}
		send_data();
	}
	void WaitIsend()override 
	{
		if(waitcount > 0){

			MPI_Waitall(waitcount, request, status);
		}
		waitcount = 0;

		for(auto node : nodes) {
			node->WaitIsend();
		}

	}


	void unmap() {

		for(auto node : nodes) {
			node->unmap();
		}
	}
	void waitall()
	{

		MPI_Waitall(waitcount, request, status);
	}


};

struct Activation: NodeBase {
	Input input;
	ActivationOP act;

	void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		name = config["name"].asString();
		input.init(config["inputs"], node_map, op_rank, batch_size);
		int act_type;
		string act_str = config["act_type"].asString();
		if(act_str == "relu")
			act_type = ActivationOP::RELU;
		else if(act_str == "tanh")
			act_type = ActivationOP::TANH;
		else if(act_str == "sigmoid")
			act_type = ActivationOP::SIGMOID;
		else
			FatalError("unsupported activation mode " + act_str);
		assert(config.isMember("inplace"));
		act.init(batch_size, input.out_channels, input.output_h, input.output_w, act_type, config["inplace"].asBool());
		this->batch_size = act.batch_size;
		this->out_channels = act.out_channels;
		this->output_h = act.output_h;
		this->output_w = act.output_w;
		this->output_data = nullptr;
		this->context = nullptr;
	}
	void map() {
		input.set_context(context);
		input.map();
		act.map(input.output_data, context);
		this->output_data = act.output_data;
	}
	void forward() {
		input.forward();
		act.forward();

	}

	void WaitIsend()override 
	{

		waitcount = 0;
		input.WaitIsend();

	}

	void unmap() {
		input.unmap();
		act.unmap();
	}

};

struct Relu: NodeBase {
	Input input;
	ActivationOP act_relu;

	void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		name = config["name"].asString();
		input.init(config["inputs"], node_map, op_rank,batch_size);
		act_relu.init(batch_size, input.out_channels, input.output_h, input.output_w, ActivationOP::RELU, false);
		this->batch_size = act_relu.batch_size;
		this->out_channels = act_relu.out_channels;
		this->output_h = act_relu.output_h;
		this->output_w = act_relu.output_w;
		this->output_data = nullptr;
		this->context = nullptr;
	}
	void map() {
		input.set_context(context);
		input.map();
		act_relu.map(input.output_data, context);
		this->output_data = act_relu.output_data;
	}


	void forward() {
		input.forward();
		act_relu.forward();
	}

	void WaitIsend()override 
	{

		waitcount = 0;
		input.WaitIsend();

	}

	void unmap() {
		input.unmap();
		act_relu.unmap();
	}

};

struct Identity: NodeBase {
	Input input;
	RankInfo rInfo[MAX_NUM_TERMS];
	int rInfo_len;
	MPI_Request request[10];
	MPI_Status  status[10];
	int waitcount;

	void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		this->name = config["name"].asString();
		this->context = context;
		this->input.init(config["inputs"], node_map, op_rank, batch_size);
		this->batch_size = input.batch_size;
		this->out_channels = input.out_channels;
		this->output_h = input.output_h;
		this->output_w = input.output_w;

		const Json::Value &out_config = config["outputs"];
		rInfo_len = out_config.size();

		for(int i = 0; i < rInfo_len; i++) {
			rInfo[i].rank = out_config[i][0].asInt();
			rInfo[i].tag = out_config[i][1].asInt();

		}

	}
	void map() override {

		input.set_context(context);
		input.map();
		this->output_data = input.output_data;

	}
	void send_data()
	{
		int myrank;
		waitcount = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		assert(output_data != nullptr);

		int isendcounter = 0;
		for(int i = 0; i < rInfo_len; i++) {
			if( rInfo[i].rank != myrank)
				isendcounter++;
		}
		if (isendcounter >= 1 and rInfo_len >= 1)
		{

			cudaStreamSynchronize(context->stream);
		}

		for(int i = 0; i < rInfo_len; i++) {


			if( rInfo[i].rank != myrank){

				assert(output_data != nullptr);


				MPI_Isend(output_data, batch_size * out_channels * output_h * output_w , MPI_FLOAT, rInfo[i].rank, rInfo[i].tag, MPI_COMM_WORLD,&request[waitcount]);

				waitcount++;


			}
		}

	}


	void forward() override {

		input.forward();

		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		send_data();


	}
	void waitall()
	{

		MPI_Waitall(waitcount, request, status);
	}

	void WaitIsend()override 
	{
		if(waitcount > 0){
			MPI_Waitall(waitcount, request, status);
		}
		waitcount = 0;					

		input.WaitIsend();
	}

	void unmap() override {

		input.unmap();
	}

};

struct Pool: NodeBase {
	Input input;
	PoolOP pool_op;
	RankInfo rInfo[MAX_NUM_TERMS];
	int rInfo_len;
	int myrank;
	MPI_Request request[5];
	MPI_Status  status[5];
	int waitcount;

	void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map, int op_rank, int batch_size) {
		this->name = config["name"].asString();
		this->input.init(config["inputs"], node_map, op_rank, batch_size);
		int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;

		int pool_type;
		if(config["pool_type"].asString().find("max") != string::npos)
			pool_type = PoolOP::MAX_POOL;
		else
			pool_type = PoolOP::AVG_POOL;
		if(config["pool_type"].asString().find("global") != string::npos) {
			kernel_h = input.output_h;
			kernel_w = input.output_w;
			stride_h = 1;
			stride_w = 1;
			padding_h = 0;
			padding_w = 0;
		} else {
			kernel_h = config["kernel"][0].asInt();
			kernel_w = config["kernel"][1].asInt();
			stride_h = config["stride"][0].asInt();
			stride_w = config["stride"][1].asInt();
			padding_h = config["padding"][0].asInt();
			padding_w = config["padding"][1].asInt();
		}
		pool_op.init(input.batch_size, input.out_channels, input.output_h, input.output_w, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type);
		pool_op.name = name;

		this->batch_size = pool_op.batch_size;
		this->out_channels = pool_op.out_channels;
		this->output_h = pool_op.output_h;
		this->output_w = pool_op.output_w;
		const Json::Value &out_config = config["outputs"];
		rInfo_len = out_config.size();

		for(int i = 0; i < rInfo_len; i++) {
			rInfo[i].rank = out_config[i][0].asInt();
			rInfo[i].tag = out_config[i][1].asInt();

		}

	}

	void send_data()
	{
		int myrank;
		waitcount = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		assert(output_data != nullptr);


		int isendcounter = 0;
		for(int i = 0; i < rInfo_len; i++) {
			if( rInfo[i].rank != myrank)
				isendcounter++;
		}
		if (isendcounter >= 1 and rInfo_len >= 1)
		{

			cudaStreamSynchronize(context->stream);
		}

		for(int i = 0; i < rInfo_len; i++) {

			if( rInfo[i].rank != myrank){

				assert(output_data != nullptr);


				MPI_Isend(output_data, batch_size * out_channels * output_h * output_w , MPI_FLOAT, rInfo[i].rank, rInfo[i].tag, MPI_COMM_WORLD,&request[waitcount]);




				waitcount++;


			}
		}

	}
	void waitall()
	{


		MPI_Waitall(waitcount, request, status);
	}

	void WaitIsend()override 
	{
		if(waitcount > 0){
			MPI_Waitall(waitcount, request, status);
		}
		waitcount = 0;

		input.WaitIsend();

	}



	void map() override {
		input.set_context(context);
		input.map();

		pool_op.map(input.output_data, context);
		this->output_data = pool_op.output_data;
	}
	void forward() override {
		MPI_Comm_rank(MPI_COMM_WORLD, &this->myrank);
		if( myrank == 1){
			assert(output_data != nullptr);
		}


		input.forward();
		pool_op.forward();
		send_data();

	}



	void unmap() override {


		input.unmap();
		pool_op.unmap();
	}
};

struct Graph {
	int num_inputs;
	Placeholder inputs[MAX_NUM_NODES];
	int num_mpi_sr;
	MPI_send_recv mpi_sr[MAX_NUM_NODES];
	int num_convs;
	Conv convs[MAX_NUM_NODES];
	int num_pools;
	Pool pools[MAX_NUM_NODES];
	int num_idents;
	Identity idents[MAX_NUM_NODES];
	int num_elem;
	Element elems[MAX_NUM_NODES];
	int num_relus;
	Relu relus[MAX_NUM_NODES];
	int num_acts;
	Activation acts[MAX_NUM_NODES];
	int num_sequential;
	Sequential sequential[MAX_NUM_NODES];

	int num_stages;
	int stage_num_seq[MAX_NUM_NODES];
	int stage_seq_num_op[MAX_NUM_NODES][MAX_NUM_GROUPS];
	NodeBase* stages[MAX_NUM_NODES][MAX_NUM_GROUPS][MAX_GROUP_SIZE];

	void reset() {
		num_inputs = num_convs = num_pools = num_idents = num_relus = num_elem = num_acts = num_sequential = num_stages = 0;
	}

	NodeBase *add_node(Json::Value node_config, std::map<string, NodeBase*> &node_map, int myrank, int batch_size) {
		NodeBase *nb = nullptr;

		if(node_config["type"].asString() == "conv") {

			assert(num_convs < MAX_NUM_NODES);
			convs[num_convs].init(node_config, node_map, node_config["rank"].asInt(), batch_size);
			nb = convs + num_convs++;
		}else if(node_config["type"].asString() == "mpi_sr") {
			assert(num_mpi_sr < MAX_NUM_NODES);
			mpi_sr[num_mpi_sr].init(node_config, node_map, node_config["rank"].asInt(), batch_size);
			nb = mpi_sr + num_mpi_sr++;
		} else if(node_config["type"].asString() == "pool") {
			assert(num_pools < MAX_NUM_NODES);
			pools[num_pools].init(node_config, node_map, node_config["rank"].asInt(), batch_size);
			nb = pools + num_pools++;
		} else if(node_config["type"].asString() == "identity") {
			assert(num_idents < MAX_NUM_NODES);
			idents[num_idents].init(node_config, node_map, node_config["rank"].asInt(), batch_size);
			nb = idents + num_idents++;
		} else if(node_config["type"].asString() == "activation") {
			assert(num_idents < MAX_NUM_NODES);
			acts[num_acts].init(node_config, node_map, node_config["rank"].asInt(), batch_size);
			nb = acts + num_acts++;
		} else if(node_config["type"].asString() == "element") {
			assert(num_elem < MAX_NUM_NODES);
			elems[num_elem].init(node_config, node_map, node_config["rank"].asInt(), batch_size);
			nb = elems + num_elem++;
		} else if(node_config["type"].asString() == "relu") {
			assert(num_relus < MAX_NUM_NODES);
			relus[num_relus].init(node_config, node_map, node_config["rank"].asInt(), batch_size);
			nb = relus + num_relus++;
		} else if(node_config["type"].asString() == "sequential") {
			assert(num_sequential < MAX_NUM_NODES);
			sequential[num_sequential].init(node_config, node_map, this, batch_size);
			nb = sequential + num_sequential++;
		} else {
			FatalError("unsupported type " + node_config["type"].asString());
		}
		return nb;
	}
	void init_graph(int batch_size, Json::Value graph_config, int myrank) {
		std::map<string, NodeBase*> node_map;
		reset();
		num_inputs = 1;

		Json::Value input = graph_config["input"];
		if(input["rank"].asInt() == myrank or input["rank"].asInt() == -1)
		{
			inputs[0].init(batch_size, graph_config["input"]);
			node_map[inputs[0].name] = &inputs[0];
		}


		Json::Value blocks = graph_config["blocks"];
		for(Json::Value block : graph_config["blocks"]) {
			NodeBase *nb;

			for(const Json::Value& node : block["inner_nodes"]) {
				if(node["rank"].asInt() == myrank)
				{
					nb = add_node(node, node_map, myrank, batch_size);
					node_map[nb->name] = nb;

				}
			}

			Json::Value& node = block["exit_node"];
			if(node["rank"].asInt() == myrank or  node["rank"].asInt() == -2 or node["rank"].asInt() == -3)
			{

				nb = add_node(block["exit_node"], node_map, myrank, batch_size);
				node_map[nb->name] = nb;



			}
			else if( node["rank"].asInt() == -1)
			{
				nb = add_node(block["exit_node"], node_map, myrank, batch_size);

				node_map[nb->name] = nb;

			}

			for(const Json::Value& mpi_config : block["stages"]) {

				if(mpi_config["rank"].asInt() == myrank)
				{

					const Json::Value& stage_config_dic = mpi_config["stages"];

					const Json::Value&  block_stage_config = stage_config_dic;

					for(int x = 0; x < block_stage_config.size(); x++) {
						const Json::Value&  stage_config = block_stage_config[x];

						stage_num_seq[num_stages] = stage_config.size();

						for(Json::ArrayIndex i = 0; i < stage_config.size(); i++) {
							const Json::Value& seq_config = stage_config[i];

							stage_seq_num_op[num_stages][i] = seq_config.size();

							for(Json::ArrayIndex j = 0; j < seq_config.size(); j++) {
								NodeBase *pnode = node_map.at(seq_config[j].asString());
								stages[num_stages][i][j] = pnode;

								pnode->set_context(contexts[i]);

							}
						}
						num_stages++;

					}
				} 
			}
		}
	}
	void init_block(int batch_size, Json::Value block_config) {
		std::map<string, NodeBase*> node_map;
		reset();
		Json::Value enter_node = block_config["enter_node"];
		Json::Value input_shape = enter_node["output_shape"];
		num_inputs = 1;
		inputs[0].init(enter_node["name"].asString(), batch_size, input_shape[0].asInt(), input_shape[1].asInt(), input_shape[2].asInt());
		node_map[enter_node["name"].asString()] = &inputs[0];

		NodeBase *nb;
		for(const Json::Value& node : block_config["inner_nodes"]) {
			nb = add_node(node, node_map, 0, batch_size);
			node_map[nb->name] = nb;
		}
		nb = add_node(block_config["exit_node"], node_map, 0,batch_size);
		node_map[nb->name] = nb;

		for(const Json::Value& stage_config : block_config["stages"]) {
			stage_num_seq[num_stages] = stage_config.size();
			for(Json::ArrayIndex i = 0; i < stage_config.size(); i++) {
				const Json::Value& seq_config = stage_config[i];
				stage_seq_num_op[num_stages][i] = seq_config.size();
				for(Json::ArrayIndex j = 0; j < seq_config.size(); j++) {
					NodeBase *pnode = node_map.at(stage_config[i][j].asString());
					stages[num_stages][i][j] = pnode;
					pnode->set_context(contexts[i]);
				}
			}
			num_stages++;
		}
	}
	void init_stage(int batch_size, Json::Value stage_config, Json::Value input_config) {
		std::map<string, NodeBase*> node_map;
		reset();
		for(Json::Value::const_iterator it = input_config.begin(); it != input_config.end(); it++) {
			string name = it.key().asString();
			int out_channels = (*it)[0].asInt();
			int output_h = (*it)[1].asInt();
			int outupt_w = (*it)[2].asInt();
			inputs[num_inputs].init(name, batch_size, out_channels, output_h, outupt_w);
			inputs[num_inputs].set_context(contexts[0]);
			node_map[name] = inputs + num_inputs;
			num_inputs++;
		}

		stage_num_seq[num_stages] = stage_config.size();
		for(Json::ArrayIndex i = 0; i < stage_config.size(); i++) {
			const Json::Value& seq_config = stage_config[i];
			stage_seq_num_op[num_stages][i] = seq_config.size();
			for(Json::ArrayIndex j = 0; j < seq_config.size(); j++) {
				Json::Value node_config = stage_config[i][j];
				NodeBase *pnode = add_node(node_config, node_map,0, batch_size);
				node_map[pnode->name] = pnode;
				stages[num_stages][i][j] = pnode;
				pnode->set_context(contexts[i]);

			}
		}

		num_stages++;

	}
	void set_input(data_type *input_data /*in host memory space*/) {
		size_t size = sizeof(data_type) * inputs[0].batch_size * inputs[0].out_channels * inputs[0].output_h * inputs[0].output_w;
		data_type *input = (data_type*)malloc(size);
		for( int i = 0 ; i < size/sizeof(data_type); i++)
		{
			input[i] = .1;

		}
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		cudaSetDevice(myrank);
		checkCUDA(cudaMemcpy(inputs[0].output_data, input, size, cudaMemcpyKind::cudaMemcpyHostToDevice));


		print("input data", inputs[0].output_data, size/sizeof(data_type));
	}
	void get_output(data_type *output_data) {
		assert(stage_num_seq[num_stages-1] == 1);
		NodeBase *nb = stages[num_stages-1][0][stage_seq_num_op[num_stages-1][0]-1];
		size_t size = sizeof(data_type) * nb->batch_size * nb->out_channels * nb->output_h * nb->output_w;
		checkCUDA(cudaMemcpy(output_data, nb->output_data, size, cudaMemcpyDeviceToHost));
		printf("size %d\n", size);

	}

	void set_conv_weights(char *name, data_type *weight, data_type *bias) {
		Conv *conv = nullptr;
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		for(int i = 0; i < num_convs; i++) {

			conv = convs + i;

			if(conv == nullptr) {

				return;
			}
			if(weight != nullptr){

				data_type *input = (data_type*)malloc(conv->conv_op.get_filter_size() );
				for( int i = 0 ; i < conv->conv_op.get_filter_size()/sizeof(data_type); i++)
				{
					input[i] = 0.001;
				}

				checkCUDA(cudaMemcpy(conv->conv_op.filter_data, input, conv->conv_op.get_filter_size(), cudaMemcpyHostToDevice));

			}
			else
				assert(false);
			if(bias != nullptr) {
				data_type *input = (data_type*)malloc(conv->conv_op.get_bias_size() );
				for( int i = 0 ; i < conv->conv_op.get_bias_size()/sizeof(data_type); i++)
				{
					input[i] = 0.001;
				}

				checkCUDA(cudaMemcpy(conv->conv_op.bias_data, input, conv->conv_op.get_bias_size(), cudaMemcpyHostToDevice));

			}
			else
				assert(false);
		}

	}
	void map() {
		for(int i = 0; i < num_inputs; i++)
			inputs[i].map();
		for(int i = 0; i < num_stages ; i++) {
			for(int j = 0; j < stage_num_seq[i]; j++)
				for(int k = 0; k < stage_seq_num_op[i][j]; k++){

					stages[i][j][k]->map();
				}
			if(stage_num_seq[i] > MAX_NUM_GROUPS) {
				fprintf(stderr, "The number of nodes in stage %d exceed the number of available streams %d\n", (int)stage_num_seq[i], MAX_NUM_GROUPS);
				assert(stage_num_seq[i] <= MAX_NUM_GROUPS);
			}
		}

	}


	void warm_up() {
		for(int i = 0; i < num_stages ; i++) {
			int stage_size = stage_num_seq[i];
			for (int j = 0; j < stage_size; j++) {
				int seq_length = stage_seq_num_op[i][j];
				for (int k = 0; k < seq_length ; k++)
					stages[i][j][k]->forward();
			}
		}
	}


	void write_output(const char * config_filename, data_type *device_data, int cnt) {
		size_t size = sizeof(data_type) * cnt;
		auto * host_data = (data_type*)malloc(size);
		checkCUDA(cudaMemcpy(host_data, device_data, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));

		std::ofstream fout(config_filename, std::ios_base::out | std::ios_base::trunc);
		for(int i = 0; i < cnt; i++)
			fout << host_data[i] << " ";
		fout << std::endl;
		fout.close();
		free(host_data);
	}	

	void forward(bool record_events, cudaEvent_t *events) {
		int x,y,z;
		for(int i = 0; i < num_stages; i++) {
			int stage_size = stage_num_seq[i];

			for (int j = 0; j < stage_size; j++) {
				int seq_length = stage_seq_num_op[i][j];

				for (int k = 0; k < seq_length; k++){

					stages[i][j][k]->forward();



				}
			}


			if(record_events) {
				if(i + 1 < num_stages)
					checkCUDA(cudaEventRecord(events[i], 0)); // profile stage latency
			} else {
				if(i + 1 == num_stages || (stage_size == 1 && stage_num_seq[i+1] == 1))
					continue;
				else
					checkCUDA(cudaDeviceSynchronize());
			}

		}

		for(int i = 0; i < num_stages; i++) {
			for(int j = 0; j < stage_num_seq[i]; j++)
				for(int k = 0; k < stage_seq_num_op[i][j]; k++) {

					stages[i][j][k]->WaitIsend();

				}
		}


	}


	void unmap() {
		for(int i = 0; i < num_inputs; i++)
			inputs[i].unmap();
		for(int i = 0; i < num_stages; i++) {
			for(int j = 0; j < stage_num_seq[i]; j++)
				for(int k = 0; k < stage_seq_num_op[i][j]; k++) {
					stages[i][j][k]->unmap();
				}
		}
	}



	void measure_stage_latency(int warmup, int number, int repeat, float* results, bool profile_stage_latency, float *stage_results) {
		assert(!profile_stage_latency);
		cudaEvent_t start, end;

		checkCUDA(cudaEventCreate(&start));
		checkCUDA(cudaEventCreate(&end));
		map();

		// warmup

		for(int i = 0; i < warmup; i++)
			warm_up();

		// measure latency

		for(int i = 0; i < repeat; i++) {
			checkCUDA(cudaDeviceSynchronize());
			checkCUDA(cudaEventRecord(start, 0));
			for (int t = 0; t < number; t++) {

				forward(false, 0);
				if(stage_num_seq[0] != 1)
                    		checkCUDA(cudaDeviceSynchronize());
				
			}
			checkCUDA(cudaEventRecord(end, 0));
			checkCUDA(cudaDeviceSynchronize());
			float latency = 0.0f;
			checkCUDA(cudaEventElapsedTime(&latency, start, end));
			results[i] = latency / float(number);
		}

		if(profile_stage_latency) {
			for(int i = 0; i < repeat; i++)
				for(int j = 0; j < num_stages; j++)
					stage_results[i * num_stages + j] /= (float) number;
		}

		// release resources

		unmap();
	}
	void measure_latency(int warmup, int number, int repeat, float* results, bool profile_stage_latency, float *stage_results) {
		cudaEvent_t start, end, events[MAX_NUM_NODES];

		// allocate resources
		if(profile_stage_latency) {
			for (int i = 0; i < num_stages; i++)
				checkCUDA(cudaEventCreate(events + i));
		}
		checkCUDA(cudaEventCreate(&start));
		checkCUDA(cudaEventCreate(&end));
		map();




		// warmup
		for(int i = 0; i < warmup; i++)
			forward(profile_stage_latency, events);

		// measure latency
		for(int i = 0; i < repeat; i++) {
			results[i] = 0.0;
			for(int j = 0; j < num_stages; j++)
				stage_results[i * num_stages + j] = 0.0f;
			for (int t = 0; t < number; t++) {
				checkCUDA(cudaDeviceSynchronize());
				checkCUDA(cudaEventRecord(start, 0));
				forward(profile_stage_latency, events);
				

				checkCUDA(cudaEventRecord(end, 0));
				checkCUDA(cudaDeviceSynchronize());
				float latency;
				
				if(profile_stage_latency) {
					for (int j = 0; j < num_stages; j++) {
						checkCUDA(cudaEventElapsedTime(&latency, j == 0 ? start : events[j - 1], j == num_stages - 1 ? end : events[j]));
						stage_results[i * num_stages + j] += latency;
					}
				}
    				
				checkCUDA(cudaEventElapsedTime(&latency, start, end));
				results[i] += latency;
			}
			results[i] /= float(number);
		}

		if(profile_stage_latency) {
			for(int i = 0; i < repeat; i++)
				for(int j = 0; j < num_stages; j++)
					stage_results[i * num_stages + j] /= (float) number;
		}

		// release resources

		unmap();
		if(profile_stage_latency) {
			for (int i = 0; i < num_stages; i++)
				checkCUDA(cudaEventDestroy(events[i]));
		}
	}
};
void Sequential::init(const Json::Value &config, std::map<string,NodeBase*> &node_map, Graph *graph,int batch_size) {
	name = config["name"].asString();
	nodes.clear();

	auto nodes_config = config["nodes"];
	int n = nodes_config.size();
	for(int i = 0; i < n; i++) {

		NodeBase *node = graph->add_node(nodes_config[i], node_map, config["rank"].asInt(),batch_size);
		nodes.push_back(node);
		node_map[node->name] = node;
	}
	const Json::Value &out_config = config["outputs"];
	rInfo_len = out_config.size();

	for(int i = 0; i < rInfo_len; i++) {
		rInfo[i].rank = out_config[i][0].asInt();
		rInfo[i].tag = out_config[i][1].asInt();

	}


	NodeBase *tail = nodes.back();

	this->batch_size = tail->batch_size;

	out_channels = tail->out_channels;
	output_h = tail->output_h;
	output_w = tail->output_w;
	context = nullptr;
	output_data = nullptr;
}

Graph graph;


extern void graph_latency(const char *graph_json, int batch_size, int warmup, int number, int repeat, int profile_stage_latency, float *results, float *stage_results, int myrank) {


	cudaSetDevice(myrank);
	CudnnContext context[MAX_STREAM] = {CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),
		CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank)

	};
	for (int i = 0; i < MAX_NUM_GROUPS; i++)
	{
		contexts[i] = &context[i];
	}

	stringstream in(graph_json);

	Json::Value graph_config;

	std::ifstream json_file(in.str(), std::ifstream::binary);
	json_file >> graph_config;


	graph.init_graph(batch_size, graph_config, myrank);

	profile_stage_latency = 0;
	graph.measure_latency(warmup, number, repeat, results, profile_stage_latency, stage_results);



	json_file.close();

}


extern void stage_latency(const char *stage_json, const char *input_json, int batch_size, int warmup, int number, int repeat, int profile_stage_latency, float *results, float *stage_results) {

	int size, myrank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if (myrank == 0){
		cudaSetDevice(myrank);
		Json::Value stage_config = json_from_cstr(stage_json);
		Json::Value input_config = json_from_cstr(input_json);


		cudaSetDevice(myrank);
		CudnnContext context[MAX_STREAM] = {CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),
			CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank),CudnnContext(myrank)
		};
		for (int i = 0; i < MAX_NUM_GROUPS; i++)
		{
			contexts[i] = &context[i];
		}
		graph.init_stage(batch_size, stage_config, input_config);

		graph.measure_stage_latency(warmup, number, repeat, results, 0, stage_results);

	}
}




