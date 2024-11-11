#ifndef __ONNX2TENSORRT_HPP_INCLUDED__   
#define __ONNX2TENSORRT_HPP_INCLUDED__ 

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

const std::string trt_engine_filename = "vdd.engine";
// destroy TensorRT objects if something goes wrong
// struct TRTDestroy;

class Logger : public nvinfer1::ILogger
{

public:
    void log(Severity severity, char const *msg) noexcept override
    {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
        {
            std::cout << msg << "\n";
        }
    }
};

struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};


template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims);

// get classes names
std::vector<std::string> getClassNames(const std::string& imagenet_classes);

// preprocessing stage ------------------------------------------------------------------------------------------------
void preprocessFrame(const cv::Mat& frame, float* gpu_input, const nvinfer1::Dims& dims);

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context);

#endif