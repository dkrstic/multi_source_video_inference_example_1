#include "onnx2tensorrt.hpp"

using namespace std;

Logger gLogger;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims &dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

// get classes names
std::vector<std::string> getClassNames(const std::string &imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

// preprocessing stage ------------------------------------------------------------------------------------------------
void preprocessFrame(const cv::Mat &frame, float *gpu_input, const nvinfer1::Dims &dims)
{
    if (frame.empty())
    {
        std::cerr << "Input frame load failed\n";
        return;
    }
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);
    // cout << "gpu_frame: " << gpu_frame.rows << ", " << gpu_frame.cols << ", " << gpu_frame.channels() << endl;
    auto input_width = dims.d[3];
    auto input_height = dims.d[2];
    auto channels = dims.d[1];
    auto input_size = cv::Size(input_width, input_height);
    // cout << "input width: " << input_width << ", input height: " << input_height << ", channels: " << channels << endl;
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
    // cout << "flt_image: " << flt_image.rows << ", " << flt_image.cols << ", " << flt_image.size() << endl;
}

// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
void parseOnnxModel(const std::string &model_path, TRTUniquePtr<nvinfer1::ICudaEngine> &engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext> &context)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    // TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(1)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible

    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // we have only one image in batch
    builder->setMaxBatchSize(1);
    nvinfer1::IHostMemory * serialized_model = builder->buildSerializedNetwork(*network, *config);
    // generate TensorRT engine optimized for the target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
    std::ofstream ofs(trt_engine_filename, std::ios::out | std::ios::binary);
    ofs.write(static_cast<char*>(serialized_model->data()), serialized_model->size());

}
