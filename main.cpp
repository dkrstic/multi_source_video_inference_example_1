#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <ctime>
#include <filesystem>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/cudacodec.hpp>
#include <NvInfer.h>
#include "onnx2tensorrt.hpp"
#include "frames_receiver.hpp"

using namespace std;
using namespace cv;

float sigmoid(const float &m1)
{
    float output;
    output = 1 / (1 + exp(-m1));
    return output;
}

vector<float> sigmoid(const vector<float> &m1)
{

    /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */

    const unsigned long VECTOR_SIZE = m1.size();
    vector<float> output(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i)
    {
        output[i] = 1 / (1 + exp(-m1[i]));
    }

    return output;
}

void display_frame(string &label)
{
    cv::putText(frame_copy, label, Point(50, 40), FONT_FACE, FONT_SCALE, Scalar(100, 50, 150));
    cv::imshow("Output", frame);
    cv::imshow("Result", frame_copy);
}

void save_defected_frame(const string &defects_store_path, cv::Mat &img)
{
    cv::putText(frame, "Defected", Point(25, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    std::cout << "Stream index string: " << stream_idx_str << endl;
    std::cout << "Defected score: " << defect_score << endl;
    std::cout << "Defected score path: " << defects_store_path << endl;
    resize(img, frame_save, Size(INPUT_WIDTH / 2, INPUT_HEIGHT / 2));
    cv::imwrite(defects_store_path, frame_save);
}

void img_saver_consumer(int id)
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(mtx);
        con_var.wait(lock, []
                     { return !img_queue.empty() || done_producing; }); // Wait for data or done signal

        // Check whether the production is done and the queue is empty
        if (img_queue.empty() && done_producing)
        {
            break; // Exit the loop if no more items to consume
        }

        if (!img_queue.empty())
        {
            std::pair<std::string_view, cv::Mat> item = img_queue.front(); // Get the item from the queue
            img_queue.pop();                                               // Remove it from the queue
            cv::Mat img = item.second;
            save_defected_frame(defects_store_path, img);
            lock.unlock(); // Release lock before processing
        }
    }
}

int main(int argc, char *argv[])
{

    string dst_address = (argv[1] != nullptr) ? argv[1] : DST_ADDRESS;
    string dst_port = (argv[2] != nullptr && argv[2] != nullptr) ? argv[2] : DST_PORT;
    string mode = (argv[3] != nullptr && argv[3] != nullptr) ? argv[3] : "prod";
    string defects_store_root = (argv[4] != nullptr && argv[4] != nullptr) ? argv[4] : DEFAULT_DST_PATH;

    std::cout << "OpenCV version : " << CV_VERSION << endl;

    std::cout << "dst_address: " << dst_address << endl;

    zmq_hostname = "tcp://" + dst_address + ":" + dst_port;

    int sndhwm = 1;

    sock.bind(zmq_hostname);
    std::cout << "zmq socket: " << zmq_hostname << " bound" << endl;

    try
    {
        int port_tr = std::stoi(dst_port);
        port_tr++;

        string socket_tr_addr = "tcp://" + dst_address + ":" + to_string(port_tr);
        zmq_setsockopt(sock_tr, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        sock_tr.connect(socket_tr_addr);
        std::cout << "Transmission socket address: " << socket_tr_addr << " is bound" << endl;
    }
    catch (std::invalid_argument const &e)
    {
        std::cout << "Bad input: dst port must be of int type" << std::endl;
        return -1;
    }
    catch (std::out_of_range const &e)
    {
        std::cout << "Integer overflow: dst port must be of int type in range" << std::endl;
        return -1;
    }

    // initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    Logger engLogger;
    TRTUniquePtr<nvinfer1::IRuntime> m_runtime{nvinfer1::createInferRuntime(engLogger)};
    if (filesystem::exists(trt_engine_filename))
    {
        ifstream engine_file(trt_engine_filename, ios::binary);
        if (!engine_file)
        {
            std::cout << "Unable to open file for" << trt_engine_filename << " reading\n";
        }
        else
        {
            auto engine_size = std::filesystem::file_size(trt_engine_filename);
            std::vector<char> engine_data(engine_size);
            engine_file.read(engine_data.data(), engine_size);

            engine = static_cast<TRTUniquePtr<nvinfer1::ICudaEngine>>(m_runtime->deserializeCudaEngine(engine_data.data(), engine_size));
            context.reset(engine->createExecutionContext());
        }
    }
    else
    {
        parseOnnxModel(model_path, engine, context);
    }

    // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims;               // we expect only one input
    std::vector<nvinfer1::Dims> output_dims;              // and one output
    std::vector<void *> buffers(engine->getNbBindings()); // buffers for input and output data

    auto in_dims = nvinfer1::Dims4{1, 3, 224 * 2, 224 * 3};
    auto out_dims = nvinfer1::Dims4{1, 2, 2, 3};

    std::cout << "Number of Bindings: " << engine->getNbBindings() << endl;
    std::cout << "Number of IOTensors: " << engine->getNbIOTensors() << endl;

    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        std::cout << "Binding Size: " << binding_size << endl;
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            // input_dims.emplace_back(engine->getBindingDimensions(i));
            input_dims.emplace_back(in_dims);
            std::cout << "Binding input dimensions: " << *engine->getBindingDimensions(i).d << endl;
        }
        else
        {
            // output_dims.emplace_back(engine->getBindingDimensions(i));
            output_dims.emplace_back(out_dims);
            std::cout << "Binding output dimensions: " << *engine->getBindingDimensions(i).d << endl;
        }
    }
    cudaMalloc(&buffers[0], 224 * 224 * 6 * 3 * 4);
    cudaMalloc(&buffers[1], 2 * 2 * 3 * 4);
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }
    else
    {
        std::cout << "Input dimensions: " << *input_dims.data()->d << endl;
        std::cout << "Input dimensions: " << *(input_dims.data()->d + 1) << endl;
        std::cout << "Input dimensions: " << *(input_dims.data()->d + 2) << endl;
        std::cout << "Input dimensions: " << *(input_dims.data()->d + 3) << endl;
        std::cout << "Output dimensions: " << *(output_dims.data()->d) << endl;
        std::cout << "Output dimensions: " << *(output_dims.data()->d + 1) << endl;
        std::cout << "Output dimensions: " << *(output_dims.data()->d + 2) << endl;
        std::cout << "Output dimensions: " << *(output_dims.data()->d + 3) << endl;
    }

    int frames_no = 0;
    int start_tick;
    int end_tick;
    vector<double> layers_times;
    double freq;
    double t;
    double t_inf;
    int msg_size;
    string label;
    vector<float> timer;

    bool is_play{true};
    cv::cuda::GpuMat out_gpu_mat(2, 3, CV_32F, buffers[1]);
    cv::Mat out_mat;
    cv::Scalar out_mat_sum;
    string video_stream_str;

    const int num_consumers = 10;
    std::vector<std::thread> img_saver;

    for (int i = 0; i < num_consumers; ++i)
    {
        img_saver.emplace_back(img_saver_consumer, i);
    }

    while (true)
    {

        auto res = sock.recv(msg, zmq::recv_flags::none);
        frame.data = static_cast<uchar *>(msg.data());
        // std::cout << "Message size: " << msg.size() << endl;

        img_data = frame.data + msg.size();

        for (int i = 0; i < STREAM_IDX_SIZE; ++i)
        {
            stream_idx_str[i] = *(img_data - STREAM_IDX_SIZE + i);
        }
        stream_idx = atoi(stream_idx_str);
        // std::cout << "Stream index string: " << stream_idx_str << endl;

        frames_no++;
        frame_copy = frame.clone();

        start_tick = getTickCount();

        resize(frame_copy, frame_copy, Size(INPUT_WIDTH, INPUT_HEIGHT));

        preprocessFrame(frame_copy, (float *)buffers[0], input_dims[0]);
        // std::cout << "Frame preprocessed" << endl;

        context->enqueue(batch_size, buffers.data(), 0, nullptr);

        // std::cout << "Inference finished" << endl;

        out_mat_sum = cv::cuda::sum(out_gpu_mat);
        defect_score = out_mat_sum[0];
        conf = sigmoid(defect_score);

        end_tick = getTickCount();
        freq = getTickFrequency() / 1000;
        t = (end_tick - start_tick) / freq;
        timer.push_back(t);

        out_gpu_mat.download(out_mat);

        defect_score_data.stream_idx = stream_idx;
        defect_score_data.defect_score = defect_score;
        defect_score_data.confidence = conf;

        zmq::message_t msg_tr(sizeof(DefectScore));
        memcpy(msg_tr.data(), &defect_score_data, sizeof(DefectScore));

        sock_tr.send(msg_tr);

        if (defect_score > defect_score_threshold)
        {
            video_stream_str = to_string(stream_idx);
            current_time = time(nullptr);
            current_time_str = to_string(current_time);
            defects_store_path = defects_store_root + "/" + video_stream_str;
            filesystem::path dir(defects_store_path);
            filesystem::path file(video_stream_str + "_" + current_time_str + ".jpg");
            filesystem::path full_path = dir / file;
            defects_store_path = full_path.c_str();
            std::lock_guard<std::mutex> lock(mtx);
            img_queue.push(std::make_pair(defects_store_path, frame_copy)); // Produce an item
            con_var.notify_one();                                           // Notify one consumer
        }

        if (mode == "demo")
        {
            keyboard = is_play ? waitKey(10) : waitKey(0);
            if (keyboard == 27)
                break;
            if (keyboard == 112)
                is_play = !is_play;
            resize(out_mat, out_mat, Size(INPUT_WIDTH, INPUT_HEIGHT));
            cv::imshow("Response map", out_mat);
            label = format("Inf time : %.2f ms, score: %.2f, conf: %.2f", t, defect_score, conf);
            display_frame(label);
        }
    }

    for (void *buf : buffers)
    {
        cudaFree(buf);
    }
    std::lock_guard<std::mutex> lock(mtx);
    done_producing = true; // Signal that production is done
    con_var.notify_all();  // Notify all img saver consumers
    for (auto &t : img_saver)
    {
        t.join(); // Wait for all img saver consumers to finish
    }
    std::cout << "Average processing time per frame: " << std::accumulate(timer.begin(), timer.end(), 0.0) / timer.size() << " miliseconds" << endl;
    std::cout << "Done !!!" << endl;
    return 0;
}