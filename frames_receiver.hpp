#ifndef THUMBNAIL_RECEIVER_HPP_INCLUDED
#define THUMBNAIL_RECEIVER_HPP_INCLUDED

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctime>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace std;
using namespace cv;
using namespace dnn;

const int MODEL_INPUT_WIDTH = 224;
const int MODEL_INPUT_HEIGHT = 224;
const int CHANNELS = 3;
const int X_STEP_SIZE = 224;
const int Y_STEP_SIZE = 224;

const int INPUT_WIDTH = MODEL_INPUT_WIDTH * 3;
const int INPUT_HEIGHT = MODEL_INPUT_HEIGHT * 2;
const int IMG_SIZE = INPUT_WIDTH * INPUT_HEIGHT * CHANNELS;
const int STREAM_IDX_SIZE = 4;
const int DEFECT_SCORE_SIZE = 4;

const int OUTPUT_WIDTH = 100;
const int OUTPUT_HEIGHT = 100;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

std::vector<std::string> class_names{"Defected", "Clear"};
Scalar stdev = Scalar(0.229, 0.224, 0.225);
string model_path = "models/VddConvSDResnet18.onnx";
string defected_class_name = "Defected";
string out_text = "";
string DEFAULT_DST_PATH = "defectstore";

int batch_size = 1;
string video_file_name = "";
Mat result;

int keyboard;
bool is_play = true;

const size_t size = INPUT_HEIGHT * INPUT_WIDTH * 3 * sizeof(uint8_t);
zmq::message_t msg(INPUT_HEIGHT *INPUT_WIDTH * 3 * sizeof(uint8_t));
cv::Mat frame(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
cv::Mat frame_copy;
cv::Mat frame_save;

// static const char *zmq_hostname = "tcp://127.0.0.1:5555";
string DST_ADDRESS = "127.0.0.1";
string DST_PORT = "5555";
string zmq_hostname;
zmq::message_t recvmsg;
zmq::context_t ctx;
zmq::socket_t sock(ctx, ZMQ_PULL);

zmq::context_t ctx_tr;
zmq::socket_t sock_tr(ctx_tr, ZMQ_PUSH);

char stream_idx_str[STREAM_IDX_SIZE];
char def_score_arr[DEFECT_SCORE_SIZE];
uchar *img_data;
int stream_idx;

float defect_score;
float defect_score_threshold{0.0};
float conf;

float sigmoid(const float &m1);
vector<float> sigmoid(const vector<float> &m1);

std::time_t current_time;
string current_time_str;
string defects_store_path;
std::vector<float> results{0, 0};

std::queue<std::pair<std::string_view, cv::Mat>> img_queue;
std::mutex mtx;
std::condition_variable con_var;
bool done_producing = false;

struct DefectScore
{
    int stream_idx;
    float defect_score;
    float confidence;
};

DefectScore defect_score_data;

void display_frame(string &label);
void save_defected_frame(string &defects_store_path);
void img_saver_consumer(int id);

#endif // THUMBNAIL_RECEIVER_HPP_INCLUDED
