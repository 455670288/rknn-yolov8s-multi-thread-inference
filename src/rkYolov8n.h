#pragma once
#include "ItcMultiDetect.h"
#include "postprocess.h"
#include "resize_function.h"
#include "rknn_utils.h"
#include "timer.h"
#include <sys/time.h>
#include "coreNum.hpp"
#include "preprocess.h"

using namespace multi_det;

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

// class ItcMultiDetect::Impl
class rkYolov8n
{
public:
    rkYolov8n(const std::string &model_path);
    ~rkYolov8n();

    int init_model(rknn_context *ctx_in, bool share_weight);
    cv::Mat detect(cv::Mat &image);
    rknn_context *get_pctx();

    // //20240919
    // int channel, width, height;   //模型尺寸
    // int img_width, img_height;

    // rknn_context ctx;
    // rknn_input_output_num io_num;
    // rknn_tensor_attr *input_attrs;
    // rknn_tensor_attr *output_attrs;
    // rknn_input inputs[1];

private:
    TIMER timer;

    // MODEL_INFO m_info;
    // YOLO_INFO y_info;
    LETTER_BOX letter_box;

    float nms_threshold = 0.45;
    float conf_threshold = 0.25;
    std::vector<void *> output_buf_list;
    std::vector<float> conf_thresholds;

    
    //20240919
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;
    int channel, width, height;   //模型尺寸
    int img_width, img_height;
    bool is_quant = true;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];


// private:
//     int detect_zero_copy(cv::Mat rgb_img, detect_result_group_t *group);
//     int detect_normal(cv::Mat rgb_img, detect_result_group_t *group);
};