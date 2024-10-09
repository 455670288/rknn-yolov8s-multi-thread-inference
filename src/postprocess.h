#pragma once
#include <opencv2/opencv.hpp>

#include "rknn_utils.h"
#include "resize_function.h"


#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 10
#define OBJ_CLASS_NUM 19
// #define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)
#define NMS_THRESHOLD 0.45
#define CONF_THRESHOLD 0.25

namespace multi_det
{
    // typedef enum
    // {
    //     Q8 = 0,
    //     FP = 1,
    // } POST_PROCESS_TYPE;

    // typedef struct _YOLO_INFO
    // {
    //     API_TYPE api_type = NORMAL_API;
    //     POST_PROCESS_TYPE post_type = Q8;

    //     int dfl_len = -1;
    //     bool score_sum_available = false;
    // } YOLO_INFO;

    typedef struct _BOX_RECT
    {
        int left;
        int right;
        int top;
        int bottom;
        // int x;      //center_x
        // int y;      //center_y
    } BOX_RECT;

    typedef struct __detect_result_t
    {
        int class_index;
        BOX_RECT box;
        float prop;
    } detect_result_t;

    typedef struct _detect_result_group_t  
    {
        int id;
        int count;
        detect_result_t results[OBJ_NUMB_MAX_SIZE];
    } detect_result_group_t;

    // void query_dfl_len(MODEL_INFO *m, YOLO_INFO *y_info);
    // void compute_dfl(float *tensor, int dfl_len, float *box);
    // int process_i8(int8_t *t_box, int32_t box_zp, float box_scale, int8_t *t_score, int32_t score_zp,
    //             float score_scale, int8_t *t_score_sum, int32_t sum_zp, float sum_scale, int dfl_len, int grid_h,
    //             int grid_w, int stride, std::vector<float> &boxes, std::vector<float> &boxScores,
    //             std::vector<int> &classId, float threshold, bool score_sum_available);
    // int process_fp(float *t_box, float *t_score, float *t_score_sum, int dfl_len, int grid_h, int grid_w, int stride,
    //             std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId,
    //             float threshold, bool score_sum_available);
    // int post_process(std::vector<void *> rk_outputs, float conf_threshold, float nms_threshold, MODEL_INFO *m_info, YOLO_INFO *y_info,
    //                 LETTER_BOX *lb,
    //                 detect_result_group_t *group);


    // int post_process(int8_t *input0, int model_in_h, int model_in_w,
    //              float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
    //              std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
    //              detect_result_group_t *group);
    // int init_post_process();
    // void deinit_post_process();
    // char *coco_cls_to_name(int cls_id);
    int post_process(rknn_context ctx, void *outputs, LETTER_BOX *letter_box, float conf_threshold, float nms_threshold, detect_result_group_t *od_results,int width, int height, rknn_input_output_num io_num, rknn_tensor_attr *output_attrs, bool is_quant);
    


}
    //20240919
    void deinitPostProcess();
    // const int num_classes = OBJ_CLASS_NUM;  // 假设模型有80个类别
    // const int feature_map_sizes[4] = {160, 80, 40, 20};  // 四个尺度的特征图
    // const int strides[4] = {4, 16, 32, 64};  // 四个尺度对应的stride