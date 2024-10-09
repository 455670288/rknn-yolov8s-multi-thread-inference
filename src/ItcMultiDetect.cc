#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "ItcMultiDetect.h"
#include "rkYolov8n.h"
#include "resize_function.h"
#include "rknn_utils.h"
#include "postprocess.h"
#include "timer.h"

ItcMultiDetect::ItcMultiDetect(const std::string &model_path)
{
    pImpl = new Impl(model_path);
}

ItcMultiDetect::~ItcMultiDetect()
{
    delete pImpl;
}

int ItcMultiDetect::init_model(rknn_context *ctx_in, const float &nmsThreshold, const float &boxThreshold, const int &NPUcore, const std::vector<float> &confs)
{
    return pImpl->init_model(ctx_in, nmsThreshold, boxThreshold, NPUcore, confs);
}

cv::Mat ItcMultiDetect::detect(cv::Mat &image, std::vector<int> *ids, std::vector<float> *conf, std::vector<cv::Rect> *boxes)
{
    return pImpl->detect(image, ids, conf, boxes);
}