#pragma once
#include <opencv2/opencv.hpp>
#include <rknn_api.h>

class ItcMultiDetect
{
public:
    ItcMultiDetect();
    ~ItcMultiDetect();

    int init_model(rknn_context *ctx_in, const float &nmsThreshold, const float &boxThreshold, const int &NPUcore, const std::vector<float> &confs);
    cv::Mat detect(cv::Mat &image, std::vector<int> *ids, std::vector<float> *conf, std::vector<cv::Rect> *boxes);

public:
    class Impl;
    Impl *pImpl;
};