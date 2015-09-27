#pragma once

#include <opencv2/opencv.hpp>
#include "faceparts.h"

class EyePatcher
{
public:
    EyePatcher(double patchWidth = 24, double patchHeight = 24);
    void operator()(const cv::Mat &frame, const FaceParts &faceParts, cv::Mat &dst, int interpolation = CV_INTER_LANCZOS4);
    void getMasked(const cv::Mat &frame, const FaceParts &faceParts, cv::Mat &dst, cv::Mat &emask, int interpolation = CV_INTER_LANCZOS4);
private:
    double patchWidth = 24;
    double patchHeight = 24;
    bool getEye(const cv::Mat &frame, const FaceParts &faceParts, FaceParts::FacePart fp, cv::Mat& dst, int interpolation);
};

