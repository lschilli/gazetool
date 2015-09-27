#pragma once

#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

class FaceParts
{
public:
    enum FacePart { JAW = 0, NOSE = 1, RBROW = 2, LBROW = 3, NOSEWINGS = 4, REYE = 5, LEYE = 6, OUTERLIPS = 7, INNERLIPS = 8};
    FaceParts();
    FaceParts(const dlib::full_object_detection& shape);
    const std::vector<std::vector<cv::Point>>& featurePolygons() const;
    const std::vector<cv::Point> &featurePolygon(FacePart part) const;
    cv::Rect boundingRect(FacePart part) const;
    cv::Point featurePoint(FacePart part, int ind) const;
    void draw(cv::Mat& frame);

protected:
    std::vector<std::vector<cv::Point>> faceFeaturePolygons;
};
