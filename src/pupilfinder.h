#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/optional.hpp>

#include "faceparts.h"

class PupilFinder
{
public:
    struct CenterCandidate {
        cv::Point2d center;
        double radius;
    };

    PupilFinder();
    PupilFinder(cv::Mat& frame, const FaceParts& faceParts);

    cv::Mat faceRegion();
    cv::Rect faceRect();
    cv::Rect leftEyeBounds() const;
    cv::Rect rightEyeBounds() const;
    const boost::optional<CenterCandidate> &rightCandidate();
    const boost::optional<CenterCandidate> &leftCandidate();
    int pupilsFound() const;
    void draw(cv::Mat& frame);

private:
    boost::optional<CenterCandidate> findEye(std::vector<cv::Point> epoly, cv::Rect_<double> eyerect, FaceParts::FacePart eyeid);
    double setupFaceRegion(const cv::Mat &frame, const cv::Rect &facerect, const cv::Rect &lebounds, const cv::Rect &rebounds);
    void drawCross(cv::Mat img, cv::Point center, cv::Scalar color, int d = 3, int thickness = 1, int lineType = 8);

    cv::Rect frect;
    cv::Mat faceROIgray;
    cv::Mat faceROIcolor;
    std::vector<cv::Point> lepoly;
    cv::Rect lebounds;
    std::vector<cv::Point> repoly;
    cv::Rect rebounds;
    double scalefac;
    int pupfound = 0;
    boost::optional<CenterCandidate> lpupCandidate;
    boost::optional<CenterCandidate> rpupCandidate;

};


