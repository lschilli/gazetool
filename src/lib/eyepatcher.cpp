#include "eyepatcher.h"

#include <iostream>

using namespace std;


EyePatcher::EyePatcher(double patchWidth, double patchHeight)
    : patchWidth(patchWidth), patchHeight(patchHeight) {
}


void EyePatcher::operator()(const cv::Mat &frame, const FaceParts &faceParts, cv::Mat &dst, int interpolation) {
    dst.create(patchHeight, patchWidth*2, frame.type());
    cv::Mat left(dst(cv::Rect(0, 0, patchWidth, patchHeight)));
    cv::Mat right(dst(cv::Rect(patchWidth, 0, patchWidth, patchHeight)));
    if (getEye(frame, faceParts, FaceParts::LEYE, left, interpolation)
            && getEye(frame, faceParts, FaceParts::REYE, right, interpolation)) {
        return;
    }
    dst = cv::Mat();
}

void EyePatcher::getMasked(const cv::Mat &frame, const FaceParts &faceParts, cv::Mat &dst, cv::Mat &emask, int interpolation) {
    //todo: inefficient
    cv::Mat mask(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    auto poly = faceParts.featurePolygon(FaceParts::LEYE);
    cv::fillConvexPoly(mask, poly, cv::Scalar(255), 'A');
    poly = faceParts.featurePolygon(FaceParts::REYE);
    cv::fillConvexPoly(mask, poly, cv::Scalar(255), 'A');
    cv::Mat patch;
    operator()(frame, faceParts, patch, interpolation);
    cv::cvtColor(patch, dst, CV_BGR2GRAY);
    //emask = cv::Mat(dst.size(), dst.type(), cv::Scalar(0));
    //vector<cv::Point> poly;
    //cv::ellipse2Poly(cv::Point(patchWidth/2, patchHeight/2), cv::Size(patchWidth/2, patchHeight*0.3214), 180, 0, 360, 1, poly);
    //cv::fillConvexPoly(emask, poly, cv::Scalar(255));
    //cv::ellipse2Poly(cv::Point(patchWidth/2+patchWidth, patchHeight/2), cv::Size(patchWidth/2, patchHeight*0.3214), 180, 0, 360, 1, poly);
    //cv::fillConvexPoly(emask, poly, cv::Scalar(255));

    operator()(mask, faceParts, emask, cv::INTER_LINEAR);
    cv::threshold(emask, emask, 1, 255, CV_MINMAX);
    //dst &= emask;
    //cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, dst.type());
    cv::equalizeHist(dst, dst);
}

bool EyePatcher::getEye(const cv::Mat &frame, const FaceParts &faceParts, FaceParts::FacePart fp, cv::Mat& dst, int interpolation) {
    cv::Point lc(faceParts.featurePoint(fp, 0));
    cv::Point rc(faceParts.featurePoint(fp, 3));
    cv::Point centersum;
    auto eyePoly = faceParts.featurePolygon(fp);
    auto epNum = eyePoly.size()-1;
    for (size_t i = 0; i < epNum; i++) {
        centersum += eyePoly[i];
    }
    //center is on the axis between left and right corner
    cv::Point2f center(centersum.x/double(epNum), centersum.y/double(epNum));
    double width = (norm(lc-rc));
    double height = (width*(patchHeight/patchWidth));
    double ang = atan2(rc.y - lc.y, rc.x - lc.x)*180/M_PI;
    cv::Rect_<float> bbox(cv::Point2f(center.x-width/2, center.y-height/2), cv::Size2f(width, height));
    //compensate for cropping due to rotation
    cv::Rect rotbbox = cv::RotatedRect(cv::Point2f(bbox.x + bbox.width/2, bbox.y + bbox.height/2), bbox.size(), -ang).boundingRect();
    double scalef = patchWidth/bbox.width;
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(rotbbox.width/2.0, rotbbox.height/2.0), ang, scalef);
    // adjust transformation matrix
    rot.at<double>(0,2) -= (rotbbox.width/2./scalef-width/2.)*scalef;
    rot.at<double>(1,2) -= (rotbbox.height/2./scalef-height/2.)*scalef;
    cv::Rect framerect(cv::Point(0, 0), frame.size());
    if (framerect.contains(rotbbox.tl()) && framerect.contains(rotbbox.br())) {
        cv::warpAffine(cv::Mat(frame, rotbbox), dst, rot, cv::Size(patchWidth, patchHeight), interpolation);
        return true;
    }
    return false;
}
