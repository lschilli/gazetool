#include "faceparts.h"

using namespace std;

FaceParts::FaceParts()
{
}

FaceParts::FaceParts(const dlib::full_object_detection &shape)
{
    //start and end indices
    typedef pair<int, int> pp;
    auto polyindices = {  pp(0, 16),  pp(27, 30),  pp(17, 21),
                          pp(22, 26),  pp(31, 35),  pp(36, 41),
                          pp(42, 47),  pp(48, 59),  pp(60, 67) };
    for (auto pit = polyindices.begin(); pit != polyindices.end(); pit++) {
        int ind = pit - polyindices.begin();
        int begin = pit->first;
        int end = pit->second;
        std::vector<cv::Point> poly;
        for (int i = begin; i <= end; ++i) {
            dlib::point p = shape.part(i);
            poly.push_back(cv::Point(p.x(), p.y()));
        }
        if (!(ind == JAW || ind == LBROW || ind == RBROW || ind == NOSEWINGS)) {
            dlib::point p = shape.part(begin);
            poly.push_back(cv::Point(p.x(), p.y()));
        }
        faceFeaturePolygons.push_back(poly);
    }
}

const std::vector<cv::Point>& FaceParts::featurePolygon(FacePart part) const
{
    return faceFeaturePolygons[part];
}

cv::Point FaceParts::featurePoint(FacePart part, int ind) const
{
    return faceFeaturePolygons[part][ind];
}

void FaceParts::draw(cv::Mat &frame)
{
    cv::polylines(frame, faceFeaturePolygons, false, cv::Scalar(150, 200, 0), 1, 'A');
}


cv::Rect FaceParts::boundingRect(FacePart part) const
{
    return cv::boundingRect(cv::Mat(faceFeaturePolygons[(int)part]));
}

const vector<vector<cv::Point>>& FaceParts::featurePolygons() const
{
    return faceFeaturePolygons;
}
