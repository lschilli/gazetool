#pragma once

#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <qt5/QtCore/QMetaType>
#include <boost/optional.hpp>

#include "pupilfinder.h"

class GazeHypList;
typedef std::shared_ptr<GazeHypList> GazeHypsPtr;
Q_DECLARE_METATYPE(GazeHypsPtr)

struct GazeHyp {
    dlib::rectangle faceDetection;
    dlib::full_object_detection shape;
    PupilFinder pupils;
    FaceParts faceParts;
    dlib::matrix<double,0,1> faceFeatures;
    dlib::matrix<double,0,1> lidFeatures;
    dlib::matrix<double,0,1> horizGazeFeatures;
    dlib::matrix<double,0,1> vertGazeFeatures;
    dlib::matrix<double,0,1> eyeHogFeatures;
    cv::Mat eyePatch;
    boost::optional<double> eyeLidClassification;
    boost::optional<double> mutualGazeClassification;
    boost::optional<double> horizontalGazeEstimation;
    boost::optional<double> verticalGazeEstimation;
    boost::optional<bool> isMutualGaze;
    boost::optional<bool> isLidClosed;
    GazeHypList& parentHyp;
    GazeHyp(GazeHypList& parent) : parentHyp(parent) {}
};

class GazeHypList
{
public:
    GazeHypList();
    cv::Mat frame;
    std::chrono::system_clock::time_point frameTime;
    dlib::cv_image<dlib::bgr_pixel> dlibimage;
    double latency = 0.0;
    double fps = 0.0;
    int frameCounter = 0;
    std::string label;
    std::string id;
    void waitready();
    void setready(int ready);
    void addGazeHyp(GazeHyp& hyp);
    std::vector<GazeHyp>::iterator begin();
    std::vector<GazeHyp>::iterator end();
    std::vector<GazeHyp>::const_iterator begin() const;
    std::vector<GazeHyp>::const_iterator end() const;
    size_t size();
    GazeHyp& hyps(int i);

private:
    std::vector<GazeHyp> _hyps;
    std::mutex _mutex;
    std::condition_variable _cond;
    int _tasks = 0;
};

