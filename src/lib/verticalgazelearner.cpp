#include "verticalgazelearner.h"
#include "eyepatcher.h"

#include <boost/lexical_cast.hpp>

using namespace std;

VerticalGazeLearner::VerticalGazeLearner(TrainingParameters& params) : AbstractLearner(params)
{

}

VerticalGazeLearner::~VerticalGazeLearner()
{

}

void VerticalGazeLearner::loadClassifier(const string &filename)
{
    _loadClassifier(filename, learned_function);
}

boost::optional<dlib::matrix<double,0,1>> VerticalGazeLearner::getFeatureVector(GazeHyp& ghyp) {
    auto result = boost::optional<dlib::matrix<double,0,1>>();
    if (ghyp.lidFeatures.size() && ghyp.faceFeatures.size() && ghyp.vertGazeFeatures.size()) {
        switch (trainParams.featureSet.get_value_or(FeatureSetConfig::ALL)) {
        case FeatureSetConfig::ALL:
            result = dlib::join_cols(dlib::join_cols(ghyp.vertGazeFeatures, ghyp.faceFeatures), ghyp.eyeHogFeatures);
            break;
        case FeatureSetConfig::POSREL:
            result = dlib::join_cols(ghyp.vertGazeFeatures, ghyp.faceFeatures);
            break;
        case FeatureSetConfig::RELATIONAL:
            result = ghyp.vertGazeFeatures;
            break;
        case FeatureSetConfig::HOG:
            result = ghyp.eyeHogFeatures;
            break;
        case FeatureSetConfig::HOGREL:
            result = dlib::join_cols(ghyp.vertGazeFeatures, ghyp.eyeHogFeatures);
            break;
        case FeatureSetConfig::HOGPOS:
            result = dlib::join_cols(ghyp.faceFeatures, ghyp.eyeHogFeatures);
            break;
        case FeatureSetConfig::POSITIONAL:
            result = ghyp.faceFeatures;
            break;
        default:
            throw std::runtime_error("feature-set not implemented");
        }
    }
    return result;

}


void VerticalGazeLearner::classify(GazeHyp &ghyp)
{
    _classify(ghyp, learned_function, ghyp.verticalGazeEstimation);
    //cerr << ghyp.verticalGazeEstimation.get() << endl;
}


void VerticalGazeLearner::train(const string &outfilename)
{
    dlib::svr_trainer<kernel_type> trainer;
    _train(outfilename, learned_function, trainer);

}

void VerticalGazeLearner::visualize(GazeHyp &ghyp, double mutualGazeTolerance)
{
    if (!ghyp.verticalGazeEstimation.is_initialized()) return;
    double relativeGazeEst = ghyp.verticalGazeEstimation.get();
    if (!std::isfinite(relativeGazeEst)) return;
    auto fr = ghyp.pupils.faceRect();
    double angle = -relativeGazeEst;
    double length = fr.width/3;
    cv::Point p1 (fr.tl().x + fr.width / 2, fr.tl().y);
    cv::Point p2;
    p2.x = round(p1.x + length * cos(angle * CV_PI / 180.0));
    p2.y = round(p1.y + length * sin(angle * CV_PI / 180.0));
    cv::line(ghyp.parentHyp.frame, p1, p2, cv::Scalar(255, 0, 0), 2, 'A');
    //cv::line(ghyp.parentHyp.frame, p1, p2, cv::Scalar(150, 250, 150), 2, 'A');

    cv::Mat facereg = ghyp.pupils.faceRegion();
    int limita = facereg.rows*(0.5-mutualGazeTolerance/90.0);
    int limitb = facereg.rows*(0.5+mutualGazeTolerance/90.0);
    cv::rectangle(facereg, cv::Rect(cv::Point(0, limita-1), cv::Size(14, 2)), cv::Scalar(255, 100, 100), -1, 'A');
    cv::rectangle(facereg, cv::Rect(cv::Point(0, limitb-1), cv::Size(14, 2)), cv::Scalar(255, 100, 100), -1, 'A');
    int offset = facereg.rows*(0.5-relativeGazeEst/90.0);
    cv::rectangle(facereg, cv::Rect(cv::Point(0, offset-3), cv::Size(10, 6)), cv::Scalar(255, 0, 0), -1, 'A');
}

string VerticalGazeLearner::getId()
{
    return "VerticalGazeLearner";
}
