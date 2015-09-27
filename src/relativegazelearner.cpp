#include "relativegazelearner.h"

#include <boost/lexical_cast.hpp>

using namespace std;

RelativeGazeLearner::RelativeGazeLearner(TrainingParameters &params) : AbstractLearner(params)
{

}


RelativeGazeLearner::~RelativeGazeLearner()
{

}

boost::optional<dlib::matrix<double,0,1>> RelativeGazeLearner::getFeatureVector(GazeHyp& ghyp) {
    auto result = boost::optional<dlib::matrix<double,0,1>>();
    if (ghyp.horizGazeFeatures.size() && ghyp.faceFeatures.size() && ghyp.eyeHogFeatures.size()) {
        switch (trainParams.featureSet.get_value_or(FeatureSetConfig::ALL)) {
        case FeatureSetConfig::ALL:
            result = dlib::join_cols(dlib::join_cols(ghyp.horizGazeFeatures, ghyp.faceFeatures), ghyp.eyeHogFeatures);
            break;
        case FeatureSetConfig::POSREL:
            result = dlib::join_cols(ghyp.horizGazeFeatures, ghyp.faceFeatures);
            break;
        case FeatureSetConfig::RELATIONAL:
            result = ghyp.horizGazeFeatures;
            break;
        case FeatureSetConfig::HOG:
            result = ghyp.eyeHogFeatures;
            break;
        case FeatureSetConfig::HOGREL:
            result = dlib::join_cols(ghyp.horizGazeFeatures, ghyp.eyeHogFeatures);
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

void RelativeGazeLearner::classify(GazeHyp& ghyp){
    _classify(ghyp, learned_function, ghyp.horizontalGazeEstimation);
    //cerr << ghyp.relativeGazeClassification << endl;
}


void RelativeGazeLearner::train(const string& outfilename) {
    dlib::svr_trainer<kernel_type> trainer;
    _train(outfilename, learned_function, trainer);
}

void RelativeGazeLearner::visualize(GazeHyp& ghyp, double mutualGazeTolerance)
{
    if (!ghyp.horizontalGazeEstimation.is_initialized()) return;
    double relativeGazeEst = ghyp.horizontalGazeEstimation.get();
    if (!std::isfinite(relativeGazeEst)) {
        return;
    }
    auto fr = ghyp.pupils.faceRect();
    double angle = relativeGazeEst - 90;
    double length = fr.width/3;
    cv::Point p1 (fr.tl().x + fr.width / 2, fr.tl().y);
    cv::Point p2;
    p2.x = round(p1.x + length * cos(angle * CV_PI / 180.0));
    p2.y = round(p1.y + length * sin(angle * CV_PI / 180.0));
    //cv::line(ghyp.parentHyp.frame, p1, p2, cv::Scalar(255, 150, 150), 2, 'A');
    cv::line(ghyp.parentHyp.frame, p1, p2, cv::Scalar(255, 0, 0), 2, 'A');
    cv::Mat facereg = ghyp.pupils.faceRegion();
    int limita = facereg.cols*(0.5+mutualGazeTolerance/90.0);
    int limitb = facereg.cols*(0.5-mutualGazeTolerance/90.0);
    cv::rectangle(facereg, cv::Rect(cv::Point(limita-1, 0), cv::Size(2, 14)), cv::Scalar(255, 100, 100), -1, 'A');
    cv::rectangle(facereg, cv::Rect(cv::Point(limitb-1, 0), cv::Size(2, 14)), cv::Scalar(255, 100, 100), -1, 'A');
    int offset = facereg.cols*(0.5+relativeGazeEst/90.0);
    cv::rectangle(facereg, cv::Rect(cv::Point(offset-3, 0), cv::Size(6, 10)), cv::Scalar(255, 0, 0), -1, 'A');
}


void RelativeGazeLearner::loadClassifier(const std::string &filename)
{
    _loadClassifier(filename, learned_function);
}

string RelativeGazeLearner::getId()
{
    return "RelativeGazeLearner";
}
