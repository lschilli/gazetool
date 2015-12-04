#include "relativeeyelidlearner.h"

#include <boost/lexical_cast.hpp>

using namespace std;

RelativeEyeLidLearner::RelativeEyeLidLearner(TrainingParameters params) : AbstractLearner(params)
{
    if (!trainParams.pca_epsilon.is_initialized()) {
        use_pca = true;
        trainParams.pca_epsilon = 0.85;
    }
}

RelativeEyeLidLearner::~RelativeEyeLidLearner()
{

}

void RelativeEyeLidLearner::loadClassifier(const string &filename)
{
    _loadClassifier(filename, learned_function);
}

boost::optional<dlib::matrix<double,0,1>> RelativeEyeLidLearner::getFeatureVector(GazeHyp& ghyp) {
    auto result = boost::optional<dlib::matrix<double,0,1>>();
    if (ghyp.faceFeatures.size() && ghyp.lidFeatures.size()) {
        switch (trainParams.featureSet.get_value_or(FeatureSetConfig::ALL)) {
        case FeatureSetConfig::ALL:
        case FeatureSetConfig::HOGPOS:
            result = dlib::join_cols(dlib::rowm(ghyp.faceFeatures, dlib::range(0, ghyp.faceFeatures.nr()-5)), ghyp.eyeHogFeatures);
            break;
        case FeatureSetConfig::HOG:
            result = ghyp.eyeHogFeatures;
            break;
        case FeatureSetConfig::POSITIONAL:
            result = dlib::rowm(ghyp.faceFeatures, dlib::range(0, ghyp.faceFeatures.nr()-5));
            break;
        default:
            throw std::runtime_error("feature-set not implemented");
        }
    }
    return result;
}

void RelativeEyeLidLearner::classify(GazeHyp &ghyp)
{
    _classify(ghyp, learned_function, ghyp.eyeLidClassification);
}


void RelativeEyeLidLearner::train(const string &outfilename)
{
    dlib::svr_trainer<kernel_type> trainer;
    _train(outfilename, learned_function, trainer, true);
}

void RelativeEyeLidLearner::visualize(GazeHyp &ghyp)
{
    if (!ghyp.eyeLidClassification.is_initialized() || !learned_function.basis_vectors.size()) return;
    auto eoclass = ghyp.eyeLidClassification.get();
    if (!std::isfinite(eoclass)) return;
    int color = ghyp.isLidClosed.get_value_or(false) ? 255 : 80;
    cv::rectangle(ghyp.eyePatch, cv::Rect(cv::Point(eoclass*ghyp.eyePatch.cols), cv::Size(2, 2)),
                  cv::Scalar(255, color-50, color-50), -1, 'A');
    cv::Rect r = ghyp.pupils.faceRect();
    cv::rectangle(ghyp.parentHyp.frame, cv::Rect(cv::Point(r.x+r.width, r.y+eoclass*r.height), cv::Size(5, 5)),
                  cv::Scalar(255, color-50, color-50), -1, 'A');
    if (ghyp.isLidClosed.get_value_or(false)) {
        cv::line(ghyp.parentHyp.frame, r.tl()+cv::Point(r.width/5, r.height/5),
                 r.br()-cv::Point(r.width/5, r.height/5), cv::Scalar(0, 0, 255), 2, 'A');
        cv::line(ghyp.parentHyp.frame, r.tl()+cv::Point(r.width-r.width/5, r.height/5),
                 r.tl()+cv::Point(r.width/5, r.height-r.height/5), cv::Scalar(0, 0, 255), 2, 'A');
    }
}

string RelativeEyeLidLearner::getId()
{
    return "RelativeEyeLidLearner";
}
