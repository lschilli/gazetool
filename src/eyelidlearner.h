#pragma once

#include <vector>
#include <utility>
#include <dlib/svm.h>
#include <opencv2/opencv.hpp>

#include "gazehyps.h"
#include "abstractlearner.h"

class EyeLidLearner : public AbstractLearner
{
public:
    EyeLidLearner(TrainingParameters& params);
    virtual ~EyeLidLearner();
    virtual void loadClassifier(const std::string& filename);
    virtual void classify(GazeHyp &ghyp);
    virtual void train(const std::string &outfilename);
    virtual void visualize(GazeHyp& ghyp);
    virtual std::string getId();

protected:
    typedef dlib::linear_kernel<sample_type> kernel_type;
    typedef dlib::decision_function<kernel_type> dec_funct_type;
    typedef dlib::probabilistic_decision_function<kernel_type> probabilistic_funct_type;
    probabilistic_funct_type decision_function;
    boost::optional<dlib::matrix<double,0,1>> getFeatureVector(GazeHyp& ghyp);
};
