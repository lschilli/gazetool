#include "abstractlearner.h"
#include <boost/lexical_cast.hpp>
#include <typeinfo>

AbstractLearner::AbstractLearner(TrainingParameters params) : trainParams(params)
{
    use_pca = trainParams.pca_epsilon.is_initialized();
}

AbstractLearner::~AbstractLearner()
{

}

bool AbstractLearner::isInitialized()
{
    return _initialized;
}

void AbstractLearner::accumulate(GazeHyp &ghyp)
{
    auto fv = getFeatureVector(ghyp);
    if (!ghyp.parentHyp.label.empty() && fv.is_initialized()) {
        samples.push_back(fv.get());
        double lbl = boost::lexical_cast<int>(ghyp.parentHyp.label);
        labels.push_back(lbl);
    }
}

size_t AbstractLearner::sampleCount()
{
    return samples.size();
}
