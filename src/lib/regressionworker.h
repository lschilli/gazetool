#pragma once

#include <string>
#include <dlib/threads.h>
#include "imageprovider.h"
#include "gazehyps.h"
#include "blockingqueue.h"
#include "mutualgazelearner.h"
#include "relativeeyelidlearner.h"
#include "relativegazelearner.h"
#include "verticalgazelearner.h"
#include "eyelidlearner.h"
#include "featureextractor.h"

class RegressionWorker : public dlib::multithreaded_object
{
public:
    RegressionWorker(BlockingQueue<GazeHypsPtr>& inqueue, EyeLidLearner& eoc,
                MutualGazeLearner& glearner, RelativeGazeLearner& rglearner,
                RelativeEyeLidLearner &rellearner, VerticalGazeLearner& vglearner, int threadcount);
    ~RegressionWorker();
    BlockingQueue<GazeHypsPtr>& hypsqueue();

private:
    dlib::thread_pool tpool;
    BlockingQueue<GazeHypsPtr>& _inqueue;
    BlockingQueue<GazeHypsPtr> _hypsqueue;
    EyeLidLearner& lidlearner;
    MutualGazeLearner& gazelearner;
    RelativeGazeLearner& relativeGazeLearner;
    RelativeEyeLidLearner& rellearner;
    VerticalGazeLearner& vglearner;
    FeatureExtractor featureExtractor;
    std::mutex allocmutex;
    void thread();
    void runTasks(GazeHypsPtr gazehyps);
    template<typename T1>
    void concurrentClassify(T1& learner, GazeHyp& ghyp);
};
