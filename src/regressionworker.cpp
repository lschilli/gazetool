#include "regressionworker.h"
#include "pupilfinder.h"
#include "eyepatcher.h"

#include <dlib/threads.h>
#include <thread>
#include <future>

using namespace std;


RegressionWorker::RegressionWorker(BlockingQueue<GazeHypsPtr>& inqueue, EyeLidLearner &eoc, MutualGazeLearner &glearner,
                         RelativeGazeLearner &rglearner, RelativeEyeLidLearner& rellearner, VerticalGazeLearner& vglearner, int threadcount)
    : tpool(threadcount), _inqueue(inqueue), _hypsqueue(threadcount),
      lidlearner(eoc), gazelearner(glearner), relativeGazeLearner(rglearner), rellearner(rellearner), vglearner(vglearner)
{
    register_thread(*this, &RegressionWorker::thread);
    start();
}

RegressionWorker::~RegressionWorker() {
    _hypsqueue.interrupt();
    tpool.wait_for_all_tasks();
    stop();
    wait();
}

BlockingQueue<GazeHypsPtr> &RegressionWorker::hypsqueue() {
    return _hypsqueue;
}

template<typename T1>
void RegressionWorker::concurrentClassify(T1& learner, GazeHyp& ghyp) {
    static dlib::thread_specific_data<std::unique_ptr<T1>> dataHolder;
    if (!learner.isInitialized()) return;
    tpool.add_task_by_value( [&ghyp, &learner, this](void) {
        if (!dataHolder.data()) {
            lock_guard<mutex> lock(allocmutex);
            dataHolder.data() = unique_ptr<T1>(new T1(learner));
        }
        dataHolder.data()->classify(ghyp);
    });
}

void RegressionWorker::runTasks(GazeHypsPtr gazehyps) {
    for (auto& ghyp : *gazehyps) {
        tpool.add_task_by_value( [&gazehyps, &ghyp](void) {ghyp.pupils = PupilFinder(gazehyps->frame, ghyp.faceParts);} );
        tpool.add_task_by_value( [&ghyp, this](void) {featureExtractor.extractLidFeatures(ghyp);} );
        tpool.add_task_by_value( [&ghyp, this](void) {featureExtractor.extractEyeHogFeatures(ghyp);} );
        tpool.wait_for_all_tasks();
        featureExtractor.extractFaceFeatures(ghyp);
        featureExtractor.extractHorizGazeFeatures(ghyp);
        featureExtractor.extractVertGazeFeatures(ghyp);
        concurrentClassify(lidlearner, ghyp);
        concurrentClassify(gazelearner, ghyp);
        concurrentClassify(rellearner, ghyp);
        concurrentClassify(relativeGazeLearner, ghyp);
        concurrentClassify(vglearner, ghyp);
    }
    tpool.wait_for_all_tasks();
    gazehyps->setready(-1);
}


void RegressionWorker::thread() {
    try {
        while (!should_stop()) {
            _hypsqueue.waitAccept();
            _inqueue.peek()->waitready();
            GazeHypsPtr ghyps = _inqueue.pop();
            ghyps->setready(1);
            tpool.add_task_by_value(  [ghyps, this](void) {runTasks(ghyps);} );
            _hypsqueue.push(ghyps);
        }
    } catch(QueueInterruptedException) {}
    _hypsqueue.interrupt();
    tpool.wait_for_all_tasks();
}

