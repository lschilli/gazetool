#pragma once

#include <dlib/threads.h>
#include "imageprovider.h"
#include "gazehyps.h"
#include "blockingqueue.h"

class ShapeDetectionWorker : public dlib::multithreaded_object
{
public:
    ShapeDetectionWorker(BlockingQueue<GazeHypsPtr>& inqueue, const std::string &modelfilename, int threadcount);
    ~ShapeDetectionWorker();
    BlockingQueue<GazeHypsPtr>& hypsqueue();

private:
    void thread();
    void alignFaces();
    BlockingQueue<GazeHypsPtr>& _inqueue;
    BlockingQueue<GazeHypsPtr> _hypsqueue;
    BlockingQueue<GazeHypsPtr> _workqueue;
    dlib::shape_predictor _shapePredictor;
};
