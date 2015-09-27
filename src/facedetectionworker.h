#pragma once

#include <dlib/threads.h>
#include "imageprovider.h"
#include "gazehyps.h"
#include "blockingqueue.h"

class FaceDetectionWorker : public dlib::multithreaded_object
{
public:
    FaceDetectionWorker(std::unique_ptr<ImageProvider> imgprovider, int threadcount);
    ~FaceDetectionWorker();
    BlockingQueue<GazeHypsPtr>& hypsqueue();

private:
    void thread();
    void detectfaces();
    dlib::frontal_face_detector _detector;
    std::unique_ptr<ImageProvider> imgprovider;
    BlockingQueue<GazeHypsPtr> _hypsqueue;
    BlockingQueue<GazeHypsPtr> _workqueue;
};
