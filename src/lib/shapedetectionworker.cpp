#include "shapedetectionworker.h"

#include <dlib/threads.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <iostream>
#include <thread>
#include <future>
#include <memory>

using namespace std;

ShapeDetectionWorker::ShapeDetectionWorker(BlockingQueue<GazeHypsPtr>& inqueue, const std::string &modelfilename, int threadcount)
    : _inqueue(inqueue), _hypsqueue(threadcount), _workqueue(threadcount) {
    dlib::deserialize(modelfilename) >> _shapePredictor; // read face model from file
    register_thread(*this, &ShapeDetectionWorker::thread);
    for (int i = 0; i < threadcount; i++) {
        register_thread(*this, &ShapeDetectionWorker::alignFaces);
    }
    start();
}

ShapeDetectionWorker::~ShapeDetectionWorker() {
    _workqueue.interrupt();
    _hypsqueue.interrupt();
    stop();
    wait();
}

BlockingQueue<GazeHypsPtr>& ShapeDetectionWorker::hypsqueue()
{
    return _hypsqueue;
}

void ShapeDetectionWorker::alignFaces() {
    //working with thread individual copy, since the detector is not thread safe.
    dlib::shape_predictor sp = _shapePredictor;
    try {
        while (true) {
            GazeHypsPtr gazehyps = _workqueue.pop();
            for (auto& ghyp : *gazehyps) {
                dlib::full_object_detection shape = sp(gazehyps->dlibimage, ghyp.faceDetection);
                ghyp.shape = shape;
                ghyp.faceParts = FaceParts(shape);
            }
            gazehyps->setready(-1);
        }
    } catch (QueueInterruptedException) {}
}

void ShapeDetectionWorker::thread() {
    try {
        while (!should_stop()) {
            _hypsqueue.waitAccept();
            _inqueue.peek()->waitready();
            GazeHypsPtr ghyps = _inqueue.pop();
            ghyps->setready(1);
            _workqueue.push(ghyps);
            _hypsqueue.push(ghyps);
        }
    } catch(QueueInterruptedException) {}
    _workqueue.interrupt();
    _hypsqueue.interrupt();
}
