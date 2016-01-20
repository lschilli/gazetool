#include "facedetectionworker.h"

#include <dlib/threads.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <iostream>
#include <thread>
#include <future>
#include <memory>

using namespace std;

FaceDetectionWorker::FaceDetectionWorker(std::unique_ptr<ImageProvider> imgprovider, int threadcount)
    : _detector(dlib::get_frontal_face_detector()), imgprovider(std::move(imgprovider)), _hypsqueue(threadcount), _workqueue(threadcount) {
    register_thread(*this, &FaceDetectionWorker::thread);
    for (int i = 0; i < threadcount; i++) {
        register_thread(*this, &FaceDetectionWorker::detectfaces);
    }
    start();
}

FaceDetectionWorker::~FaceDetectionWorker() {
    _workqueue.interrupt();
    _hypsqueue.interrupt();
    stop();
    wait();
}

BlockingQueue<GazeHypsPtr>& FaceDetectionWorker::hypsqueue()
{
    return _hypsqueue;
}

void FaceDetectionWorker::detectfaces() {
    //working with thread individual copy, since the detector is not thread safe.
    dlib::frontal_face_detector detector = _detector;
    try {
        while (true) {
            GazeHypsPtr gazehyps = _workqueue.pop();
            const auto faceDetections = detector(gazehyps->dlibimage);
            for (const auto& facerect : faceDetections) {
                GazeHyp ghyp(*gazehyps);
                ghyp.faceDetection = facerect;
                gazehyps->addGazeHyp(ghyp);
            }
            gazehyps->setready(-1);
        }
    } catch (QueueInterruptedException) {}
}

void FaceDetectionWorker::thread() {
    try {
        while (!should_stop()) {
            GazeHypsPtr ghyps(new GazeHypList());
            ghyps->setready(1);
            _hypsqueue.waitAccept();
            _workqueue.waitAccept();
            if (imgprovider->get(ghyps->frame)) {
                ghyps->frameTime = std::chrono::system_clock::now();
                ghyps->label = imgprovider->getLabel();
                ghyps->id = imgprovider->getId();
                ghyps->dlibimage = dlib::cv_image<dlib::bgr_pixel>(ghyps->frame);
                _workqueue.push(ghyps);
                _hypsqueue.push(ghyps);
            } else {
                break;
            }
        }
    } catch(QueueInterruptedException) {}
    _workqueue.interrupt();
    _hypsqueue.interrupt();
}
