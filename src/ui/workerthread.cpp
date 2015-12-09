#include "workerthread.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <QCoreApplication>

#include "imageprovider.h"
#include "faceparts.h"
#include "pupilfinder.h"
#include "mutualgazelearner.h"
#include "verticalgazelearner.h"
#include "eyelidlearner.h"
#include "relativeeyelidlearner.h"
#include "relativegazelearner.h"
#include "facedetectionworker.h"
#include "shapedetectionworker.h"
#include "regressionworker.h"
#include "eyepatcher.h"
#include "rlssmoother.h"

#ifdef ENABLE_YARP_SUPPORT
    #include "yarpsupport.h"
#endif

using namespace std;
using namespace boost::accumulators;


class TemporalStats {
private:
    const int accumulatorWindowSize = 50;
    int counter = 0;
    accumulator_set<double, stats<tag::rolling_sum>> fps_acc;
    accumulator_set<double, stats<tag::rolling_sum>> latency_acc;
    std::chrono::time_point<std::chrono::system_clock> starttime;

public:
    TemporalStats() : fps_acc(tag::rolling_window::window_size = accumulatorWindowSize),
                      latency_acc(tag::rolling_window::window_size = accumulatorWindowSize),
                      starttime(std::chrono::system_clock::now())
    {}
    void operator()(GazeHypsPtr gazehyps) {
        auto tnow = chrono::high_resolution_clock::now();
        auto mcs = chrono::duration_cast<std::chrono::microseconds> (tnow - starttime);
        auto lat = chrono::duration_cast<std::chrono::microseconds>(tnow - gazehyps->frameTime);
        starttime = tnow;
        fps_acc(mcs.count());
        latency_acc(lat.count());
        gazehyps->frameCounter = counter;
        if (counter > accumulatorWindowSize) {
            gazehyps->fps = 1e6*min(accumulatorWindowSize, counter) / rolling_sum(fps_acc);
            gazehyps->latency = rolling_sum(latency_acc) / (1e3*accumulatorWindowSize);
        }
        counter++;
    }
    void printStats(GazeHypsPtr gazehyps) {
        if (gazehyps->frameCounter % 10 == 0)  {
            cerr << "fps: " << round(gazehyps->fps) << " | lat: " << round(gazehyps->latency)
                 << " | cnt: " << gazehyps->frameCounter << endl;
        }
    }
};


WorkerThread::WorkerThread(QObject *parent) :
    QObject(parent)
{
}


std::unique_ptr<ImageProvider> WorkerThread::getImageProvider() {
    std::unique_ptr<ImageProvider> imgProvider;
    if (inputType == "port") {
#ifdef ENABLE_YARP_SUPPORT
        imgProvider.reset(new YarpImageProvider(inputParam));
#else
        throw runtime_error("yarp support not enabled");
#endif
    } else if (inputType == "camera") {
        imgProvider.reset(new CvVideoImageProvider(boost::lexical_cast<int>(inputParam), inputSize, desiredFps));
    } else if (inputType == "video") {
        imgProvider.reset(new CvVideoImageProvider(inputParam, inputSize));
    } else if (inputType == "batch") {
        imgProvider.reset(new BatchImageProvider(inputParam));
    } else if (inputType == "image") {
        vector<string> filenames;
        filenames.push_back(inputParam);
        imgProvider.reset(new BatchImageProvider(filenames));
    } else {
        throw runtime_error("invalid input type " + inputType);
    }
    return imgProvider;
}


void WorkerThread::normalizeMat(const cv::Mat& in, cv::Mat& out) {
    cv::Scalar avg, sdv;
    cv::meanStdDev(in, avg, sdv);
    sdv.val[0] = sqrt(in.cols*in.rows*sdv.val[0]*sdv.val[0]);
    in.convertTo(out, CV_64FC1, 1/sdv.val[0], -avg.val[0]/sdv.val[0]);
}

void WorkerThread::dumpPpm(ofstream& fout, const cv::Mat& frame) {
    if (fout.is_open()) {
        vector<uchar> buff;
        cv::imencode(".pgm", frame, buff);
        fout.write((char*)&buff.front(), buff.size());
    }
}

void WorkerThread::writeEstHeader(ofstream& fout) {
    fout << "Frame" << "\t"
         << "Id" << "\t"
         << "Label" << "\t"
         << "Lid" << "\t"
         << "HorizGaze" << "\t"
         << "VertGaze" << "\t"
         << "MutualGaze"
         << endl;
}

void WorkerThread::dumpEst(ofstream& fout, GazeHypsPtr gazehyps) {
    if (fout.is_open()) {
        double lid = std::nan("not set");
        double gazeest = std::nan("not set");
        double vertest = std::nan("not set");
        bool mutgaze = false;
        if (gazehyps->size()) {
            GazeHyp& ghyp = gazehyps->hyps(0);
            lid = ghyp.eyeLidClassification.get_value_or(lid);
            gazeest = ghyp.horizontalGazeEstimation.get_value_or(gazeest);
            vertest = ghyp.verticalGazeEstimation.get_value_or(vertest);
            mutgaze = ghyp.isMutualGaze.get_value_or(false);
        }
        fout << gazehyps->frameCounter << "\t"
             << gazehyps->id << "\t"
             << gazehyps->label << "\t"
             << lid << "\t"
             << gazeest << "\t"
             << vertest << "\t"
             << mutgaze
             << endl;
    }
}

void WorkerThread::stop() {
    shouldStop = true;
}

void WorkerThread::setHorizGazeTolerance(double tol)
{
    horizGazeTolerance = tol;
}

void WorkerThread::setVerticalGazeTolerance(double tol)
{
    verticalGazeTolerance = tol;
}

void WorkerThread::setSmoothing(bool enabled)
{
    smoothingEnabled = enabled;
}


void WorkerThread::interpretHyp(GazeHyp& ghyp) {
    double lidclass = ghyp.eyeLidClassification.get_value_or(0);
    if (ghyp.eyeLidClassification.is_initialized()) {
        ghyp.isLidClosed = (lidclass > 0.7);
    }
    if (ghyp.mutualGazeClassification.is_initialized()) {
        ghyp.isMutualGaze = (ghyp.mutualGazeClassification.get() > 0) && !ghyp.isLidClosed.get_value_or(false);
    }
    if (ghyp.horizontalGazeEstimation.is_initialized()) {
        ghyp.isMutualGaze = ghyp.isMutualGaze.get_value_or(true)
             && (abs(ghyp.horizontalGazeEstimation.get()) < horizGazeTolerance);
    }
    if (ghyp.verticalGazeEstimation.is_initialized()) {
        ghyp.isMutualGaze = ghyp.isMutualGaze.get_value_or(true)
             && (abs(ghyp.verticalGazeEstimation.get()) < verticalGazeTolerance);
    }
    if (ghyp.isMutualGaze.is_initialized()) {
        ghyp.isMutualGaze = ghyp.isMutualGaze.get() && !ghyp.isLidClosed.get_value_or(false);
    }
}

template<typename T>
static void tryLoadModel(T& learner, const string& filename) {
    try {
        if (!filename.empty())
            learner.loadClassifier(filename);
    } catch (dlib::serialization_error &e) {
        cerr << filename << ":" << e.what() << endl;
    }
}

void WorkerThread::process() {
    MutualGazeLearner glearner(trainingParameters);
    RelativeGazeLearner rglearner(trainingParameters);
    EyeLidLearner eoclearner(trainingParameters);
    RelativeEyeLidLearner rellearner(trainingParameters);
    VerticalGazeLearner vglearner(trainingParameters);
    tryLoadModel(glearner, classifyGaze);
    tryLoadModel(eoclearner, classifyLid);
    tryLoadModel(rglearner, estimateGaze);
    tryLoadModel(rellearner, estimateLid);
    tryLoadModel(vglearner, estimateVerticalGaze);
    emit statusmsg("Setting up detector threads...");
    std::unique_ptr<ImageProvider> imgProvider(getImageProvider());
    FaceDetectionWorker faceworker(std::move(imgProvider), threadcount);
    ShapeDetectionWorker shapeworker(faceworker.hypsqueue(), modelfile, max(1, threadcount/2));
    RegressionWorker regressionWorker(shapeworker.hypsqueue(), eoclearner, glearner, rglearner, rellearner, vglearner, max(1, threadcount));
    emit statusmsg("Detector threads started");
#ifdef ENABLE_YARP_SUPPORT
    unique_ptr<YarpSender> yarpSender;
    if (inputType == "port") {
        yarpSender.reset(new YarpSender(inputParam));
    }
#endif
    ofstream ppmout;
    if (!streamppm.empty()) {
        ppmout.open(streamppm);
    }
    ofstream estimateout;
    if (!dumpEstimates.empty()) {
        estimateout.open(dumpEstimates);
        if (estimateout.is_open()) {
            writeEstHeader(estimateout);
        } else {
            cerr << "Warning: could not open " << dumpEstimates << endl;
        }
    }
    RlsSmoother horizGazeSmoother;
    RlsSmoother vertGazeSmoother;
    RlsSmoother lidSmoother(5, 0.95, 0.09);
    emit statusmsg("Entering processing loop...");
    cerr << "Processing frames..." << endl;
    TemporalStats temporalStats;
    while(!shouldStop) {
        GazeHypsPtr gazehyps;
        try {
            gazehyps = regressionWorker.hypsqueue().peek();
            gazehyps->waitready();
        } catch(QueueInterruptedException) {
            break;
        }
        cv::Mat frame = gazehyps->frame;

        for (auto& ghyp : *gazehyps) {
            if (smoothingEnabled) {
                horizGazeSmoother.smoothValue(ghyp.horizontalGazeEstimation);
                vertGazeSmoother.smoothValue(ghyp.verticalGazeEstimation);
                lidSmoother.smoothValue(ghyp.eyeLidClassification);
            }
            interpretHyp(ghyp);
            auto& pupils = ghyp.pupils;
            auto& faceparts = ghyp.faceParts;
            faceparts.draw(frame);
            pupils.draw(frame);
            glearner.visualize(ghyp);
            eoclearner.visualize(ghyp);
            rellearner.visualize(ghyp);
            vglearner.visualize(ghyp, verticalGazeTolerance);
            rglearner.visualize(ghyp, horizGazeTolerance);
            if (!trainLid.empty()) eoclearner.accumulate(ghyp);
            if (!trainGaze.empty()) glearner.accumulate(ghyp);
            if (!trainGazeEstimator.empty()) rglearner.accumulate(ghyp);
            if (!trainLidEstimator.empty()) rellearner.accumulate(ghyp);
            if (!trainVerticalGazeEstimator.empty()) vglearner.accumulate(ghyp);
        }
        temporalStats(gazehyps);
        dumpPpm(ppmout, frame);
        dumpEst(estimateout, gazehyps);
        if (showstats) temporalStats.printStats(gazehyps);
#ifdef ENABLE_YARP_SUPPORT
        if (yarpSender) yarpSender->sendGazeHypotheses(gazehyps);
#endif
        emit imageProcessed(gazehyps);
        QCoreApplication::processEvents();
        if (limitFps > 0) {
            usleep(1e6/limitFps);
        }
        regressionWorker.hypsqueue().pop();
    }
    regressionWorker.hypsqueue().interrupt();
    regressionWorker.wait();
    cerr << "Frames processed..." << endl;
    if (glearner.sampleCount() > 0) {
        glearner.train(trainGaze);
    }
    if (eoclearner.sampleCount() > 0) {
        eoclearner.train(trainLid);
    }
    if (vglearner.sampleCount() > 0) {
        vglearner.train(trainVerticalGazeEstimator);
    }
    if (rglearner.sampleCount() > 0) {
        rglearner.train(trainGazeEstimator);
    }
    if (rellearner.sampleCount() > 0) {
        rellearner.train(trainLidEstimator);
    }
    emit finished();
    cerr << "Primary worker thread finished processing" << endl;
}

