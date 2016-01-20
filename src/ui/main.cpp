#include <iostream>
#include <stdexcept>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <QApplication>
#include <QThread>

#include "workerthread.h"
#include "gazergui.h"

using namespace std;

namespace po = boost::program_options;

class OptionParser : public QThread {

public:
    po::variables_map options;

    OptionParser(int argc, char** argv, WorkerThread& worker, GazerGui& gui)
        : argc(argc), argv(argv), worker(worker), gui(gui)
    {}

private:
    int argc;
    char** argv;
    WorkerThread& worker;
    GazerGui& gui;

    template<typename T>
    void copyCheckArg(const string& name, T& target) {
        if (options.count(name)) {
            target = options[name].as<T>();
        }
    }

    template<typename T>
    void copyCheckArg(const string& name, boost::optional<T>& target) {
        if (options.count(name)) {
            target = options[name].as<T>();
        }
    }

    TrainingParameters parseTrainingOpts() {
        TrainingParameters params;
        map<string, FeatureSetConfig> m = { {"all", FeatureSetConfig::ALL},
                                            {"posrel", FeatureSetConfig::POSREL},
                                            {"relational", FeatureSetConfig::RELATIONAL},
                                            {"hogrel", FeatureSetConfig::HOGREL},
                                            {"hogpos", FeatureSetConfig::HOGPOS},
                                            {"positional", FeatureSetConfig::POSITIONAL},
                                            {"hog", FeatureSetConfig::HOG}};
        if (options.count("feature-set")) {
            string featuresetname = options["feature-set"].as<string>();
            std::transform(featuresetname.begin(), featuresetname.end(), featuresetname.begin(), ::tolower);
            if (m.count(featuresetname)) {
                params.featureSet = m[featuresetname];
            } else {
                throw po::error("unknown feature-set provided: " + featuresetname);
            }
        }
       copyCheckArg("svm-c", params.c);
       copyCheckArg("svm-epsilon", params.epsilon);
       copyCheckArg("svm-epsilon-insensitivity", params.epsilon_insensitivity);
       copyCheckArg("pca-epsilon", params.pca_epsilon);
       return params;
    }

    void run() {
        po::options_description allopts("\n*** dlibgazer options");
        po::options_description desc("general options");
        desc.add_options()
                ("help,h", "show help messages")
                ("model,m", po::value<string>()->required(), "read models from file arg")
                ("threads", po::value<int>(), "set number of threads per processing step")
                ("noquit", "do not quit after processing")
                ("novis", "do not display frames")
                ("quiet,q", "do not print statistics")
                ("limitfps", po::value<double>(), "slow down display fps to arg")
                ("streamppm", po::value<string>(), "stream ppm files to arg. e.g. "
                                                   ">(ffmpeg -f image2pipe -vcodec ppm -r 30 -i - -r 30 -preset ultrafast out.mp4)")
                ("dump-estimates", po::value<string>(), "dump estimated values to file")
                ("mirror", "mirror output");
        po::options_description inputops("input options");
        inputops.add_options()
                ("camera,c", po::value<string>(), "use camera number arg")
                ("video,v", po::value<string>(), "process video file arg")
                ("image,i", po::value<string>(), "process single image arg")
                ("port,p", po::value<string>(), "expect image on yarp port arg")
                ("batch,b", po::value<string>(), "batch process image filenames from arg")
                ("size", po::value<string>(), "request image size arg and scale if required")
                ("fps", po::value<int>(), "request video with arg frames per second");
        po::options_description classifyopts("classification options");
        classifyopts.add_options()
                ("classify-gaze", po::value<string>(), "load classifier from arg")
                ("train-gaze-classifier", po::value<string>(), "train gaze classifier and save to arg")
                ("classify-lid", po::value<string>(), "load classifier from arg")
                ("estimate-lid", po::value<string>(), "load classifier from arg")
                ("train-lid-classifier", po::value<string>(), "train lid classifier and save to arg")
                ("train-lid-estimator", po::value<string>(), "train lid estimator and save to arg")
                ("estimate-gaze", po::value<string>(), "estimate gaze")
                ("estimate-verticalgaze", po::value<string>(), "estimate vertical gaze")
                ("horizontal-gaze-tolerance", po::value<double>(), "mutual gaze tolerance in deg")
                ("vertical-gaze-tolerance", po::value<double>(), "mutual gaze tolerance in deg")
                ("train-gaze-estimator", po::value<string>(), "train gaze estimator and save to arg")
                ("train-verticalgaze-estimator", po::value<string>(), "train vertical gaze estimator and save to arg");
        po::options_description trainopts("parameters applied to all active trainers");
        trainopts.add_options()
                ("svm-c", po::value<double>(), "svm c parameter")
                ("svm-epsilon", po::value<double>(), "svm epsilon parameter")
                ("svm-epsilon-insensitivity", po::value<double>(), "svmr insensitivity parameter")
                ("feature-set", po::value<string>(), "use feature set arg")
                ("pca-epsilon", po::value<double>(), "pca dimension reduction depending on arg");
        allopts.add(desc).add(inputops).add(classifyopts).add(trainopts);
        try {
            po::store(po::parse_command_line(argc, argv, allopts), options);
            if (options.count("help")) {
                allopts.print(cout);
                std::exit(0);
            }
            po::notify(options);
            for (const auto& s : { "camera", "image", "video", "port", "batch"}) {
                if (options.count(s)) {
                    if (worker.inputType.empty()) {
                        worker.inputParam = options[s].as<string>();
                        worker.inputType = s;
                    } else {
                        throw po::error("More than one input option provided");
                    }
                }
            }
            if (worker.inputType.empty()) {
                throw po::error("No input option provided");
            }
            if (options.count("size")) {
                auto sizestr = options["size"].as<string>();
                vector<string> args;
                boost::split(args, sizestr, boost::is_any_of(":x "));
                if (args.size() != 2) throw po::error("invalid size " + sizestr);
                worker.inputSize = cv::Size(boost::lexical_cast<int>(args[0]), boost::lexical_cast<int>(args[1]));
            }
            copyCheckArg("fps", worker.desiredFps);
            copyCheckArg("threads", worker.threadcount);
            copyCheckArg("streamppm", worker.streamppm);
            copyCheckArg("model", worker.modelfile);
            copyCheckArg("classify-gaze", worker.classifyGaze);
            copyCheckArg("train-gaze-classifier", worker.trainGaze);
            copyCheckArg("train-lid-classifier", worker.trainLid);
            copyCheckArg("train-lid-estimator", worker.trainLidEstimator);
            copyCheckArg("classify-lid", worker.classifyLid);
            copyCheckArg("estimate-lid", worker.estimateLid);
            copyCheckArg("estimate-gaze", worker.estimateGaze);
            copyCheckArg("estimate-verticalgaze", worker.estimateVerticalGaze);
            copyCheckArg("train-gaze-estimator", worker.trainGazeEstimator);
            copyCheckArg("train-verticalgaze-estimator", worker.trainVerticalGazeEstimator);
            copyCheckArg("limitfps", worker.limitFps);
            copyCheckArg("dump-estimates", worker.dumpEstimates);
            copyCheckArg("horizontal-gaze-tolerance", worker.horizGazeTolerance);
            copyCheckArg("vertical-gaze-tolerance", worker.verticalGazeTolerance);
            if (options.count("quiet")) worker.showstats = false;
            gui.setHorizGazeTolerance(worker.horizGazeTolerance);
            gui.setVerticalGazeTolerance(worker.verticalGazeTolerance);
            bool mirror = false;
            copyCheckArg("mirror", mirror);
            worker.trainingParameters = parseTrainingOpts();
            gui.setMirror(mirror);
        }
        catch(po::error& e) {
            cerr << "Error parsing command line:" << endl << e.what() << endl;
            std::exit(1);
        }

    }

};

int main(int argc, char** argv) {
    qRegisterMetaType<GazeHypsPtr>();
    qRegisterMetaType<std::string>();
    WorkerThread gazer;
    QApplication app(argc, argv);
    GazerGui gui;
    //trying to write to cerr, cout, or throw an exception leads to deadlock in this function.
    //the reason is currently a mystery.
    //As a workaround a new thread is started. This does not make much sense but it works.
    OptionParser optparser(argc, argv, gazer, gui);
    optparser.start();
    optparser.wait();
    if (!optparser.options.count("novis")) gui.show();
    QThread thread;
    gazer.moveToThread(&thread);

    QObject::connect(&app, SIGNAL(lastWindowClosed()), &gazer, SLOT(stop()));
    QObject::connect(&gazer, SIGNAL(finished()), &thread, SLOT(quit()));
    if (!optparser.options.count("novis")) {
        QObject::connect(&gazer, SIGNAL(imageProcessed(GazeHypsPtr)), &gui,
                         SLOT(displayGazehyps(GazeHypsPtr)), Qt::QueuedConnection);
    }
    QObject::connect(&gazer, SIGNAL(statusmsg(std::string)), &gui, SLOT(setStatusmsg(std::string)));
    QObject::connect(&thread, SIGNAL(started()), &gazer, SLOT(process()));
    QObject::connect(&gui, SIGNAL(horizGazeToleranceChanged(double)), &gazer, SLOT(setHorizGazeTolerance(double)));
    QObject::connect(&gui, SIGNAL(verticalGazeToleranceChanged(double)), &gazer, SLOT(setVerticalGazeTolerance(double)));
    QObject::connect(&gui, SIGNAL(smoothingChanged(bool)), &gazer, SLOT(setSmoothing(bool)));
    if (!optparser.options.count("noquit")) {
        QObject::connect(&gazer, SIGNAL(finished()), &app, SLOT(quit()));
    }

    thread.start();
    app.exec();
    //process events after event loop terminates allowing unfinished threads to send signals
    while (thread.isRunning()) {
        thread.wait(10);
        QCoreApplication::processEvents();
    }
}
