#include "eyelidlearner.h"
#include "eyepatcher.h"

#include <boost/lexical_cast.hpp>

using namespace std;

EyeLidLearner::EyeLidLearner(TrainingParameters &params) : AbstractLearner(params)
{

}


EyeLidLearner::~EyeLidLearner()
{

}


void EyeLidLearner::loadClassifier(const string &filename) {
    ifstream infile(filename, ios::in | ios::binary);
    if (!infile.is_open()) throw dlib::serialization_error("Error: Cannot open " + filename);
    deserialize(normalizer_pca, infile);
    deserialize(decision_function, infile);
    _initialized = true;
}

boost::optional<dlib::matrix<double,0,1>> EyeLidLearner::getFeatureVector(GazeHyp& ghyp) {
    auto result = boost::optional<dlib::matrix<double,0,1>>();
    if (ghyp.faceFeatures.size() && ghyp.lidFeatures.size()) {
        result = dlib::join_cols(dlib::rowm(ghyp.faceFeatures, dlib::range(0, ghyp.faceFeatures.nr()-3)), ghyp.lidFeatures);
    }
    return result;
}

void EyeLidLearner::classify(GazeHyp& ghyp) {
    auto fv = getFeatureVector(ghyp);
    if (!fv.is_initialized() || !decision_function.decision_funct.basis_vectors.size()) return;
    ghyp.eyeLidClassification = decision_function(normalizer_pca(fv.get()));
}


void EyeLidLearner::train(const string &outfilename) {
    cerr << "EyeOpenCloseLearner train...." << endl;
    normalizer_pca.train(samples, 0.85);
    cerr << "pca matrix rows: " << normalizer_pca.pca_matrix().nr() << " cols: " << normalizer_pca.pca_matrix().nc() << endl;
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer_pca(samples[i]);
    cerr << "Normalizer completed...." << endl;
    randomize_samples(samples, labels);

    dlib::svm_c_trainer<kernel_type> trainer;
    trainer.set_epsilon(0.1);
//    for (double c = 0.001; c < 1; c *= 1.5) {
//        trainer.set_c_class1(c);
//        trainer.set_c_class2(c*0.9);
////         The first element of the vector is the fraction of +1 training examples
////         correctly classified and the second number is the fraction of -1 training
////         examples correctly classified.
//        cerr << "c: " << c  << "  cross validation accuracy: " << cross_validate_trainer(trainer, samples, labels, 3);
//    }
    trainer.set_c_class1(0.0113906);
    trainer.set_c_class2(0.01025154);
    cerr << "cross validation accuracy: " << cross_validate_trainer(trainer, samples, labels, 3);

    decision_function = train_probabilistic_decision_function(trainer, samples, labels, 3);
    cerr << "number of support vectors: " << decision_function.decision_funct.basis_vectors.size() << endl;
    ofstream outfile(outfilename, ios::out | ios::binary);
    serialize(normalizer_pca, outfile);
    serialize(decision_function, outfile);
}


void EyeLidLearner::visualize(GazeHyp& ghyp)
{
    if (!ghyp.eyeLidClassification.is_initialized() || !decision_function.decision_funct.basis_vectors.size()) return;
    auto eoclass = ghyp.eyeLidClassification.get();
    int color = eoclass > 0.5 ? 255 : 127;
    cv::rectangle(ghyp.eyePatch, cv::Rect(cv::Point(eoclass*(ghyp.eyePatch.cols-1)-1), cv::Size(2, 2)),
                  cv::Scalar(255, color-50, color-50), -1);
    cv::Rect r = ghyp.pupils.faceRect();
    cv::rectangle(ghyp.parentHyp.frame, cv::Rect(cv::Point(r.x+r.width, r.y+eoclass*r.height), cv::Size(5, 5)),
                  cv::Scalar(255, color-50, color-50), -1, 'A');
    if (eoclass > 0.5) {
        cv::line(ghyp.parentHyp.frame, r.tl()+cv::Point(r.width/5, r.height/5),
                 r.br()-cv::Point(r.width/5, r.height/5), cv::Scalar(250, 0, 250), 2, 'A');
        cv::line(ghyp.parentHyp.frame, r.tl()+cv::Point(r.width-r.width/5, r.height/5),
                 r.tl()+cv::Point(r.width/5, r.height-r.height/5), cv::Scalar(250, 0, 250), 2, 'A');
    }
}

string EyeLidLearner::getId()
{
    return "EyeLidLearner";
}

