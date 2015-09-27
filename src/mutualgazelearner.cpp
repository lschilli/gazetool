#include "mutualgazelearner.h"

#include <boost/lexical_cast.hpp>
#include "eyepatcher.h"

using namespace std;



MutualGazeLearner::MutualGazeLearner(TrainingParameters& params) : AbstractLearner(params)
{
}


MutualGazeLearner::~MutualGazeLearner()
{

}


void MutualGazeLearner::loadClassifier(const std::string &filename)
{
    ifstream infile(filename, ios::in | ios::binary);
    if (!infile.is_open()) throw dlib::serialization_error("Error: Cannot open " + filename);
    deserialize(normalizer_pca, infile);
    deserialize(decision_function, infile);
    _initialized = true;
}


boost::optional<dlib::matrix<double,0,1>> MutualGazeLearner::getFeatureVector(GazeHyp& ghyp) {
    auto result = boost::optional<dlib::matrix<double,0,1>>();
    if (ghyp.horizGazeFeatures.size() && ghyp.faceFeatures.size() && ghyp.eyeHogFeatures.size()) {
        result = dlib::join_cols(dlib::join_cols(ghyp.horizGazeFeatures, ghyp.faceFeatures), ghyp.eyeHogFeatures);
    }
    return result;
}


void MutualGazeLearner::classify(GazeHyp& ghyp){
    auto fv = getFeatureVector(ghyp);
    if (decision_function.basis_vectors.size() && fv.is_initialized()) {
        ghyp.mutualGazeClassification = decision_function(normalizer_pca(fv.get()));
    }
}


void MutualGazeLearner::train(const string& outfilename) {
    normalizer_pca.train(samples, 0.99);
    cerr << "pca matrix rows: " << normalizer_pca.pca_matrix().nr() << " cols: " << normalizer_pca.pca_matrix().nc() << endl;
    //cerr << pca.in_vector_size() << endl;
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer_pca(samples[i]);
    cerr << "Normalizer completed...." << endl;

    double initialc1 = 0;
    double initialc2 = 0;
    for (double l : labels) {
        if (l < 0) initialc1++;
        if (l > 0) initialc2++;
    }
    double mincls = min(initialc1, initialc2);
    initialc1 *= 12.4/mincls;
    initialc2 *= 16.0/mincls;
    dlib::svm_c_trainer<kernel_type> trainer;
    cerr << "initial c1,c2 = " << initialc1 << ", " << initialc2 << endl;
    double bestc1 = 0;
    double bestc2 = 0;
    double bestgamma = 0;
    double maxresult = 0;
    trainer.set_epsilon(0.1);
    for (auto gamma : {0.16, 0.2, 0.3}) {
        trainer.set_kernel(kernel_type(gamma));
        double c1 = initialc1;
        double c2 = initialc2;
        for (int i = 1; i < 7; i++) {
            trainer.set_c_class1(c1);
            trainer.set_c_class2(c2);
            // The first element of the vector is the fraction of +1 training examples correctly classified
            // and the second number is the fraction of -1 training examples correctly classified.
            auto valresult = dlib::cross_validate_trainer(trainer, samples, labels, 3);
            cerr << "cross validation accuracy (c1, c2 = " << c1 << ", " << c2
                 << " gamma: " << gamma << "): " << valresult;
            double fm = valresult(0) * valresult(1);
            if (fm > maxresult) {
                maxresult = fm;
                bestc1 = c1;
                bestc2 = c2;
                bestgamma = gamma;
            }
            c1 *= 0.5;
            c2 *= 0.5;
        }
    }
    cerr << "best c1, c2 = " << bestc1 << ", " << bestc2 << " gamma: " << bestgamma << endl;
    trainer.set_kernel(kernel_type(bestgamma));
    trainer.set_c_class1(bestc1);
    trainer.set_c_class2(bestc2);
    //auto redtrainer = dlib::reduced2(trainer, 400);
    decision_function = trainer.train(samples, labels);
    cerr << "basis vectors: " << decision_function.basis_vectors.size() << endl;
    ofstream outfile(outfilename, ios::out | ios::binary);
    serialize(normalizer_pca, outfile);
    serialize(decision_function, outfile);
}


void MutualGazeLearner::visualize(GazeHyp& ghyp)
{
    if (ghyp.isMutualGaze.get_value_or(false)) {
        auto fr = ghyp.pupils.faceRect();
        cv::rectangle(ghyp.parentHyp.frame, fr, cv::Scalar(0, 0, 255), 2, 'A');
    }
}

string MutualGazeLearner::getId()
{
    return "MutualGazeLearner";
}
