#pragma once

#include <boost/optional.hpp>
#include <fstream>
#include <dlib/serialize.h>
#include <dlib/svm.h>
#include "gazehyps.h"

enum class FeatureSetConfig {POSITIONAL, RELATIONAL, HOG, POSREL, HOGREL, HOGPOS, ALL};
static std::vector<std::string> featureSetNames = {"POSITIONAL", "RELATIONAL", "HOG", "POSREL", "HOGREL", "HOGPOS", "ALL"};

struct TrainingParameters {
    boost::optional<double> epsilon;
    boost::optional<double> epsilon_insensitivity;
    boost::optional<double> c;
    boost::optional<double> pca_epsilon;
    boost::optional<FeatureSetConfig> featureSet;
};

class AbstractLearner
{
public:
    AbstractLearner(TrainingParameters params);
    virtual ~AbstractLearner();
    virtual bool isInitialized();
    virtual boost::optional<dlib::matrix<double,0,1>> getFeatureVector(GazeHyp& ghyp) = 0;
    virtual void accumulate(GazeHyp &ghyp);
    virtual size_t sampleCount();
    virtual std::string getId() = 0;

protected:
    typedef dlib::matrix<double,0,1> sample_type;
    typedef std::vector<double> label_type;
    std::vector<sample_type> samples;
    label_type labels;
    dlib::vector_normalizer<sample_type> normalizer;
    dlib::vector_normalizer_pca<sample_type> normalizer_pca;
    bool _initialized = false;
    bool use_pca;
    TrainingParameters trainParams;

    template<typename T>
    void _loadClassifier(const std::string &filename, T& learned_function)
    {
        std::ifstream infile(filename, std::ios::in | std::ios::binary);
        if (!infile.is_open()) throw dlib::serialization_error("Error: Cannot open " + filename);
        dlib::deserialize(use_pca, infile);
        if (use_pca) {
            deserialize(normalizer_pca, infile);
        } else {
            deserialize(normalizer, infile);
        }
        deserialize(learned_function, infile);
        int fsc;
        dlib::deserialize(fsc, infile);
        if (trainParams.featureSet.is_initialized()) {
            std::cerr << "Warning: Overriding featureset for " << getId()
                 << " due to classifier loading from " << filename << std::endl;
        }
        trainParams.featureSet = static_cast<FeatureSetConfig>(fsc);
        _initialized = true;
    }

    template<typename T1, typename T2>
    void _classify(GazeHyp& ghyp, T1& learned_function, T2& target) {
        auto fv = getFeatureVector(ghyp);
        if (learned_function.basis_vectors.size() && fv.is_initialized()) {
            if (use_pca) {
                target = learned_function(normalizer_pca(fv.get()));
            } else {
                target = learned_function(normalizer(fv.get()));
            }
        }
    }

    template<typename T1, typename T2>
    void _train(const std::string &outfilename, T1& learned_function, T2& trainer, bool samplerandomization = false)
    {
        std::cerr << getId() << " training..." << std::endl;
        int indim, outdim;
        if (use_pca) {
            normalizer_pca.train(samples, trainParams.pca_epsilon.get());
            indim = normalizer_pca.in_vector_size();
            outdim = normalizer_pca.out_vector_size();
            for (unsigned long i = 0; i < samples.size(); ++i)
                samples[i] = normalizer_pca(samples[i]);
            indim = normalizer_pca.in_vector_size();
            outdim = normalizer_pca.out_vector_size();
        } else {
            normalizer.train(samples);
            for (unsigned long i = 0; i < samples.size(); ++i)
                samples[i] = normalizer(samples[i]);
            indim = normalizer.in_vector_size();
            outdim = normalizer.out_vector_size();
        }
        std::cout << "Normalizer completed:" << std::endl
                  << "Input dimensions: " << indim << std::endl
                  << "Output dimensions: " << outdim << std::endl
                  << "Featureset: " << featureSetNames.at(static_cast<int>(trainParams.featureSet.get_value_or(FeatureSetConfig::ALL)))
                  << std::endl
                  << "pca eps: " << trainParams.pca_epsilon.get_value_or(nan("not set")) << std::endl;
        if (samplerandomization) randomize_samples(samples, labels);
        trainer.set_c(trainParams.c.get_value_or(5));
        trainer.set_epsilon_insensitivity(trainParams.epsilon_insensitivity.get_value_or(0.1));
        trainer.set_epsilon(trainParams.epsilon.get_value_or(0.05));
        trainer.set_cache_size(10000);
        std::cout << "Training parameters:" << std::endl
                  << "svr c: " << trainer.get_c() << std::endl
                  << "svr eps: " << trainer.get_epsilon() << std::endl
                  << "svr eps-insens: " << trainer.get_epsilon_insensitivity() << std::endl;
        learned_function = trainer.train(samples, labels);
        std::cout << "Basis vectors: " << learned_function.basis_vectors.size() << std::endl;
        std::ofstream outfile(outfilename, std::ios::out | std::ios::binary);
        dlib::serialize(use_pca, outfile);
        if (use_pca) {
            dlib::serialize(normalizer_pca, outfile);
        } else {
            dlib::serialize(normalizer, outfile);
        }
        dlib::serialize(learned_function, outfile);
        dlib::serialize(static_cast<int>(trainParams.featureSet.get_value_or(FeatureSetConfig::ALL)), outfile);
        std::cout << "Crosseval..." << std::endl;
        std::cout << "MSE and R-Squared: "<< dlib::cross_validate_regression_trainer(trainer, samples, labels, 3);
    }




};
