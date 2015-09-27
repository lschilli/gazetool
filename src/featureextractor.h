#pragma once

#include "gazehyps.h"

class FeatureExtractor
{
public:
    FeatureExtractor();
    ~FeatureExtractor();

    void extractLidFeatures(GazeHyp &ghyp);
    void extractFaceFeatures(GazeHyp &ghyp);
    void combineFeatures(GazeHyp &ghyp);
    void extractEyeHogFeatures(GazeHyp &ghyp);
    void extractVertGazeFeatures(GazeHyp &ghyp);
    void extractHorizGazeFeatures(GazeHyp &ghyp);
};
