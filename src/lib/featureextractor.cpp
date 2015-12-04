#include "featureextractor.h"
#include "eyepatcher.h"

#include <vector>

using namespace std;

FeatureExtractor::FeatureExtractor()
{

}

FeatureExtractor::~FeatureExtractor()
{

}

void FeatureExtractor::extractLidFeatures(GazeHyp& ghyp) {
    EyePatcher ep;
    ep(ghyp.parentHyp.frame, ghyp.faceParts, ghyp.eyePatch, cv::INTER_LINEAR);
    if (!ghyp.eyePatch.empty()) {
        dlib::cv_image<dlib::rgb_pixel> rgbcv(ghyp.eyePatch);
        ghyp.lidFeatures = dlib::extract_fhog_features(rgbcv, 8);
        //cerr << "hog dimensions: " << fhogmat.nr() << endl;
    }
}

void FeatureExtractor::extractEyeHogFeatures(GazeHyp &ghyp)
{
    // hog features on eye area to provide context
    EyePatcher ep(32, 32);
    cv::Mat eyePatch;
    ep(ghyp.parentHyp.frame, ghyp.faceParts, eyePatch, cv::INTER_LINEAR);
    if (eyePatch.empty()) return;
    dlib::cv_image<dlib::rgb_pixel> rgbcv(eyePatch);
    ghyp.eyeHogFeatures = dlib::extract_fhog_features(rgbcv, 8);

}


void FeatureExtractor::extractFaceFeatures(GazeHyp &ghyp)
{
    auto& faceParts = ghyp.faceParts;
    auto& pupils = ghyp.pupils;
    if (pupils.pupilsFound() < 2) return;
    vector<double> features;

    cv::Point2d lleft;
    lleft = faceParts.featurePoint(FaceParts::LEYE, 3);
    //cv::circle(frame, lleft, 3, cv::Scalar(200, 0, 0), 1, 'A');
    cv::Point2d lright;
    lright = faceParts.featurePoint(FaceParts::LEYE, 0);
    //cv::circle(frame, lright, 3, cv::Scalar(255, 255, 50), 1, 'A');
    cv::Point2d rleft;
    rleft = faceParts.featurePoint(FaceParts::REYE, 3);
    //cv::circle(frame, rleft, 3, cv::Scalar(0, 255, 0), 1, 'A');
    cv::Point2d rright;
    rright = faceParts.featurePoint(FaceParts::REYE, 0);
    //cv::circle(frame, rright, 3, cv::Scalar(0, 50, 255), 1, 'A');

    cv::Point2d coordinateRoot;
    coordinateRoot = faceParts.featurePolygon(FaceParts::NOSE).at(0);
    //cv::circle(ghyp.parentHyp.frame, coordinateRoot, 3, cv::Scalar(0, 50, 255), 1, 'A');


    cv::Point2d lpup = pupils.leftCandidate().get().center;
    cv::Point2d rpup = pupils.rightCandidate().get().center;

    vector<cv::Vec2d> fxpoints;
    for (unsigned long i = 0; i < ghyp.shape.num_parts(); i++) {
        dlib::point p = ghyp.shape.part(i);
        fxpoints.push_back(cv::Vec2d(p.x(), p.y()));
    }
    fxpoints.push_back(cv::Vec2d(lpup));
    fxpoints.push_back(cv::Vec2d(rpup));

    cv::Vec2d b1(lright - rleft);
    cv::Vec2d b2(b1[1], -b1[0]);
    for (auto& p : fxpoints) {
        cv::Vec2d pvec(p[0] - lright.x, p[1] - lright.y);
        double mag = cv::norm(b1);
        cv::Vec2d proj(pvec.dot(b1)/mag+lright.x, -pvec.dot(b2)/mag+lright.y);
        cv::Vec2d croot;
        croot = coordinateRoot;
        proj = (proj - croot) / mag;
        features.push_back(proj[0]);
        features.push_back(proj[1]);
        //cv::circle(ghyp.parentHyp.frame, cv::Point(proj[0]*40+100, proj[1]*40+100), 2, cv::Scalar(0, 50, 255), 1, 'A');
    }
    ghyp.faceFeatures = dlib::mat(features);
}


void FeatureExtractor::extractHorizGazeFeatures(GazeHyp &ghyp)
{
    auto& faceParts = ghyp.faceParts;
    auto& pupils = ghyp.pupils;
    if (pupils.pupilsFound() < 2) return;
    vector<double> features;

    cv::Point2d lleft;
    lleft = faceParts.featurePoint(FaceParts::LEYE, 3);
    //cv::circle(frame, lleft, 3, cv::Scalar(200, 0, 0), 1, 'A');
    cv::Point2d lright;
    lright = faceParts.featurePoint(FaceParts::LEYE, 0);
    //cv::circle(frame, lright, 3, cv::Scalar(255, 255, 50), 1, 'A');
    cv::Point2d rleft;
    rleft = faceParts.featurePoint(FaceParts::REYE, 3);
    //cv::circle(frame, rleft, 3, cv::Scalar(0, 255, 0), 1, 'A');
    cv::Point2d rright;
    rright = faceParts.featurePoint(FaceParts::REYE, 0);
    //cv::circle(frame, rright, 3, cv::Scalar(0, 50, 255), 1, 'A');

    cv::Point2d coordinateRoot;
    coordinateRoot = faceParts.featurePolygon(FaceParts::NOSE).at(0);
    {
        cv::Vec2d a(coordinateRoot - lright);
        cv::Vec2d b(rleft - lright);
        double mag = cv::norm(b);
        coordinateRoot = a.dot(b)*b/(mag*mag) + cv::Vec2d(lright);
    }

    lleft -= coordinateRoot;
    lright -= coordinateRoot;
    rleft -= coordinateRoot;
    rright -= coordinateRoot;

    cv::Point2d lpup = pupils.leftCandidate().get().center;
    lpup -= coordinateRoot;
    cv::Point2d rpup = pupils.rightCandidate().get().center;
    rpup -= coordinateRoot;

    { // left pupil position, projected to axis between inner eye corners
        cv::Vec2d b(lright-rleft);
        //cv::Vec2d b(lleft - lright);
        cv::Vec2d a(lpup - lright);
        double cdist = cv::norm(lleft - lright);
        double mag = cv::norm(b);
        double rel = a.dot(b)/(mag*cdist);
        features.push_back(rel);
    }

    { //right pupil position, projected to axis between inner eye corners
        cv::Vec2d b(rleft - lright);
        //cv::Vec2d b(rright - rleft);
        cv::Vec2d a(rpup - rleft);
        double cdist = cv::norm(rright - rleft);
        double mag = cv::norm(b);
        double rel = a.dot(b)/(mag*cdist);
        features.push_back(rel);
    }

    { //relation of corners to center (pan)
        double n1 = cv::norm(lright);
        double n2 = cv::norm(rleft);
        double rel = (n1-n2)/(n1+n2);
        features.push_back(rel);
    }

    { //relation of corners to center (pan)
        double n1 = cv::norm(rright);
        double n2 = cv::norm(lleft);
        double rel = (n1-n2)/(n1+n2);
        features.push_back(rel);
    }

    { //angle of the nose back
        cv::Vec2d b(rleft - lright);
        cv::Point2d nosetip;
        nosetip = faceParts.featurePolygon(FaceParts::NOSE).at(3);
        nosetip -= coordinateRoot;
        cv::Vec2d a(nosetip);
        double rel = a.dot(b)/(cv::norm(a)*cv::norm(b));
        features.push_back(rel);
    }

    { // nose triangle test
        cv::Point2d nosetip;
        nosetip = faceParts.featurePolygon(FaceParts::NOSE).at(3);
        cv::Point2d noseroot;
        noseroot = faceParts.featurePolygon(FaceParts::NOSE).at(0);
        cv::Point2d wingcenter;
        wingcenter = faceParts.featurePolygon(FaceParts::NOSEWINGS).at(2);
        cv::Vec2d a(noseroot-nosetip);
        cv::Vec2d b(nosetip-wingcenter);
        cv::Vec2d c(wingcenter-noseroot);
        double rel = (norm(a) + norm(b)) / norm(c);
        features.push_back(rel);
    }

    ghyp.horizGazeFeatures = dlib::mat(features);
}



void FeatureExtractor::extractVertGazeFeatures(GazeHyp &ghyp)
{
    auto& faceParts = ghyp.faceParts;
    auto& pupils = ghyp.pupils;
    if (pupils.pupilsFound() < 2) return;
    vector<double> features;

    cv::Point2d lright;
    lright = faceParts.featurePoint(FaceParts::LEYE, 0);
    cv::Point2d rleft;
    rleft = faceParts.featurePoint(FaceParts::REYE, 3);
    cv::Point2d lleft;
    lleft = faceParts.featurePoint(FaceParts::LEYE, 3);
    cv::Point2d rright;
    rright = faceParts.featurePoint(FaceParts::REYE, 0);
    cv::Point2d coordinateRoot;
    coordinateRoot = faceParts.featurePolygon(FaceParts::NOSE).at(0);
    {
        cv::Vec2d a(coordinateRoot - lright);
        cv::Vec2d b(rleft - lright);
        double mag = cv::norm(b);
        coordinateRoot = a.dot(b)*b/(mag*mag) + cv::Vec2d(lright);
    }

    cv::Point2d rlwingcenter;
    {
        rlwingcenter = faceParts.featurePolygon(FaceParts::NOSEWINGS).at(0)
                + faceParts.featurePolygon(FaceParts::NOSEWINGS).at(4);
        rlwingcenter = cv::Vec2d(rlwingcenter)/2. - cv::Vec2d(coordinateRoot);
    }

    {
        cv::Point2d pup = pupils.leftCandidate().get().center;
        pup -= coordinateRoot;
        cv::Vec2d a(pup);
        cv::Vec2d b(rlwingcenter);
        double mag = cv::norm(b)*cv::norm(lleft - lright);
        cv::Point2d projpup;
        projpup = a.dot(b)*b/mag;
        double n1 = cv::norm(rlwingcenter);
        double n2 = cv::norm(projpup);
        double rel = (n1-n2) / (n1+n2);
        features.push_back(rel);
    }

    {
        cv::Point2d pup = pupils.rightCandidate().get().center;
        pup -= coordinateRoot;
        cv::Vec2d a(pup);
        cv::Vec2d b(rlwingcenter);
        double mag = cv::norm(b)*cv::norm(rleft - rright);
        cv::Point2d projpup;
        projpup = a.dot(b)*b/mag;
        double n1 = cv::norm(rlwingcenter);
        double n2 = cv::norm(projpup);
        double rel = (n1-n2) / (n1+n2);
        features.push_back(rel);
    }

    for (int i : {2, 3})
    { // projection of the nosetip on the root-nosewingcenter axis
      // feature: relation to this axis
        cv::Point2d nosetip;
        nosetip = faceParts.featurePolygon(FaceParts::NOSE).at(i);
        nosetip -= coordinateRoot;
        cv::Vec2d a(nosetip);
        cv::Vec2d b(rlwingcenter);
        double mag = b.dot(b);
        cv::Point2d projcenter;
        projcenter = a.dot(b)*b/mag;
        double n1 = cv::norm(rlwingcenter);
        double n2 = cv::norm(projcenter);
        double rel = (n1-n2) / (n1+n2);
        features.push_back(rel);
        //cv::circle(ghyp.parentHyp.frame, projcenter+coordinateRoot, 3, cv::Scalar(255, 10, 50), 1, 'A');
        //cv::circle(ghyp.parentHyp.frame, rlwingcenter+coordinateRoot, 3, cv::Scalar(255, 255, 50), 1, 'A');
    }

    for (int i : {0, 1, 2, 14, 15, 16})
    { // projection of the jaw on the root-nosewingcenter axis
      // feature: relation to this axis
        cv::Point2d jaw;
        jaw = faceParts.featurePolygon(FaceParts::JAW).at(i);
        jaw -= coordinateRoot;
        cv::Vec2d a(jaw);
        cv::Vec2d b(rlwingcenter);
        double mag = b.dot(b);
        cv::Point2d projjaw;
        projjaw = a.dot(b)*b/mag;
        double n1 = cv::norm(rlwingcenter);
        double n2 = cv::norm(projjaw);
        double rel = (n1-n2) / (n1+n2);
        features.push_back(rel);
        //cv::circle(ghyp.parentHyp.frame, projcenter+coordinateRoot, 3, cv::Scalar(255, 10, 50), 1, 'A');
        //cv::circle(ghyp.parentHyp.frame, rlwingcenter+coordinateRoot, 3, cv::Scalar(255, 255, 50), 1, 'A');
    }
    //features.push_back(ghyp.eyeLidClassification.get());
    ghyp.vertGazeFeatures = dlib::mat(features);
    //cerr << ghyp.vertGazeFeatures << endl;
}

