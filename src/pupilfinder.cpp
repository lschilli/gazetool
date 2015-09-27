#include "pupilfinder.h"
#include <future>

using namespace std;

//parameters
static constexpr int CANDIDATE_MAP_WIDTH = 48;
static constexpr double GRADIENT_THRESHOLD_FACTOR = 15;

class CenterDetector {

private:
    double getGradientThreshold(const cv::Mat &mat) {
        cv::Scalar meanGradMag, stdGradMag;
        cv::meanStdDev(mat, meanGradMag, stdGradMag);
        return meanGradMag[0] + stdGradMag[0] / sqrt(mat.rows * mat.cols) * GRADIENT_THRESHOLD_FACTOR;
    }

    void getGradientxy(const cv::Mat& in, cv::Mat& out, cv::Mat& mask) {
        static const cv::Mat derivkernelx = (cv::Mat_<float>(1,3)<<-0.5, 0, 0.5);
        static const cv::Mat derivkernely = (cv::Mat_<float>(3,1)<<-0.5, 0, 0.5);
        cv::Mat gradientX(in.size(), CV_32F);
        cv::Mat gradientY(in.size(), CV_32F);
        cv::filter2D(in, gradientX, CV_32F, derivkernelx);
        cv::filter2D(in, gradientY, CV_32F, derivkernely);
        cv::Mat sqaredMags = gradientX.mul(gradientX) + gradientY.mul(gradientY);
        //normalizing the gradient with squared magnitudes improves result
        std::vector<cv::Mat> planes = {gradientX/sqaredMags, gradientY/sqaredMags};
        cv::merge(planes, out);
        double thresh = getGradientThreshold(sqaredMags);
        cv::threshold(sqaredMags, sqaredMags, thresh, 255, cv::THRESH_BINARY);
        sqaredMags.convertTo(mask, CV_8UC1);
    }

    void getWeights(const cv::Mat& img, const cv::Mat& mask, cv::Mat& fweights) {
        cv::Mat weights; //weight map to prefer dark
        cv::GaussianBlur(img, weights, cv::Size(3, 3), 0, 0);
        double minval, maxval;
        cv::minMaxIdx(weights, &minval, &maxval, NULL, NULL, mask);
        weights.convertTo(fweights, CV_32F, 0.3/(maxval-minval), -0.3*minval/(maxval-minval));
    }

    float circleObjective(const cv::Mat& gradientxy, const cv::Mat& gradThreshMask, const cv::Point& c) {
        auto gradientp = gradientxy.ptr<cv::Vec2f>(0);
        auto maskp = gradThreshMask.ptr<uchar>(0);
        float result = 0;
        for (int y = 0; y < gradientxy.rows; y++) {
            for (int x = 0; x < gradientxy.cols; x++, gradientp++, maskp++) {
                if (!(*maskp)) continue;
                cv::Vec2f di = cv::normalize(cv::Vec2f(x - c.x, y - c.y)); // (xi - c)/||xi-c||
                const cv::Vec2f& gi = *gradientp;
                const float dotprod = di.dot(gi); // di^T*gi
                // only positive values, no square ( )^2
                result += max(dotprod, 0.0f);
            }
        }
        return result;
    }

    double scaleToFixedWidth(const cv::Mat &src,cv::Mat &dst, int interpolation) {
        double sf = CANDIDATE_MAP_WIDTH/double(src.cols);
        cv::resize(src, dst, cv::Size(CANDIDATE_MAP_WIDTH, sf*src.rows), 0, 0, interpolation);
        return sf;
    }

    void addDistanceTransformation(
            const cv::Mat& eyeROI, const cv::Mat& mask, cv::Mat& dst, FaceParts::FacePart eyeid) {
        cv::Mat eyeseg(eyeROI.clone());
        cv::GaussianBlur(eyeseg, eyeseg, cv::Size(3, 3), 0, 0);
        eyeseg &= mask;
        cv::threshold(eyeseg, eyeseg, 0, 255, CV_THRESH_BINARY_INV+CV_THRESH_OTSU);
        eyeseg &= mask;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(eyeseg.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::drawContours(eyeseg, contours, -1, cv::Scalar(255), -1);
        eyeseg &= mask;
        cv::Mat dsttran;
        cv::distanceTransform(eyeseg.clone(), dsttran, CV_DIST_L2, 0);
        cv::normalize(dsttran, dsttran, 0, 1, cv::NORM_MINMAX);
        //eyeseg.convertTo(cv::Mat(faceROIgray, cv::Rect(cv::Point((eyeid-FaceParts::REYE)*eyeseg.cols,0), eyeseg.size())), CV_8UC1);
        //dsttran.convertTo(cv::Mat(faceROIgray, cv::Rect(cv::Point((eyeid-FaceParts::REYE)*eyeseg.cols,0), eyeseg.size())), CV_8UC1, 255);
        dst += dsttran*0.8; //max. combined value is 1.8 for normalized dst
    }

    void getCandidateMap(const cv::Mat& img, const cv::Mat& innerMask, const cv::Mat& outerMask,
                         cv::Mat& gradientxy, cv::Mat& gradThreshMask, cv::Mat& candidates) {
        getGradientxy(img, gradientxy, gradThreshMask);
        gradThreshMask &= outerMask;
        candidates = cv::Mat::zeros(img.size(), CV_32F);
        cv::Mat weights;
        getWeights(img, innerMask, weights);
        auto objFuncMatPtr = candidates.ptr<float>(0);
        auto maskPtr = innerMask.ptr<uchar>(0);
        auto weightsPtr = weights.ptr<float>(0);
        float maxVal = 0;
        for (int cy = 0; cy < img.rows; cy++) {
            for (int cx = 0; cx < img.cols; cx++, objFuncMatPtr++, maskPtr++, weightsPtr++) {
                if (!*maskPtr) continue;
                *objFuncMatPtr = circleObjective(gradientxy, gradThreshMask, cv::Point(cx, cy))*(1-(*weightsPtr));
                maxVal = max(maxVal, *objFuncMatPtr);
            }
        }
        candidates /= maxVal;
    }

    float estimateRadius(const cv::Mat& gradientxy, const cv::Mat& gradThreshMask, const cv::Point& c) {
        auto gradientp = gradientxy.ptr<cv::Vec2f>(0);
        auto maskp = gradThreshMask.ptr<uchar>(0);
        vector<float> votes(gradientxy.rows*gradientxy.cols, 0);
        int cnt = 0;
        for (int y = 0; y < gradientxy.rows; y++) {
            for (int x = 0; x < gradientxy.cols; x++, gradientp++, maskp++) {
                if (!(*maskp)) continue;
                const cv::Vec2f di = cv::normalize(cv::Vec2f(x - c.x, y - c.y)); // (xi - c)/||xi-c||
                const cv::Vec2f& gi = *gradientp;
                const float dotprod = di.dot(gi); // di^T*gi
                const int vpos = sqrt((x-c.x)*(x-c.x)+(y-c.y)*(y-c.y));
                votes[vpos] += abs(dotprod);
                cnt++;
            }
        }
        auto biggest = max_element(votes.begin(), votes.end());
        int pos = distance(begin(votes), biggest);
        return pos;
    }

public:
    boost::optional<PupilFinder::CenterCandidate> findEyeCenter(
                const cv::Mat& face, const std::vector<cv::Point>& poly,
                const cv::Rect& eye, FaceParts::FacePart eyeid) {
        cv::Mat eyeROIUnscaled = face(eye);
        cv::equalizeHist(eyeROIUnscaled, eyeROIUnscaled);
        eyeROIUnscaled.copyTo(face(eye));

        cv::Mat eyeROI;
        double sf = scaleToFixedWidth(eyeROIUnscaled, eyeROI, cv::INTER_LINEAR);

        std::vector<cv::Point> scaledPoly;
        for (const auto& p : poly) {
            scaledPoly.push_back(cv::Point(round(sf*p.x), round(sf*p.y)));
        }
        cv::Mat polyMask(eyeROI.size(), CV_8UC1, cv::Scalar(0));
        cv::fillConvexPoly(polyMask, scaledPoly, cv::Scalar(255));
        cv::RotatedRect ellrect = cv::fitEllipse(scaledPoly);
        ellrect.size.width *= 1.05;
        ellrect.size.height *= 1.05;
        cv::Mat ellipseMask(eyeROI.size(), CV_8UC1, cv::Scalar(0));
        cv::ellipse(ellipseMask, ellrect, cv::Scalar(255), -1);
        cv::Mat gradientxy, gradThreshMask, cndMap;
        getCandidateMap(eyeROI, polyMask, ellipseMask, gradientxy, gradThreshMask, cndMap);
        addDistanceTransformation(eyeROI, polyMask, cndMap, eyeid);

        //we prepare for calculating the center of mass for regions:
        //>0.98 will use gradient values even if distances are zero
        //<1.6 will truncate outliers / noise: truncated mean idea
        cv::Mat cndMapOrig;
        cv::threshold(cndMap, cndMapOrig, 0.98, 0, cv::THRESH_TOZERO);
        cv::threshold(cndMapOrig, cndMap, 1.6, 0, cv::THRESH_TOZERO_INV);
        cv::Moments mu = cv::moments(cndMap, true);
        cndMap.convertTo(cv::Mat(face, cv::Rect(cv::Point((eyeid-FaceParts::REYE)
                         *(face.cols-cndMap.cols), face.rows-cndMap.rows), cndMap.size())), CV_8U, 146.0);
        boost::optional<PupilFinder::CenterCandidate> possibleCandidate;
        if (mu.m00 != 0) {
            cv::Vec2f centervec = cv::Vec2f(mu.m10, mu.m01)/mu.m00;
            cv::Point p(centervec[0], centervec[1]);
            //if (cndMapOrig.at<float>(p) != 0) {
                PupilFinder::CenterCandidate centCand;
                centCand.center = cv::Point2d(centervec[0]/sf, centervec[1]/sf);
                double rad = sqrt(mu.m00/CV_PI);
                //double rad = estimateRadius(gradientxy, gradThreshMask, p);
                centCand.radius = rad/sf;
                possibleCandidate = centCand;
            //}
        }
        return possibleCandidate;
    }

};

PupilFinder::PupilFinder()
{
}

PupilFinder::PupilFinder(cv::Mat &frame, const FaceParts &faceParts)
{
    //select subrectangle containing some facial features
    std::vector<cv::Point> fpoly;
    for (const auto& fp : {FaceParts::LBROW, FaceParts::RBROW, FaceParts::REYE,
                           FaceParts::LEYE, FaceParts::NOSEWINGS}) {
        const auto& pol = faceParts.featurePolygon(fp);
        fpoly.insert(fpoly.end(), pol.begin(), pol.end());
    }
    frect = cv::boundingRect(cv::Mat(fpoly));

    lepoly = faceParts.featurePolygon(FaceParts::LEYE);
    lebounds = faceParts.boundingRect(FaceParts::LEYE);
    repoly = faceParts.featurePolygon(FaceParts::REYE);
    rebounds = faceParts.boundingRect(FaceParts::REYE);
    scalefac = setupFaceRegion(frame, frect, lebounds, rebounds);

    lpupCandidate = findEye(lepoly, lebounds, FaceParts::LEYE);
    if (lpupCandidate.is_initialized()) {
        pupfound++;
    }
    rpupCandidate = findEye(repoly, rebounds, FaceParts::REYE);
    if (rpupCandidate.is_initialized()) {
        pupfound++;
    }
    cv::cvtColor(faceROIgray, faceROIcolor, CV_GRAY2BGR);
}

void PupilFinder::drawCross(cv::Mat img, cv::Point center, cv::Scalar color, int d, int thickness, int lineType) {
    cv::line(img, cv::Point(center.x - d, center.y), cv::Point(center.x + d, center.y), color, thickness, lineType);
    cv::line(img, cv::Point(center.x, center.y - d), cv::Point(center.x, center.y + d), color, thickness, lineType);
}

cv::Mat PupilFinder::faceRegion()
{
    return faceROIcolor;
}

cv::Rect PupilFinder::faceRect()
{
    return frect;
}

cv::Rect PupilFinder::leftEyeBounds() const
{
    return lebounds;
}

cv::Rect PupilFinder::rightEyeBounds() const
{
    return rebounds;
}

const boost::optional<PupilFinder::CenterCandidate>& PupilFinder::rightCandidate()
{
    return rpupCandidate;
}

const boost::optional<PupilFinder::CenterCandidate>& PupilFinder::leftCandidate()
{
    return lpupCandidate;
}

int PupilFinder::pupilsFound() const
{
    return pupfound;
}

void PupilFinder::draw(cv::Mat &frame)
{
    if (lpupCandidate.is_initialized()) {
        CenterCandidate& pup = lpupCandidate.get();
        drawCross(frame, pup.center, cv::Scalar(0, 255, 0), 1, 1, 'A');
        cv::circle(frame, pup.center, round(pup.radius), cv::Scalar(0, 255, 0), 1, 'A');
    }
    if (rpupCandidate.is_initialized()) {
        CenterCandidate& pup = rpupCandidate.get();
        drawCross(frame, pup.center, cv::Scalar(0, 255, 0), 1, 1, 'A');
        cv::circle(frame, pup.center, round(pup.radius), cv::Scalar(0, 255, 0), 1, 'A');
    }
}

boost::optional<PupilFinder::CenterCandidate> PupilFinder::findEye(std::vector<cv::Point> epoly, cv::Rect_<double> eyerect, FaceParts::FacePart eyeid)
{
    cv::Point2d faceoffs;
    faceoffs = frect.tl();
    for (auto& p : epoly) {
        p.x = ((p.x - faceoffs.x) * scalefac + faceoffs.x);
        p.y = ((p.y - faceoffs.y) * scalefac + faceoffs.y);
    }
    eyerect.x = (eyerect.x - faceoffs.x) * scalefac + faceoffs.x;
    eyerect.y = (eyerect.y - faceoffs.y) * scalefac + faceoffs.y;
    eyerect.width *= scalefac;
    eyerect.height *= scalefac;
    cv::Point2d lexpand(eyerect.width*0.1, eyerect.height*0.3);
    for (auto& p : epoly) {
        cv::Point offset(eyerect.x-lexpand.x, eyerect.y-lexpand.y);
        p -= offset;
    }
    eyerect -= faceoffs + lexpand;
    //we round here, thats the accuracy we finally have
    eyerect.width = floor(eyerect.width*1.2);
    eyerect.height = floor(eyerect.height*1.6);

    boost::optional<CenterCandidate> pupilcandidate;
    cv::Rect froirect(cv::Point(0, 0), faceROIgray.size());
    if (!froirect.contains(eyerect.tl()) || !froirect.contains(eyerect.br())) {
        return pupilcandidate;
    }
    //find and draw eye centers
    CenterDetector cdet;
    pupilcandidate = cdet.findEyeCenter(faceROIgray, epoly, eyerect, eyeid);
    // draw eye region
    //cv::rectangle(face, eye, cv::Scalar(150));
    //cv::polylines(faceROI, origpoly, false, cv::Scalar(255), 1, 'A');
    if (pupilcandidate.is_initialized()) {
        CenterCandidate& pupil = pupilcandidate.get();
        pupil.center += eyerect.tl();
        drawCross(faceROIgray, pupil.center, cv::Scalar(255), 1, 1, 'A');
        cv::circle(faceROIgray, pupil.center, round(pupil.radius), cv::Scalar(255), 1, 'A');
        pupil.center = cv::Point2d(pupil.center.x / scalefac, pupil.center.y / scalefac) + faceoffs;
        pupil.radius /= scalefac;
    }
    return pupilcandidate;
}


double PupilFinder::setupFaceRegion(const cv::Mat& frame, const cv::Rect& facerect,
                      const cv::Rect& lebounds, const cv::Rect& rebounds) {
    cv::Rect framerect(cv::Point(0, 0), frame.size());
    double scaleFactor = 1.0;
    if (framerect.contains(facerect.tl()) && framerect.contains(facerect.br())) {
        cv::cvtColor(frame(facerect), faceROIgray, CV_BGR2GRAY);
        int mineyewidth = std::max(lebounds.width, rebounds.width);
        if (mineyewidth) {
            scaleFactor = CANDIDATE_MAP_WIDTH/double(mineyewidth);
            cv::Size nsize(round(faceROIgray.cols*scaleFactor), round(faceROIgray.rows*scaleFactor));
            // using double size for for visualization purposes and
            // to minimize errors in subsequent scale operations
            nsize.width *= 2;
            nsize.height *= 2;
            scaleFactor = nsize.width/double(faceROIgray.cols);
            cv::Mat tmp;
            cv::resize(faceROIgray, tmp, nsize, cv::INTER_LINEAR);
            faceROIgray = tmp;
        }
    }
    return scaleFactor;
}
