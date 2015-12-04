#include "imageprovider.h"

#include <fstream>
#include <boost/tokenizer.hpp>


using namespace std;

typedef boost::tokenizer<boost::char_separator<char> > CharTokenizer;

/**
 * @brief CvVideoImageProvider::CvVideoImageProvider
 */

CvVideoImageProvider::CvVideoImageProvider()
{
}

CvVideoImageProvider::CvVideoImageProvider(int camera, cv::Size size, int desiredFps)
{
    capture = cv::VideoCapture(camera);
    desiredSize = size;
    if (size != cv::Size()) {
        capture.set(CV_CAP_PROP_FRAME_WIDTH, size.width);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, size.height);
    }
    if (desiredFps != 0) {
        capture.set(CV_CAP_PROP_FPS, desiredFps);
    }
}

CvVideoImageProvider::CvVideoImageProvider(const string &infile, cv::Size size)
{
    capture = cv::VideoCapture(infile);
    desiredSize = size;
}

bool CvVideoImageProvider::get(cv::Mat &frame)
{
    bool ret = capture.read(frame);
    if (ret && desiredSize != cv::Size() && frame.size() != desiredSize) {
        cv::Mat tmp;
        cv::resize(frame, tmp, desiredSize, 0, 0, cv::INTER_LINEAR);
        frame = tmp;
    }
    return ret;
}

string CvVideoImageProvider::getLabel()
{
    return "";
}

string CvVideoImageProvider::getId()
{
    return "";
}



/**
 * @brief BatchImageProvider::BatchImageProvider
 */
BatchImageProvider::BatchImageProvider() : position(0)
{
}

BatchImageProvider::BatchImageProvider(const string &batchfile) : position(-1)
{
    ifstream batchfs(batchfile);
    if (!batchfs.is_open()) {
        throw runtime_error(string("Cannot open file list " + batchfile));
    }
    string line;
    boost::char_separator<char> fieldsep("\t", "", boost::keep_empty_tokens);
    while (getline(batchfs, line)) {
        if (line != "") {
            CharTokenizer tokenizer(line, fieldsep);
            auto it = tokenizer.begin();
            filenames.push_back(*it++);
            if (it != tokenizer.end()) {
                labels.push_back(*it);
            } else {
                labels.push_back("");
            }
            //cerr << filenames.back() << " " << labels.back() << endl;
        }
    }
}

BatchImageProvider::BatchImageProvider(const std::vector<string> &filelist)
    : position(-1), filenames(filelist)
{
}

bool BatchImageProvider::get(cv::Mat &frame)
{
    if (position < (int)filenames.size()-1) {
        position++;
        string filename(filenames.at(position));
        cv::Mat tmp(cv::imread(filename));
        frame = tmp;
        if (frame.empty()) {
            throw runtime_error(string("Cannot read image from " + filename));
        } else {
            return true;
        }
    }
    frame = cv::Mat();
    return false;
}

string BatchImageProvider::getLabel()
{
    if (position < (int)labels.size() && position >= 0) {
        return labels[position];
    }
    return "";
}

string BatchImageProvider::getId()
{
    if (position < (int)filenames.size() && position >= 0) {
        return filenames[position];
    }
    return "";
}

