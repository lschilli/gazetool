#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

class ImageProvider
{
  public:
    virtual ~ImageProvider() {}
    virtual bool get(cv::Mat& frame) = 0;
    virtual std::string getLabel() = 0;
    virtual std::string getId() = 0;

  protected:
    cv::Mat image;
};

class CvVideoImageProvider : public ImageProvider
{
public:
    CvVideoImageProvider();
    CvVideoImageProvider(int camera, cv::Size size, int desiredFps);
    CvVideoImageProvider(const std::string& infile, cv::Size size);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~CvVideoImageProvider() {}
private:
    cv::VideoCapture capture;
    cv::Size desiredSize;
};

class BatchImageProvider : public ImageProvider
{
public:
    BatchImageProvider();
    BatchImageProvider(const std::string& batchfile);
    BatchImageProvider(const std::vector<std::string>& filelist);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~BatchImageProvider() {}

protected:
    int position;
    std::vector<std::string> filenames;
    std::vector<std::string> labels;
};

