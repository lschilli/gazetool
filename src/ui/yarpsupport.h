#pragma once

#include <yarp/os/all.h>
#include <yarp/sig/all.h>

#include "imageprovider.h"
#include "gazehyps.h"

class YarpImageProvider : public ImageProvider
{
public:
    YarpImageProvider();
    YarpImageProvider(const std::string& portname);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~YarpImageProvider();

protected:
    yarp::os::Network yarp;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > imagePort;

};

class YarpSender {

public:
    YarpSender(const std::string& portname);
    void sendGazeHypotheses(GazeHypsPtr hyps);
    ~YarpSender();

private:
    yarp::os::Network yarp;
    yarp::os::BufferedPort<yarp::os::Bottle> port;

};
