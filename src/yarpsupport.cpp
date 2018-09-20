#include "yarpsupport.h"

using namespace std;
using namespace yarp::os;

/**
 * @brief YarpImageProvider::YarpImageProvider
 */

YarpImageProvider::YarpImageProvider() {}

YarpImageProvider::YarpImageProvider(const std::string &portname)
{
    if (!imagePort.open(portname + "/in")) throw runtime_error("Could not open yarp port " + portname + "/in");
}

bool YarpImageProvider::get(cv::Mat& frame) {
    if (imagePort.isClosed()) return false;
    yarp::sig::ImageOf<yarp::sig::PixelRgb> *image = imagePort.read(true);
    if (image) {
        cv::Mat rgbframe = cv::cvarrToMat(static_cast<IplImage*>(image->getIplImage()));
        cv::cvtColor(rgbframe, frame, CV_RGB2BGR);
        return true;
    }
    return false;
}

string YarpImageProvider::getLabel()
{
    return "";
}

string YarpImageProvider::getId()
{
    return "";
}

YarpImageProvider::~YarpImageProvider()
{
    imagePort.close();
    yarp.fini();
}


YarpSender::YarpSender(const std::string& portname)
{
    if (!port.open(portname + "/out")) throw runtime_error("Could not open yarp port " + portname + "/out");
}

void YarpSender::sendGazeHypotheses(GazeHypsPtr hyps)
{
    if (port.isClosed()) return;
    Bottle& b = port.prepare();
    b.clear();
    Bottle& allfaces = b.addList();
    allfaces.add("faces");
    for (GazeHyp& ghyp : *hyps) {
        Bottle& bghyp = allfaces.addList();
        bghyp.addString("face");
        {   Bottle& bfacerect = bghyp.addList();
            auto fr = ghyp.pupils.faceRect();
            bfacerect.addString("facerect");
            bfacerect.addDouble(fr.x);
            bfacerect.addDouble(fr.y);
            bfacerect.addDouble(fr.width);
            bfacerect.addDouble(fr.height);
        }
        {   Bottle& brelgaze = bghyp.addList();
            brelgaze.addString("gaze");
            if (ghyp.horizontalGazeEstimation.is_initialized()) {
                brelgaze.addDouble(ghyp.horizontalGazeEstimation.get());
            }
        }
        {   Bottle& blid = bghyp.addList();
            blid.addString("lid");
            if (ghyp.eyeLidClassification.is_initialized()) {
                blid.addDouble(ghyp.eyeLidClassification.get());
            }
        }
        {   Bottle& bmutgaze = bghyp.addList();
            bmutgaze.addString("mutualgaze");
            if (ghyp.isMutualGaze.is_initialized()) {
                bmutgaze.addInt8(ghyp.isMutualGaze.get());
            }
        }
        {   Bottle& lidclosed = bghyp.addList();
            lidclosed.addString("lidclosed");
            if (ghyp.isLidClosed.is_initialized()) {
                lidclosed.addIn8(ghyp.isLidClosed.get());
            }
        }
    }
    auto ts = double(chrono::duration_cast<chrono::nanoseconds>(hyps->frameTime.time_since_epoch()).count())*1.0e-9;
    Stamp yts(hyps->frameCounter, ts);
    port.setEnvelope(yts);
    port.write();
}

YarpSender::~YarpSender()
{
    port.close();
    yarp.fini();
}
