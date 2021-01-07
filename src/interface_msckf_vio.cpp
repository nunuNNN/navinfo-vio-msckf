#include "interface_msckf_vio.h"

#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "global_param.h"
#include "vio_manager.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// init 3d optical flow, the PublishMsckfVio is
// the callback function to publish the result to flight control
int VISION_MsckfVio_Init(float f_forw, float cx_forw, float cy_forw, float baseline_forw,
                         void (*PublishMsckfVio)(
                             uint64_t timestamp,
                             const Eigen::Vector3d &p,
                             const Eigen::Quaterniond &q,
                             float covVx, float covVy, float covVz,
                             uint8_t resetFlag,
                             float harrisVal, float rate1, float rate2))
{
    // cout << "1 VISION_MsckfVio_Init" << endl;
    InitParams(f_forw, cx_forw, cy_forw, baseline_forw);

    VioManager::getInstance()->InitVioManager(PublishMsckfVio);

    return 0;
}

// send imu data without bias
void VISION_MsckfVio_SendImu(uint64_t timestamp,
                             const Vector3d &acc,
                             const Vector3d &gyr)
{
    VioManager::getInstance()->PushImu(timestamp * 1e-9, acc, gyr);
}

// send left and right images after rectified, and the depth image
void VISION_MsckfVio_SendStereoAndDepthImage(uint64_t timestamp,
                                            const cv::Mat &left_image_rectified,
                                            const cv::Mat &right_image_rectified,
                                            const cv::Mat &depth_image)
{
    static double lasttime = timestamp * 1e-9;
    const double currtime = timestamp * 1e-9;
    VioManager::getInstance()->PushImage(currtime, left_image_rectified, right_image_rectified, depth_image);

    if (currtime - lasttime > 0.1 || currtime - lasttime < 0.0)
    {
        // printf("ERROR FORW image dt is big or current is old! last:%f curr:%f dt:%f\r\n", currtime, lasttime, currtime - lasttime);
    }
    lasttime = currtime;
}

// send PVQ from flight control
void VISION_MsckfVio_SendPVQB(uint64_t timestamp,
                            const Eigen::Isometry3d &T_world_from_imu,
                            const Vector3d &velocity_in_world,
                            const Eigen::Vector3d &bias_acc,
                            const Eigen::Vector3d &bias_gyr)
{
    // cout << fixed << timestamp
    //      << ", t: " << T_world_from_imu.translation().transpose()
    //      << ", v: " << velocity_in_world.transpose() << endl;
     VioManager::getInstance()->PushPVQB(timestamp * 1e-9, T_world_from_imu, velocity_in_world, bias_acc, bias_gyr);
}

// stop
void VISION_MsckfVio_Stop()
{
    VioManager::getInstance()->ReleaseVioManager();
}
