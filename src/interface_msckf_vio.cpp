#include "interface_msckf_vio.h"

#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "global_param.h"
#include "vio_manager.h"

#include "imu_gps_localizer/imu_gps_localizer.h"
#include "imu_gps_localizer/base_type.h"

using namespace std;
using namespace cv;
using namespace Eigen;

std::unique_ptr<ImuGpsLocalization::ImuGpsLocalizer> imu_gps_localizer_ptr_;

// init 3d optical flow, the PublishMsckfVio is
// the callback function to publish the result to flight control
int VISION_MsckfVio_Init(float f_forw, float cx_forw, float cy_forw, float baseline_forw,
                        void (*PublishMsckfVio)(
                            uint64_t timestamp,
                            const Eigen::Vector3d &p,
                            const Eigen::Quaterniond &q,
                            float covVx, float covVy, float covVz,
                            uint8_t resetFlag, float rate1, float rate2),
                        void (*PublishPoints)(
                            uint64_t timestamp, 
                            const std::vector<int> &curr_init_ids,
                            const std::vector<Eigen::Vector2d> &curr_init_obs,
                            const std::vector<Eigen::Vector3d> &curr_init_pts))
{
    // cout << "1 VISION_MsckfVio_Init" << endl;
    InitParams(f_forw, cx_forw, cy_forw, baseline_forw);

    VioManager::getInstance()->InitVioManager(PublishMsckfVio, PublishPoints);



    double acc_noise, gyro_noise, acc_bias_noise, gyro_bias_noise;
    acc_noise = 1e-2;
    gyro_noise = 1e-4;
    acc_bias_noise = 1e-6;
    gyro_bias_noise = 1e-8;

    const Eigen::Vector3d I_p_Gps(0., 0., 0.);

    // Initialization imu gps localizer.
    imu_gps_localizer_ptr_ = 
            std::unique_ptr<ImuGpsLocalization::ImuGpsLocalizer>( new 
                          ImuGpsLocalization::ImuGpsLocalizer(acc_noise, gyro_noise,
                                                              acc_bias_noise, gyro_bias_noise,
                                                              I_p_Gps) );

    return 0;
}

// send imu data without bias
void VISION_MsckfVio_SendImu(uint64_t timestamp,
                             const Vector3d &acc,
                             const Vector3d &gyr)
{
    VioManager::getInstance()->PushImu(timestamp * 1e-9, acc, gyr);

    ImuGpsLocalization::ImuDataPtr imu_data_ptr = std::make_shared<ImuGpsLocalization::ImuData>();
    imu_data_ptr->timestamp = timestamp;
    imu_data_ptr->acc = acc;
    imu_data_ptr->gyro = gyr;
    ImuGpsLocalization::State fused_state;
    const bool ok = imu_gps_localizer_ptr_->ProcessImuData(imu_data_ptr, &fused_state);
    if (!ok) {
        return;
    }

    // 将imu与gps融合定位出来的值给vio系统使用
    // VioManager::getInstance()->PushPVQB(timestamp * 1e-9, T_world_from_imu, velocity_in_world, bias_acc, bias_gyr);

}

// send gps data and cov
void VISION_MsckfVio_SendGps(uint64_t timestamp,
                             const Eigen::Vector3d &lla,
                             const Eigen::Matrix3d &cov)
{
    // GPS的消息格式被简化了，后面可以丰富
    ImuGpsLocalization::GpsPositionDataPtr gps_data_ptr = std::make_shared<ImuGpsLocalization::GpsPositionData>();
    gps_data_ptr->timestamp = timestamp;
    gps_data_ptr->lla = lla;
    gps_data_ptr->cov = cov;

    imu_gps_localizer_ptr_->ProcessGpsPositionData(gps_data_ptr);
}

// send left image after rectified
void VISION_MsckfVio_SendMonoImage(uint64_t timestamp,
                            const cv::Mat &left_image_rectified)
{
    static double lasttime = timestamp * 1e-9;
    const double currtime = timestamp * 1e-9;
    VioManager::getInstance()->PushImage(currtime, left_image_rectified);

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
