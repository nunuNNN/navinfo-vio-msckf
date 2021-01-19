
#ifndef __INTERFACE_MSCKF_VIO_H__
#define __INTERFACE_MSCKF_VIO_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


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
                            const std::vector<Eigen::Vector3d> &curr_init_pts));

// send imu data without bias
void VISION_MsckfVio_SendImu(uint64_t timestamp,
                             const Eigen::Vector3d &acc,
                             const Eigen::Vector3d &gyr);

// send gps data and cov
void VISION_MsckfVio_SendGps(uint64_t timestamp,
                             const Eigen::Vector3d &lla,
                             const Eigen::Matrix3d &cov);

// send left image after rectified,
void VISION_MsckfVio_SendMonoImage(uint64_t timestamp,
                            const cv::Mat &left_image_rectified);

// send PVQ from flight control
void VISION_MsckfVio_SendPVQB(
    uint64_t timestamp,
    const Eigen::Isometry3d &T_world_from_imu,
    const Eigen::Vector3d &velocity_in_world,
    const Eigen::Vector3d &bias_acc,
    const Eigen::Vector3d &bias_gyr);

// stop
void VISION_MsckfVio_Stop();

#endif //__INTERFACE_MSCKF_VIO_H__
