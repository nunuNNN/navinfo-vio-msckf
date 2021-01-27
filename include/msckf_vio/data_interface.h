#ifndef MSCKF_VIO_DATA_INTERFACE_H
#define MSCKF_VIO_DATA_INTERFACE_H

#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

namespace msckf_vio 
{

// 接口数据类型,前后端的接口,前后端都需要访问
typedef struct
{
    uint32_t id;
    float u0;
    float v0;
    float u_vel;
    float v_vel;
} Mono_feature_t;
typedef struct
{
    double stamp;
    std::vector<Mono_feature_t> features;
} Feature_measure_t;

typedef struct 
{
    Eigen::Isometry3d T_w_b;
    Eigen::Vector3d v_in_world;
} Translation_velocity_t;
typedef struct
{
    Translation_velocity_t T_vel;
    Eigen::Vector3d bias_acc;
    Eigen::Vector3d bias_gyr;
} Ground_truth_t;


/* 前后端加载的参数*/
typedef struct
{
    double position_std_threshold;

    double rotation_threshold;
    double translation_threshold;
    double tracking_rate_threshold;

    // Feature optimization parameters
    double feature_trans_threshold;

    // Noise related parameters
    double gyro_noise;
    double acc_noise;
    double gyro_bias_noise;
    double acc_bias_noise;
    double observation_noise;

    // covariance
    double velocity_cov;
    double gyro_bias_cov;
    double acc_bias_cov;
    double extrinsic_rotation_cov;
    double extrinsic_translation_cov;

    // Maximum number of camera states to be stored
    int max_cam_state_size;
} Parameter_estimate_t;

/*
   * @brief ProcessorConfig Configuration parameters for feature detection and tracking.
*/
typedef struct
{
    int32_t grid_row;
    int32_t grid_col;
    int32_t grid_min_feature_num;
    int32_t grid_max_feature_num;

    int32_t pyramid_levels;
    int32_t patch_size;
    int32_t fast_threshold;
    int32_t max_iteration;
    double track_precision;
    double ransac_threshold;
    double stereo_threshold;
} Processor_config_t;


} // namespace msckf_vio

//1. 相机的内参
typedef struct
{
    double m_f;
    // center x
    double m_cx;
    // center y
    double m_cy;

    double m_k1;
    double m_k2;
    double m_p1;
    double m_p2;
} Parameter_camera_t;


//2. 数据流 图片\IMU\PVQB结构体
typedef struct
{
    double timestamp;
    cv::Mat left_rect_image;
} StrImageData;

typedef struct
{
    double timestamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
} StrImuData;

typedef struct
{
    double timestamp;
    Eigen::Isometry3d T_world_from_imu;
    Eigen::Vector3d v_in_world;
    Eigen::Vector3d bias_acc;
    Eigen::Vector3d bias_gyr;
} StrPVQB;


// 前后端结构体,包含同一个时刻的pvqb和特征点
typedef struct
{
    double curr_time;
    int camera_id;
    msckf_vio::Feature_measure_t curr_features;
    std::vector<StrImuData> curr_from_last_imu;
    StrPVQB curr_pvqb;
} Feature_and_PVQB_t;

typedef std::shared_ptr<msckf_vio::Feature_measure_t> FeatureMeasurePtr;

#endif // MSCKF_VIO_DATA_INTERFACE_H
