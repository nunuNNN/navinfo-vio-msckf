#include "global_param.h"



#include <iostream>
#include <mutex>
#include <fstream>

using namespace std;

bool flag_thread_start[2] = {true, false};


Parameter_camera_t params_camera;
Eigen::Isometry3d T_cam0_from_imu;
msckf_vio::Processor_config_t processor_config;
msckf_vio::Parameter_estimate_t params_estimste;

void InitParams(double f_forw, double cx_forw, double cy_forw)
{
    /*********** 初始化相机参数(内参) ***********/
    params_camera.m_f = f_forw;
    params_camera.m_cx = cx_forw;
    params_camera.m_cy = cy_forw;

    params_camera.m_k1 = -0.28340811;
    params_camera.m_k2 = 0.07395907;
    params_camera.m_p1 = 0.00019359;
    params_camera.m_p2 = 1.76187114e-05;

     /*********** 初始化imu相机的外参数 ***********/
    Eigen::Matrix3d R_cam0_imu_f; 
    Eigen::Vector3d t_cam0_imu_f;
    R_cam0_imu_f << 0.014865542981794,   0.999557249008346,  -0.025774436697440,
                    -0.999880929698575,   0.014967213324719,   0.003756188357967,
                    0.004140296794224,   0.025715529947966,   0.999660727177902;
    t_cam0_imu_f << 0.065222909535531, -0.020706385492719, -0.008054602460030;

    T_cam0_from_imu = Eigen::Isometry3d::Identity();
    T_cam0_from_imu.rotate(R_cam0_imu_f);
    T_cam0_from_imu.pretranslate(t_cam0_imu_f);


    /****** 初始化前端参数(特征提取及匹配) ******/
    // Image Processor parameters
    processor_config.grid_row = 8;
    processor_config.grid_col = 6;
    processor_config.grid_min_feature_num = 2;
    processor_config.grid_max_feature_num = 4;
    processor_config.pyramid_levels = 3;
    processor_config.patch_size = 15;
    processor_config.fast_threshold = 10; //10
    processor_config.max_iteration = 30;
    processor_config.track_precision = 0.01;
    processor_config.ransac_threshold = 3;
    processor_config.stereo_threshold = 5;

    /****** 初始化后端参数(协方差及门限值) ******/
    // Noise related parameters
    params_estimste.gyro_noise = 0.005;
    params_estimste.acc_noise = 0.05;
    params_estimste.gyro_bias_noise = 0.001;
    params_estimste.acc_bias_noise = 0.01;
    params_estimste.observation_noise = 0.05;

    // These values should be covariance
    params_estimste.velocity_cov = 0.5;
    params_estimste.gyro_bias_cov = 0.02;
    params_estimste.acc_bias_cov = 0.02;

    params_estimste.extrinsic_rotation_cov = 3.0462e-2;
    params_estimste.extrinsic_translation_cov = 2.5e-5;

    // Maximum number of camera states to be stored
    params_estimste.max_cam_state_size = 20;

    params_estimste.position_std_threshold = 8.0;

    params_estimste.rotation_threshold = 0.2618;
    params_estimste.translation_threshold = 0.4;
    params_estimste.tracking_rate_threshold = 0.5;

    // Feature optimization parameters
    params_estimste.feature_trans_threshold = -1;

}
