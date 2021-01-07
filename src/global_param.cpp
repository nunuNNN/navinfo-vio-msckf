#include "global_param.h"



#include <iostream>
#include <mutex>
#include <fstream>

using namespace std;

bool flag_thread_start[2] = {true, false};


Parameter_camera_t params_camera;
Parameter_extrinsic_t params_extrinsic;
msckf_vio::Processor_config_t processor_config;
msckf_vio::Parameter_estimate_t params_estimste;

void InitParams(double f_forw, double cx_forw, double cy_forw, double baseline_forw)
{
    /*********** 初始化相机参数(内参) ***********/
    params_camera.m_f = f_forw;
    params_camera.m_cx = cx_forw;
    params_camera.m_cy = cy_forw;
    params_camera.m_baseline = baseline_forw;

     /*********** 初始化imu相机的外参数 ***********/
    Eigen::Matrix3d R_cam0_imu_f; 
    Eigen::Vector3d t_cam0_imu_f;
    R_cam0_imu_f << 0.014865542981794,   0.999557249008346,  -0.025774436697440,
                    -0.999880929698575,   0.014967213324719,   0.003756188357967,
                    0.004140296794224,   0.025715529947966,   0.999660727177902;
    t_cam0_imu_f << 0.065222909535531, -0.020706385492719, -0.008054602460030;
    Eigen::Matrix3d R_cam1_imu_f;
    Eigen::Vector3d t_cam1_imu_f;
    R_cam1_imu_f << 0.012555267089103,   0.999598781151433,  -0.025389800891747,
                    -0.999755099723116,   0.013011905181504,   0.017900583825251,
                    0.018223771455443,   0.025158836311552,   0.999517347077547;
    t_cam1_imu_f << -0.044901980682509, -0.020569771258915, -0.008638135126028;
    Eigen::Matrix3d R_cam0_cam1_f;
    Eigen::Vector3d t_cam0_cam1_f;
    R_cam0_cam1_f << 0.999997256477881,   0.002312067192424,   0.000376008102415,
                    -0.002317135723281,   0.999898048506644,   0.014089835846648,
                    -0.000343393120525,  -0.014090668452714,   0.999900662637729;
    t_cam0_cam1_f << -0.110073808127187, -0.020569771258915, -0.008638135126028;

    params_extrinsic.T_cam0_from_imu = Eigen::Isometry3d::Identity();
    params_extrinsic.T_cam0_from_imu.rotate(R_cam0_imu_f);
    params_extrinsic.T_cam0_from_imu.pretranslate(t_cam0_imu_f);

    params_extrinsic.T_cam1_from_imu = Eigen::Isometry3d::Identity();
    params_extrinsic.T_cam1_from_imu.rotate(R_cam1_imu_f);
    params_extrinsic.T_cam1_from_imu.pretranslate(t_cam1_imu_f);

    params_extrinsic.T_cam1_form_cam0 = Eigen::Isometry3d::Identity();
    params_extrinsic.T_cam1_form_cam0.rotate(R_cam0_cam1_f);
    params_extrinsic.T_cam1_form_cam0.pretranslate(t_cam0_cam1_f);

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
