//
// Created by hyj on 17-6-22.
//

#ifndef IMUSIM_PARAM_H
#define IMUSIM_PARAM_H

#include <eigen3/Eigen/Core>

class Param{

public:

    Param();

    // time
    int imu_frequency = 200;
    int cam_frequency = 20;
    double imu_timestep = 1./imu_frequency;
    double cam_timestep = 1./cam_frequency;
    double t_start = 0.;
    double t_end = 20;  //  20 s

    // noise
    double gyro_bias_sigma = 1.0e-4;
    double acc_bias_sigma = 1.0e-4;

    double gyro_noise_sigma = 0.01;    // rad/s * 1/sqrt(hz)
    double acc_noise_sigma = 0.0196;      //　m/(s^2) * 1/sqrt(hz)

    double pixel_noise = 1;              // 1 pixel noise

    // cam f
    double fx = 574.247559;
    double fy = 574.247559;
    double cx = 242.220688;
    double cy = 321.135437;
    double image_w = 500;
    double image_h = 640;


    // 外参数
    Eigen::Matrix3d R_bc;   // cam to body
    Eigen::Vector3d t_bc;     // cam to body

    Eigen::Matrix3d R_bc_right;   // cam to body
    Eigen::Vector3d t_bc_right;     // cam to body

    Eigen::Vector3d T_extri;

    Eigen::Vector4d cam_intrinsics;

};


#endif //IMUSIM_PARAM_H
