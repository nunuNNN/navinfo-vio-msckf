#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "data_interface.h"

// start forw or down thread flag
extern bool flag_thread_start[2];


// 前后端的算法参数,参数的结构体在数据结构文件中定义
extern msckf_vio::Processor_config_t processor_config;
extern msckf_vio::Parameter_estimate_t params_estimste;

extern Parameter_extrinsic_t params_extrinsic;
extern Parameter_camera_t params_camera;

void InitParams(double f_forw, double cx_forw, double cy_forw, double baseline_forw);
