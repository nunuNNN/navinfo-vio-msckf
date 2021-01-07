//
// Added by xiaochen at 19-8-16.
// Type and methods for initial alignment.
// The original file belong to VINS-MONO (https://github.com/HKUST-Aerial-Robotics/VINS-Mono).
//

#ifndef INITIAL_ALIGNMENT_H
#define INITIAL_ALIGNMENT_H

// #pragma once
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include "Initializer/ImuPreintegration.h"
#include "Initializer/feature_manager.h"
#include <map>


using namespace Eigen;
using namespace std;

namespace initializer {

class ImageFrame
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        ImageFrame(){};
        ImageFrame(FeatureMeasurePtr _points):is_key_frame{false}
        {
            t = _points->stamp;
            for (const auto& pt : _points->features)
            {
                double x = pt.u0;
                double y = pt.v0;
                double z = 1;
                double p_u = pt.u0;
                double p_v = pt.v0;
                double velocity_x = pt.u_vel;
                double velocity_y = pt.v_vel;
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                points[pt.id] = xyz_uv_velocity;
            }
        };
        map<int, Eigen::Matrix<double, 7, 1>> points;
        double t;
        Matrix3d R;
        Vector3d T;
        std::shared_ptr<IntegrationBase> pre_integration;
        bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x, const Vector3d& TIC);

}


#endif //INITIAL_ALIGNMENT_H
