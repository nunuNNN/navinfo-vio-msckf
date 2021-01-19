#ifndef DYNAMIC_INITIALIZER_H
#define DYNAMIC_INITIALIZER_H

#include <map>
#include <list>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "feature_manager.h"
#include "initial_alignment.h"
#include "initial_sfm.h"
#include "solve_5pts.h"

#include "imu_state.h"
#include "data_interface.h"

#include <iostream>

using namespace std;

namespace initializer {

class DynamicInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor.
    DynamicInitializer() = delete;
    DynamicInitializer(const double& acc_n_, const double& acc_w_, const double& gyr_n_, 
        const double& gyr_w_, const Eigen::Matrix3d& R_c2b, const Eigen::Vector3d& t_bc_b) : 
        bInit(false), state_time(0.0), curr_time(-1), first_imu(false), frame_count(0), 
        acc_n(acc_n_), acc_w(acc_w_), gyr_n(gyr_n_), gyr_w(gyr_w_), initial_timestamp(0.0),
        RIC(R_c2b), TIC(t_bc_b)
    {
        gyro_bias = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
        orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);

        for (int i = 0; i < WINDOW_SIZE + 1; i++)   
        {
            Rs[i].setIdentity();
            Ps[i].setZero();
            Vs[i].setZero();
            Bas[i].setZero();
            Bgs[i].setZero();
        }

        g = Eigen::Vector3d::Zero();

        // Initialize feature manager
        f_manager.clearState();
        f_manager.setRic(R_c2b);

        Times.resize(WINDOW_SIZE + 1);
    }

    // Destructor.
    ~DynamicInitializer(){};

    // Interface for trying to initialize.
    bool tryDynInit(const std::vector<StrImuData>& curr_from_last_imu,
                    FeatureMeasurePtr measure);

    // Assign the initial state if initialized successfully.
    void assignInitialState(msckf_vio::IMUState& imu_state);

    // If initialized.
    bool ifInitialized() {
        return bInit;
    }

private:
    // Flag indicating if initialized.
    bool bInit;

    // Relative rotation between camera and imu.
    Eigen::Matrix3d RIC;
    Eigen::Vector3d TIC;

    // Initialize results.
    double state_time;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;
    Eigen::Vector4d orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;

    // Save the last imu data that have been processed.
    Eigen::Vector3d last_acc;
    Eigen::Vector3d last_gyro;

    // Flag for declare the first imu data.
    bool first_imu;

    // Imu data for initialize every imu preintegration base.
    Vector3d acc_0, gyr_0;

    // Frame counter in sliding window.
    int frame_count;

    // Current imu time.
    double curr_time;

    // Imu noise param.
    double acc_n, acc_w;
    double gyr_n, gyr_w;

    // Temporal buff for imu preintegration between ordinary frames.
    std::shared_ptr<IntegrationBase> tmp_pre_integration;

    // Store the information of ordinary frames
    map<double, ImageFrame> all_image_frame;

    // Gravity under reference camera frame.
    Eigen::Vector3d g;

    // Feature manager.
    FeatureManager f_manager;

    // State of body frame under reference frame.
    Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
    // Bias of gyro and accelerometer of imu corresponding to every keyframe.
    Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];

    // Flags for marginalization.
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    MarginalizationFlag  marginalization_flag;

    // Timestamps of sliding window.
    vector<double> Times;

    // Initial timestamp
    double initial_timestamp;

    // For solving relative motion.
    MotionEstimator m_estimator;

private:

    // Process every imu frame before the img.
    void processIMU(const StrImuData& imu_msg);

    // Process img frame.
    void processImage(FeatureMeasurePtr measure);

    // Check if the condition is fit to conduct the vio initialization, and conduct it while suitable.
    bool initialStructure();

    // Try to recover relative pose.
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    // Align the visual sfm with imu preintegration.
    bool visualInitialAlign();

    // Slide the window.
    void slideWindow();
};

} // initializer

#endif //DYNAMIC_INITIALIZER_H
