#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H


#include <map>
#include <set>
#include <queue>
#include <string>
#include <thread>
#include <condition_variable>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#ifdef USING_SPARSE_QR
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#endif

#include "imu_state.h"
#include "cam_state.h"
#include "feature.hpp"
#include "data_interface.h"

#include "initializer.h"

#ifdef USE_ROS_IMSHOW
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>
#include <nav_msgs/Path.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/PoseStamped.h>
#endif

namespace msckf_vio
{

class MsckfVio
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    MsckfVio(const Parameter_estimate_t &estimste_params,
            const Parameter_extrinsic_t &extrinsic_params);
    // Disable copy and assign constructor
    MsckfVio(const MsckfVio &) = delete;
    MsckfVio operator=(const MsckfVio &) = delete;

    // Destructor
    ~MsckfVio() {}

    void Process(double timestamp,
                const Feature_measure_t &measure,
                std::vector<StrImuData> &curr_from_last_imu,
                const Ground_truth_t &groundtruth,
                Translation_velocity_t &T_vel_out);

    bool resetCallback(const Parameter_estimate_t &estimste_params,
                            const Parameter_extrinsic_t &extrinsic_params);

    bool currFeatureInitCallback(std::vector<int> &init_ids,
                        std::vector<Eigen::Vector2d> &init_obs,
                        std::vector<Eigen::Vector3d> &init_pts);

private:
    bool InitStaticParams(const Parameter_estimate_t &estimste_params,
                        const Parameter_extrinsic_t &extrinsic_params);

    void InitPVQBAndCov();

    void ProcessBackEnd(const Feature_measure_t &measure,
                        std::vector<StrImuData> &curr_from_last_imu);

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(std::vector<StrImuData> &curr_from_last_imu);

    void processModel(const double &time,
                      const Eigen::Vector3d &m_gyro,
                      const Eigen::Vector3d &m_acc);

    void predictNewState(const double &dt,
                         const Eigen::Vector3d &gyro,
                         const Eigen::Vector3d &acc);

    // Measurement update
    void stateAugmentation(const double &time);

    void addFeatureObservations(const Feature_measure_t &msg);

    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const StateIDType &cam_state_id,
                             const FeatureIDType &feature_id,
                             Eigen::Matrix<double, 2, 6> &H_x,
                             Eigen::Matrix<double, 2, 3> &H_f,
                             Eigen::Vector2d &r);

    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType &feature_id,
                         const std::vector<StateIDType> &cam_state_ids,
                         Eigen::MatrixXd &H_x, Eigen::VectorXd &r);

    void measurementUpdate(const Eigen::MatrixXd &H,
                           const Eigen::VectorXd &r);

    bool gatingTest(const Eigen::MatrixXd &H,
                    const Eigen::VectorXd &r, const int &dof);

    void removeLostFeatures();

    void findRedundantCamStates(std::vector<StateIDType> &rm_cam_state_ids);

    void pruneCamStateBuffer();

    // Reset the system online if the uncertainty is too large.
    void onlineReset();

    void publish();

private:
    /*
     * @brief StateServer Store one IMU states and several
     *    camera states for constructing measurement
     *    model.
     */
    struct StateServer
    {
        IMUState imu_state;
        CamStateServer cam_states;

        // State covariance matrix
        Eigen::MatrixXd state_cov;
        Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };

    double curr_image_timestamp;
    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;

    // State vector
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;

    // Features used
    MapServer map_server;

    // Indicate if the gravity vector is set.
    bool b_init_finish;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;
    
    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // The initial covariance of orientation and position can be
    // set to 0. But for velocity, bias and extrinsic parameters,
    // there should be nontrivial uncertainty.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    double extrinsic_rotation_cov, extrinsic_translation_cov;

    // Tracking rate
    double tracking_rate;

    // 记录当前帧已经完成初始化的点的信息
    std::vector<int> curr_init_ids;
    std::vector<Eigen::Vector2d> curr_init_obs;
    std::vector<Eigen::Vector3d> curr_init_pts;

    Ground_truth_t curr_groundtruth;

    std::shared_ptr<initializer::Initializer> initializerPtr;

#ifdef USE_ROS_IMSHOW
    // Ros node handle
    // ros::NodeHandle nh;

    // Subscribers and publishers
    ros::Publisher odom_pub;
    ros::Publisher feature_pub;
    ros::Publisher pub_path;
    ros::Publisher mocap_odom_pub;

    nav_msgs::Path path;

    tf::TransformBroadcaster tf_pub;
#endif
};

} // namespace msckf_vio

#endif
