#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iterator>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>

using namespace std;
using namespace Eigen;

namespace msckf_vio
{

// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_body_from_imu = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table;

MsckfVio::MsckfVio(const Parameter_estimate_t &estimste_params,
            const Parameter_extrinsic_t &extrinsic_params)
    : is_first_img(true), b_init_finish(false)
{
    #ifdef USE_ROS_IMSHOW
        ros::NodeHandle nh("~");
        odom_pub = nh.advertise<nav_msgs::Odometry>("/firefly_sbx/vio/odom", 10);
        feature_pub = nh.advertise<sensor_msgs::PointCloud2>("/firefly_sbx/vio/feature_point_cloud", 10);
        pub_path = nh.advertise<nav_msgs::Path>("/firefly_sbx/vio/path", 10);
        mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("/firefly_sbx/vio/gt_odom", 1);
    #endif

    InitStaticParams(estimste_params, extrinsic_params);

    return;
}

bool MsckfVio::InitStaticParams(const Parameter_estimate_t &estimste_params,
                        const Parameter_extrinsic_t &extrinsic_params)
{
    // The position uncertainty threshold is used to determine
    position_std_threshold = estimste_params.position_std_threshold;

    // Threshold for determine keyframes
    translation_threshold = estimste_params.translation_threshold;
    rotation_threshold = estimste_params.rotation_threshold;
    tracking_rate_threshold = estimste_params.tracking_rate_threshold;

    // Feature optimization parameters
    Feature::optimization_config.translation_threshold = estimste_params.feature_trans_threshold;

    // Use variance instead of standard deviation.
    IMUState::gyro_noise = estimste_params.gyro_noise * estimste_params.gyro_noise;
    IMUState::acc_noise = estimste_params.acc_noise * estimste_params.acc_noise;
    IMUState::gyro_bias_noise = estimste_params.gyro_bias_noise * estimste_params.gyro_bias_noise;
    IMUState::acc_bias_noise = estimste_params.acc_bias_noise * estimste_params.acc_bias_noise;
    Feature::observation_noise = estimste_params.observation_noise * estimste_params.observation_noise;

    // Initialize state server
    state_server.continuous_noise_cov = Matrix<double, 12, 12>::Zero();
    state_server.continuous_noise_cov.block<3, 3>(0, 0) = Matrix3d::Identity() * IMUState::gyro_noise;
    state_server.continuous_noise_cov.block<3, 3>(3, 3) = Matrix3d::Identity() * IMUState::gyro_bias_noise;
    state_server.continuous_noise_cov.block<3, 3>(6, 6) = Matrix3d::Identity() * IMUState::acc_noise;
    state_server.continuous_noise_cov.block<3, 3>(9, 9) = Matrix3d::Identity() * IMUState::acc_bias_noise;

    // covariance
    gyro_bias_cov = estimste_params.gyro_bias_cov;
    acc_bias_cov = estimste_params.acc_bias_cov;
    velocity_cov = estimste_params.velocity_cov;
    extrinsic_rotation_cov = estimste_params.extrinsic_rotation_cov;
    extrinsic_translation_cov = estimste_params.extrinsic_translation_cov;

    // Maximum number of camera states
    max_cam_state_size = estimste_params.max_cam_state_size;

    // Initialize the chi squared test table with confidence
    // level 0.95.
    for (int i = 1; i < 100; ++i)
    {
        // boost::math::chi_squared chi_squared_dist(i);
        // chi_squared_test_table[i] = boost::math::quantile(chi_squared_dist, 0.05);
        chi_squared_test_table[i] = chi2inv(0.95, i);
    }

    // Transformation offsets between the frames involved.
    Eigen::Isometry3d T_imu_from_cam0 = extrinsic_params.T_cam0_from_imu.inverse();
    state_server.imu_state.R_cam0_from_imu = T_imu_from_cam0.linear().transpose();
    state_server.imu_state.t_imu_from_cam0 = T_imu_from_cam0.translation();

    IMUState::T_body_from_imu.matrix() = Matrix4d::Identity();

    // initialize initializer
    initializerPtr.reset(new initializer::Initializer(2e-3, 4, 
        sqrt(IMUState::acc_noise), sqrt(IMUState::acc_bias_noise),
        sqrt(IMUState::gyro_noise), sqrt(IMUState::gyro_bias_noise),
        state_server.imu_state.R_cam0_from_imu.transpose(),
        state_server.imu_state.t_imu_from_cam0));

    printf("===========================================\n");
    printf("position std threshold: %f\n", position_std_threshold);
    printf("Keyframe rotation threshold: %f\n", rotation_threshold);
    printf("Keyframe translation threshold: %f\n", translation_threshold);
    printf("Keyframe tracking rate threshold: %f\n", tracking_rate_threshold);
    printf("Feature::optimization_config.translation_threshold: %f\n",
           Feature::optimization_config.translation_threshold);
    printf("gyro noise: %.10f\n", IMUState::gyro_noise);
    printf("gyro bias noise: %.10f\n", IMUState::gyro_bias_noise);
    printf("acc noise: %.10f\n", IMUState::acc_noise);
    printf("acc bias noise: %.10f\n", IMUState::acc_bias_noise);
    printf("observation noise: %.10f\n", Feature::observation_noise);
    printf("initial velocity: %f, %f, %f\n",
           state_server.imu_state.velocity(0),
           state_server.imu_state.velocity(1),
           state_server.imu_state.velocity(2));
    printf("initial gyro bias cov: %f\n", gyro_bias_cov);
    printf("initial acc bias cov: %f\n", acc_bias_cov);
    printf("initial velocity cov: %f\n", velocity_cov);
    printf("initial extrinsic rotation cov: %f\n", extrinsic_rotation_cov);
    printf("initial extrinsic translation cov: %f\n", extrinsic_translation_cov);

    cout << "R_cam0_from_imu: " << state_server.imu_state.R_cam0_from_imu << endl;
    cout << "t_imu_from_cam0: " << state_server.imu_state.t_imu_from_cam0.transpose() << endl;

    printf("max camera state #: %d\n", max_cam_state_size);
    printf("===========================================\n");
    return true;
}

void MsckfVio::Process(double timestamp,
                       const Feature_measure_t &measure,
                       std::vector<StrImuData> &curr_from_last_imu,
                       const Ground_truth_t &groundtruth,
                       Translation_velocity_t &T_vel_out)
{
    /*******************后端逻辑流程,需要对数据做一些判断在运行算法******************/
    curr_image_timestamp = measure.stamp;

    /************************* 用于真值初始化 ****************************/
    //保存导航传过来的位姿、速度.只用来做真值初始化.
    curr_groundtruth = groundtruth;

    /*******************************初始化过程*********************************/
    if (!b_init_finish)
    {
        /**
         * 使用初始化有两部分。1.系统初始化的时候；2.系统发散之后系统初始化。
         * 现在存在问题。     1.在系统初始化的时候，系统还有一段时间没有输出，系统不能等待；
         *                  2.系统重置的时候，系统需要将初始初始化的位置为参考位置，不能与之前位置连续。 
        */

        // 初始化，静态初始化及动态初始化
        FeatureMeasurePtr features = make_shared<Feature_measure_t>(measure);
        bool succeed = initializerPtr->tryIncInit(curr_from_last_imu, features, state_server.imu_state);

        if(succeed)
        {
            InitPVQBAndCov();

            b_init_finish = true;

            T_vel_out.T_w_b.linear() = quaternionToRotation(state_server.imu_state.orientation).transpose();
            T_vel_out.T_w_b.translation()  = state_server.imu_state.position;
            T_vel_out.v_in_world = state_server.imu_state.velocity;
        }

        return;
    }

    /************************初始化成功,进行后端计算,算法主体***********************/
    ProcessBackEnd(measure, curr_from_last_imu);

    /*************************后端计算成功后输出计算的位姿************************/
    T_vel_out.T_w_b.linear() = quaternionToRotation(state_server.imu_state.orientation).transpose();
    T_vel_out.T_w_b.translation()  = state_server.imu_state.position;
    T_vel_out.v_in_world = state_server.imu_state.velocity;

}

void MsckfVio::InitPVQBAndCov()
{
    // cout << "1 InitializeState curr_image_timestamp: " << curr_image_timestamp << endl;

    if(false)
    {
        // Start the system if the first image is received.
        // The frame where the first image is received will be the origin.
        state_server.imu_state.time = curr_image_timestamp;
        //q_b_from_w
        state_server.imu_state.orientation = rotationToQuaternion(curr_groundtruth.T_vel.T_w_b.linear().transpose());
        //t_w_from_b
        state_server.imu_state.position = curr_groundtruth.T_vel.T_w_b.translation();
        // v_in_world
        state_server.imu_state.velocity = curr_groundtruth.T_vel.v_in_world;
        state_server.imu_state.gyro_bias = curr_groundtruth.bias_gyr;
        state_server.imu_state.acc_bias = curr_groundtruth.bias_acc;
    }

    // set coviance
    state_server.state_cov = MatrixXd::Zero(21, 21);

    for (int i = 3; i < 6; ++i)
    {
        state_server.state_cov(i, i) = gyro_bias_cov;
    }
    for (int i = 6; i < 9; ++i)
    {
        state_server.state_cov(i, i) = velocity_cov;
    }
    for (int i = 9; i < 12; ++i)
    {
        state_server.state_cov(i, i) = acc_bias_cov;
    }
    for (int i = 15; i < 18; ++i)
    {
        state_server.state_cov(i, i) = extrinsic_rotation_cov;
    }
    for (int i = 18; i < 21; ++i)
    {
        state_server.state_cov(i, i) = extrinsic_translation_cov;
    }

    // cout << "2 InitializeState PVQ: " << endl;
    // cout << "t_w_b: " << state_server.imu_state.position.transpose() << endl;
    // cout << "v_w_b: " << state_server.imu_state.velocity.transpose() << endl;
    // cout << "R_w_b: " << quaternionToRotation(state_server.imu_state.orientation).transpose()
    //     << endl;
}

void MsckfVio::ProcessBackEnd(const Feature_measure_t &measure,
                            std::vector<StrImuData> &curr_from_last_imu)
{
    // Propogate the IMU state.
    // that are received before the image msg.
    batchImuProcessing(curr_from_last_imu);

    // Augment the state vector.
    stateAugmentation(curr_image_timestamp);

    // Add new observations for existing features or new
    // features in the map server.
    int num_features = measure.features.size();
    cout << "num_features is : " << num_features << endl;
    addFeatureObservations(measure);

    // Perform measurement update if necessary.
    removeLostFeatures();

    pruneCamStateBuffer();

    // Publish the odometry.
    publish();

    // Reset the system if necessary.
    onlineReset();
}


void MsckfVio::batchImuProcessing(std::vector<StrImuData> &curr_from_last_imu)
{
    // cout << "batchImuProcessing curr_from_last_imu.size: " << curr_from_last_imu.size()
    //      << " state_server.imu_state.time: " << state_server.imu_state.time << endl;

    for(int i=0; i<curr_from_last_imu.size(); i++)
    {
        auto imu_msg = curr_from_last_imu[i];
        processModel(imu_msg.timestamp, imu_msg.gyr, imu_msg.acc);
    }

    // Set the state ID for the new IMU state.
    state_server.imu_state.id = IMUState::next_id++;

    return;
}

void MsckfVio::processModel(const double &time,
                            const Vector3d &m_gyro,
                            const Vector3d &m_acc)
{
    // Remove the bias from the measured gyro and acceleration
    IMUState &imu_state = state_server.imu_state;
    Vector3d gyro = m_gyro - imu_state.gyro_bias;
    Vector3d acc = m_acc - imu_state.acc_bias;
    double dtime = time - imu_state.time;

    // Compute discrete transition and noise covariance matrix
    Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
    Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();

    F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    F.block<3, 3>(0, 3) = -Matrix3d::Identity();
    F.block<3, 3>(6, 0) = -quaternionToRotation(imu_state.orientation).transpose() * skewSymmetric(acc);
    F.block<3, 3>(6, 9) = -quaternionToRotation(imu_state.orientation).transpose();
    F.block<3, 3>(12, 6) = Matrix3d::Identity();

    G.block<3, 3>(0, 0) = -Matrix3d::Identity();
    G.block<3, 3>(3, 3) = Matrix3d::Identity();
    G.block<3, 3>(6, 6) = -quaternionToRotation(imu_state.orientation).transpose();
    G.block<3, 3>(9, 9) = Matrix3d::Identity();

    // Approximate matrix exponential to the 3rd order,
    // which can be considered to be accurate enough assuming
    // dtime is within 0.01s.
    Matrix<double, 21, 21> Fdt = F * dtime;
    Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
    Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
    Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() + Fdt + 0.5 * Fdt_square + (1.0 / 6.0) * Fdt_cube;

    // Propogate the state using 4th order Runge-Kutta
    predictNewState(dtime, gyro, acc);

    // Modify the transition matrix
    Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
    Phi.block<3, 3>(0, 0) = quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

    Vector3d u = R_kk_1 * IMUState::gravity;
    RowVector3d s = (u.transpose() * u).inverse() * u.transpose();

    Matrix3d A1 = Phi.block<3, 3>(6, 0);
    Vector3d w1 = skewSymmetric(imu_state.velocity_null - imu_state.velocity) * IMUState::gravity;
    Phi.block<3, 3>(6, 0) = A1 - (A1 * u - w1) * s;

    Matrix3d A2 = Phi.block<3, 3>(12, 0);
    Vector3d w2 = skewSymmetric(dtime * imu_state.velocity_null + imu_state.position_null - imu_state.position) * IMUState::gravity;
    Phi.block<3, 3>(12, 0) = A2 - (A2 * u - w2) * s;

    // Propogate the state covariance matrix.
    Matrix<double, 21, 21> Q = Phi * G * state_server.continuous_noise_cov * G.transpose() * Phi.transpose() * dtime;
    state_server.state_cov.block<21, 21>(0, 0) = Phi * state_server.state_cov.block<21, 21>(0, 0) * Phi.transpose() + Q;

    if (state_server.cam_states.size() > 0)
    {
        state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21) =
            Phi * state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21);
        state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) =
            state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) * Phi.transpose();
    }

    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    // Update the state correspondes to null space.
    imu_state.orientation_null = imu_state.orientation;
    imu_state.position_null = imu_state.position;
    imu_state.velocity_null = imu_state.velocity;

    // Update the state info
    state_server.imu_state.time = time;
    return;
}

void MsckfVio::predictNewState(const double &dt,
                               const Vector3d &gyro,
                               const Vector3d &acc)
{

    // TODO: Will performing the forward integration using
    //    the inverse of the quaternion give better accuracy?
    double gyro_norm = gyro.norm();
    Matrix4d Omega = Matrix4d::Zero();
    Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    Omega.block<3, 1>(0, 3) = gyro;
    Omega.block<1, 3>(3, 0) = -gyro;

    Vector4d &q = state_server.imu_state.orientation;
    Vector3d &v = state_server.imu_state.velocity;
    Vector3d &p = state_server.imu_state.position;

    // Some pre-calculation
    Vector4d dq_dt, dq_dt2;
    if (gyro_norm > 1e-5)
    {
        dq_dt = (cos(gyro_norm * dt * 0.5) * Matrix4d::Identity() +
                 1 / gyro_norm * sin(gyro_norm * dt * 0.5) * Omega) * q;
        dq_dt2 = (cos(gyro_norm * dt * 0.25) * Matrix4d::Identity() +
                  1 / gyro_norm * sin(gyro_norm * dt * 0.25) * Omega) * q;
    }
    else
    {
        dq_dt = (Matrix4d::Identity() + 0.5 * dt * Omega) *
                cos(gyro_norm * dt * 0.5) * q;
        dq_dt2 = (Matrix4d::Identity() + 0.25 * dt * Omega) *
                 cos(gyro_norm * dt * 0.25) * q;
    }
    Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
    Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

    // k1 = f(tn, yn)
    Vector3d k1_v_dot = quaternionToRotation(q).transpose() * acc + IMUState::gravity;
    // cout << "k1_v_dot: " << k1_v_dot.transpose() << endl;
    Vector3d k1_p_dot = v;

    // k2 = f(tn+dt/2, yn+k1*dt/2)
    Vector3d k1_v = v + k1_v_dot * dt / 2;
    Vector3d k2_v_dot = dR_dt2_transpose * acc + IMUState::gravity;
    Vector3d k2_p_dot = k1_v;

    // k3 = f(tn+dt/2, yn+k2*dt/2)
    Vector3d k2_v = v + k2_v_dot * dt / 2;
    Vector3d k3_v_dot = dR_dt2_transpose * acc + IMUState::gravity;
    Vector3d k3_p_dot = k2_v;

    // k4 = f(tn+dt, yn+k3*dt)
    Vector3d k3_v = v + k3_v_dot * dt;
    Vector3d k4_v_dot = dR_dt_transpose * acc + IMUState::gravity;
    Vector3d k4_p_dot = k3_v;

    // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
    q = dq_dt;
    quaternionNormalize(q);
    v = v + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
    p = p + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);

    return;
}

void MsckfVio::stateAugmentation(const double &time)
{
    const Matrix3d &R_c_from_i = state_server.imu_state.R_cam0_from_imu;
    const Vector3d &t_i_from_c = state_server.imu_state.t_imu_from_cam0;

    // Add a new camera state to the state server.
    Matrix3d R_i_from_w = quaternionToRotation(state_server.imu_state.orientation);
    Matrix3d R_c_from_w = R_c_from_i * R_i_from_w;
    Vector3d t_w_from_c = state_server.imu_state.position + R_i_from_w.transpose() * t_i_from_c;

    state_server.cam_states[state_server.imu_state.id] = CAMState(state_server.imu_state.id);
    CAMState &cam_state = state_server.cam_states[state_server.imu_state.id];

    cam_state.time = time;
    cam_state.orientation = rotationToQuaternion(R_c_from_w);
    cam_state.position = t_w_from_c;

    cam_state.orientation_null = cam_state.orientation;
    cam_state.position_null = cam_state.position;

    // Update the covariance matrix of the state.
    // To simplify computation, the matrix J below is the nontrivial block
    // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
    // -aided Inertial Navigation".
    Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
    J.block<3, 3>(0, 0) = R_c_from_i;
    J.block<3, 3>(0, 15) = Matrix3d::Identity();
    J.block<3, 3>(3, 0) = skewSymmetric(R_i_from_w.transpose() * t_i_from_c);
    //J.block<3, 3>(3, 0) = -R_i_from_w.transpose()*skewSymmetric(t_i_from_c);
    J.block<3, 3>(3, 12) = Matrix3d::Identity();
    J.block<3, 3>(3, 18) = Matrix3d::Identity();

    // Resize the state covariance matrix.
    size_t old_rows = state_server.state_cov.rows();
    size_t old_cols = state_server.state_cov.cols();
    state_server.state_cov.conservativeResize(old_rows + 6, old_cols + 6);

    // Rename some matrix blocks for convenience.
    const Matrix<double, 21, 21> &P11 = state_server.state_cov.block<21, 21>(0, 0);
    const MatrixXd &P12 = state_server.state_cov.block(0, 21, 21, old_cols - 21);

    // Fill in the augmented state covariance.
    state_server.state_cov.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;
    state_server.state_cov.block(0, old_cols, old_rows, 6) = state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
    state_server.state_cov.block<6, 6>(old_rows, old_cols) = J * P11 * J.transpose();

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;

    state_server.state_cov = state_cov_fixed;

    return;
}

void MsckfVio::addFeatureObservations(const Feature_measure_t &msg)
{
    StateIDType state_id = state_server.imu_state.id;
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;
    // Add new observations for existing features or new
    // features in the map server.
    for (const auto &feature : msg.features)
    {
        if (map_server.find(feature.id) == map_server.end())
        {
            // This is a new feature.
            map_server[feature.id] = Feature(feature.id);
            map_server[feature.id].observations[state_id] = Vector2d(feature.u0, feature.v0);
        }
        else
        {
            // This is an old feature.
            map_server[feature.id].observations[state_id] = Vector2d(feature.u0, feature.v0);
            ++tracked_feature_num;
        }
    }

    tracking_rate = static_cast<double>(tracked_feature_num) / static_cast<double>(curr_feature_num);

    return;
}

void MsckfVio::measurementJacobian(const StateIDType &cam_state_id, const FeatureIDType &feature_id,
    Matrix<double, 2, 6> &H_x, Matrix<double, 2, 3> &H_f, Vector2d &r)
{
    // Prepare all the required data.
    const CAMState &cam_state = state_server.cam_states[cam_state_id];
    const Feature &feature = map_server[feature_id];

    // Cam0 pose.
    Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
    const Vector3d &t_c0_w = cam_state.position;

    // 3d feature position in the world frame.
    // And its observation with the stereo cameras.
    const Vector3d &p_w = feature.position;
    const Vector2d &z = feature.observations.find(cam_state_id)->second;

    // Convert the feature position from the world frame to
    // the cam0 frame.
    Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);

    // Compute the Jacobians.
    Matrix<double, 2, 3> dz_dpc0 = Matrix<double, 2, 3>::Zero();
    dz_dpc0(0, 0) = 1 / p_c0(2);
    dz_dpc0(1, 1) = 1 / p_c0(2);
    dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
    dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

    Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
    dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
    dpc0_dxc.rightCols(3) = -R_w_c0;

    Matrix3d dpc0_dpg = R_w_c0;

    H_x = dz_dpc0 * dpc0_dxc;
    H_f = dz_dpc0 * dpc0_dpg;

    // Modifty the measurement Jacobian to ensure
    // observability constrain.
    Matrix<double, 2, 6> A = H_x;
    Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
    u.block<3, 1>(0, 0) = quaternionToRotation(cam_state.orientation_null) * IMUState::gravity;
    u.block<3, 1>(3, 0) = skewSymmetric(p_w - cam_state.position_null) * IMUState::gravity;
    H_x = A - A * u * (u.transpose() * u).inverse() * u.transpose();
    H_f = -H_x.block<2, 3>(0, 3);

    // Compute the residual.
    r = z - Vector2d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2));

    return;
}

void MsckfVio::featureJacobian(const FeatureIDType &feature_id,
    const std::vector<StateIDType> &cam_state_ids, MatrixXd &H_x, VectorXd &r)
{

    const auto &feature = map_server[feature_id];

    // Check how many camera states in the provided camera
    // id camera has actually seen this feature.
    vector<StateIDType> valid_cam_state_ids(0);
    for (const auto &cam_id : cam_state_ids)
    {
        if (feature.observations.find(cam_id) == feature.observations.end())
            continue;

        valid_cam_state_ids.push_back(cam_id);
    }

    int jacobian_row_size = 0;
    jacobian_row_size = 2 * valid_cam_state_ids.size();

    MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size, 21 + state_server.cam_states.size() * 6);
    MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
    VectorXd r_j = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for (const auto &cam_id : valid_cam_state_ids)
    {
        Matrix<double, 2, 6> H_xi = Matrix<double, 2, 6>::Zero();
        Matrix<double, 2, 3> H_fi = Matrix<double, 2, 3>::Zero();
        Vector2d r_i = Vector2d::Zero();
        measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

        auto cam_state_iter = state_server.cam_states.find(cam_id);
        int cam_state_cntr = std::distance(state_server.cam_states.begin(), cam_state_iter);

        // Stack the Jacobians.
        H_xj.block<2, 6>(stack_cntr, 21 + 6 * cam_state_cntr) = H_xi;
        H_fj.block<2, 3>(stack_cntr, 0) = H_fi;
        r_j.segment<2>(stack_cntr) = r_i;
        stack_cntr += 2;
    }

    // Project the residual and Jacobians onto the nullspace
    // of H_fj.
    JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
    MatrixXd A = svd_helper.matrixU().rightCols(jacobian_row_size - 3);

    H_x = A.transpose() * H_xj;
    r = A.transpose() * r_j;

    return;
}

void MsckfVio::measurementUpdate(const MatrixXd &H, const VectorXd &r)
{

    if (H.rows() == 0 || r.rows() == 0) return;

    // Decompose the final Jacobian matrix to reduce computational
    // complexity as in Equation (28), (29).
    MatrixXd H_thin;
    VectorXd r_thin;

    if (H.rows() > H.cols())
    {
#ifdef USING_SPARSE_QR
        /// Convert H to a sparse matrix.
        SparseMatrix<double> H_sparse = H.sparseView();

        // Perform QR decompostion on H_sparse.
        SPQR<SparseMatrix<double>> spqr_helper;
        spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
        spqr_helper.compute(H_sparse);

        MatrixXd H_temp;
        VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        H_thin = H_temp.topRows(21 + state_server.cam_states.size() * 6);
        r_thin = r_temp.head(21 + state_server.cam_states.size() * 6);
#else
        Eigen::HouseholderQR<Eigen::MatrixXd> qr_helper(H);
        Eigen::MatrixXd Q = qr_helper.householderQ();
        Eigen::MatrixXd Q1 = Q.leftCols(21 + state_server.cam_states.size() * 6);

        H_thin = Q1.transpose() * H;
        r_thin = Q1.transpose() * r;
#endif
    }
    else
    {
        H_thin = H;
        r_thin = r;
    }

    // Compute the Kalman gain.
    const MatrixXd &P = state_server.state_cov;
    MatrixXd S = H_thin * P * H_thin.transpose() + Feature::observation_noise * MatrixXd::Identity(H_thin.rows(), H_thin.rows());
    //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
    MatrixXd K_transpose = S.ldlt().solve(H_thin * P);
    MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // Update the IMU state.
    const VectorXd &delta_x_imu = delta_x.head<21>();

    if ( //delta_x_imu.segment<3>(0).norm() > 0.15 ||
        //delta_x_imu.segment<3>(3).norm() > 0.15 ||
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        //delta_x_imu.segment<3>(9).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0)
    {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        printf("Update change is too large.");
        return;
    }

    const Vector4d dq_imu = smallAngleQuaternion(delta_x_imu.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.position += delta_x_imu.segment<3>(12);

    const Vector4d dq_extrinsic = smallAngleQuaternion(delta_x_imu.segment<3>(15));
    state_server.imu_state.R_cam0_from_imu = quaternionToRotation(dq_extrinsic) * state_server.imu_state.R_cam0_from_imu;
    state_server.imu_state.t_imu_from_cam0 += delta_x_imu.segment<3>(18);

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size(); ++i, ++cam_state_iter)
    {
        const VectorXd &delta_x_cam = delta_x.segment<6>(21 + i * 6);
        const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        cam_state_iter->second.orientation = quaternionMultiplication(dq_cam, cam_state_iter->second.orientation);
        cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
    //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //  K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH * state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    return;
}

bool MsckfVio::gatingTest(const MatrixXd &H, const VectorXd &r, const int &dof)
{

    MatrixXd P1 = H * state_server.state_cov * H.transpose();
    MatrixXd P2 = Feature::observation_noise *
                  MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

    //cout << dof << " " << gamma << " " <<
    //  chi_squared_test_table[dof] << " ";

    if (gamma < chi_squared_test_table[dof])
    {
        //cout << "passed" << endl;
        return true;
    }
    else
    {
        //cout << "failed" << endl;
        return false;
    }
}

void MsckfVio::removeLostFeatures()
{
    // Remove the features that lost track.
    // BTW, find the size the final Jacobian matrix and residual vector.
    int jacobian_row_size = 0;
    vector<FeatureIDType> invalid_feature_ids(0);
    vector<FeatureIDType> processed_feature_ids(0);

    for (auto iter = map_server.begin(); iter != map_server.end(); ++iter)
    {
        // Rename the feature to be checked.
        auto &feature = iter->second;

        // Pass the features that are still being tracked.
        if (feature.observations.find(state_server.imu_state.id) != feature.observations.end())
        {
            continue;
        }

        if (feature.observations.size() < 3)
        {
            invalid_feature_ids.push_back(feature.id);
            continue;
        }

        // Check if the feature can be initialized if it
        // has not been.

        if (!feature.is_initialized)
        {
            if (!feature.checkMotion(state_server.cam_states))
            {
                invalid_feature_ids.push_back(feature.id);
                continue;
            }
            else
            {
                if (!feature.initializePosition(state_server.cam_states))
                {
                    invalid_feature_ids.push_back(feature.id);
                    continue;
                }
            }
        }

        jacobian_row_size += 2 * feature.observations.size() - 3;
        processed_feature_ids.push_back(feature.id);
    }

    // cout << "invalid/processed map_server.size " << map_server.size()
    //      << " feature #: " << invalid_feature_ids.size()
    //      << "/" << processed_feature_ids.size() << endl;
    // cout << "jacobian row #: " << jacobian_row_size << endl;

    // Remove the features that do not have enough measurements.
    for (const auto &feature_id : invalid_feature_ids)
    {
        map_server.erase(feature_id);
    }

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.size() == 0)
    {
        return;
    }

    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 21 + 6 * state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // Process the features which lose track.
    for (const auto &feature_id : processed_feature_ids)
    {
        auto &feature = map_server[feature_id];

        vector<StateIDType> cam_state_ids(0);
        for (const auto &measurement : feature.observations)
        {
            cam_state_ids.push_back(measurement.first);
        }

        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, cam_state_ids.size() - 1))
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        // Put an upper bound on the row size of measurement Jacobian,
        // which helps guarantee the executation time.
        if (stack_cntr > 1500)
        {
            break;
        }
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    measurementUpdate(H_x, r);

    // Remove all processed features from the map.
    for (const auto &feature_id : processed_feature_ids)
    {
        map_server.erase(feature_id);
    }

    return;
}

void MsckfVio::findRedundantCamStates(vector<StateIDType> &rm_cam_state_ids)
{
    // Move the iterator to the key position.
    auto key_cam_state_iter = state_server.cam_states.end();
    for (int i = 0; i < 2; ++i)
        --key_cam_state_iter;
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;
    auto first_cam_state_iter = state_server.cam_states.begin();

    // Pose of the key camera state.
    const Vector3d key_position = key_cam_state_iter->second.position;
    const Matrix3d key_rotation = quaternionToRotation(key_cam_state_iter->second.orientation);

    // Mark the camera states to be removed based on the
    // motion between states.
    for (int i = 0; i < 2; ++i)
    {
        const Vector3d position = cam_state_iter->second.position;
        const Matrix3d rotation = quaternionToRotation(cam_state_iter->second.orientation);

        double distance = (position - key_position).norm();
        double angle = AngleAxisd(rotation * key_rotation.transpose()).angle();

        if (angle < rotation_threshold && distance < translation_threshold && tracking_rate > tracking_rate_threshold)
        {
            rm_cam_state_ids.push_back(cam_state_iter->first);
            ++cam_state_iter;
        }
        else
        {
            rm_cam_state_ids.push_back(first_cam_state_iter->first);
            ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

    return;
}

void MsckfVio::pruneCamStateBuffer()
{
    // cout << "1 pruneCamStateBuffer state_server.cam_states.size(): "
    //      << state_server.cam_states.size()
    //      << " max_cam_state_size: " << max_cam_state_size << endl;

    if (state_server.cam_states.size() < max_cam_state_size)
    {
        return;
    }

    // Find two camera states to be removed.
    vector<StateIDType> rm_cam_state_ids(0);
    findRedundantCamStates(rm_cam_state_ids);

    // Find the size of the Jacobian matrix.
    int jacobian_row_size = 0;
    for (auto &item : map_server)
    {
        auto &feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto &cam_id : rm_cam_state_ids)
        {
            if (feature.observations.find(cam_id) != feature.observations.end())
                involved_cam_state_ids.push_back(cam_id);
        }

        if (involved_cam_state_ids.size() == 0)
        {
            continue;
        }
        if (involved_cam_state_ids.size() == 1)
        {
            feature.observations.erase(involved_cam_state_ids[0]);
            continue;
        }

        if (!feature.is_initialized)
        {
            // Check if the feature can be initialize.
            if (!feature.checkMotion(state_server.cam_states))
            {
                // If the feature cannot be initialized, just remove
                // the observations associated with the camera states
                // to be removed.
                for (const auto &cam_id : involved_cam_state_ids)
                {
                    feature.observations.erase(cam_id);
                }
                continue;
            }
            else
            {
                if (!feature.initializePosition(state_server.cam_states))
                {
                    for (const auto &cam_id : involved_cam_state_ids)
                    {
                        feature.observations.erase(cam_id);
                    }
                    continue;
                }
            }
        }

        jacobian_row_size += 2 * involved_cam_state_ids.size() - 3;
    }

    //cout << "jacobian row #: " << jacobian_row_size << endl;

    // Compute the Jacobian and residual.
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 21 + 6 * state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for (auto &item : map_server)
    {
        auto &feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto &cam_id : rm_cam_state_ids)
        {
            if (feature.observations.find(cam_id) != feature.observations.end())
            {
                involved_cam_state_ids.push_back(cam_id);
            }
        }

        if (involved_cam_state_ids.size() == 0)
        {
            continue;
        }

        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, involved_cam_state_ids.size()))
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        for (const auto &cam_id : involved_cam_state_ids)
        {
            feature.observations.erase(cam_id);
        }
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform measurement update.
    measurementUpdate(H_x, r);

    for (const auto &cam_id : rm_cam_state_ids)
    {
        int cam_sequence = std::distance(state_server.cam_states.begin(), state_server.cam_states.find(cam_id));
        int cam_state_start = 21 + 6 * cam_sequence;
        int cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state
        // covariance matrix.
        if (cam_state_end < state_server.state_cov.rows())
        {
            state_server.state_cov.block(cam_state_start, 0, state_server.state_cov.rows() - cam_state_end, state_server.state_cov.cols()) =
                state_server.state_cov.block(cam_state_end, 0, state_server.state_cov.rows() - cam_state_end, state_server.state_cov.cols());

            state_server.state_cov.block(0, cam_state_start, state_server.state_cov.rows(), state_server.state_cov.cols() - cam_state_end) =
                state_server.state_cov.block(0, cam_state_end, state_server.state_cov.rows(), state_server.state_cov.cols() - cam_state_end);

            state_server.state_cov.conservativeResize(state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        }
        else
        {
            state_server.state_cov.conservativeResize(state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        }

        // Remove this camera state in the state vector.
        state_server.cam_states.erase(cam_id);
    }

    return;
}

void MsckfVio::onlineReset()
{
    // Never perform online reset if position std threshold
    // is non-positive.
    if (position_std_threshold <= 0)
    {
        return;
    }
    static long long int online_reset_counter = 0;

    // Check the uncertainty of positions to determine if
    // the system can be reset.
    double position_x_std = std::sqrt(state_server.state_cov(12, 12));
    double position_y_std = std::sqrt(state_server.state_cov(13, 13));
    double position_z_std = std::sqrt(state_server.state_cov(14, 14));

    if (position_x_std < position_std_threshold &&
        position_y_std < position_std_threshold &&
        position_z_std < position_std_threshold)
    {
        return;
    }

    printf("Start %lld online reset procedure...", ++online_reset_counter);
    printf("Stardard deviation in xyz: %f, %f, %f",
           position_x_std, position_y_std, position_z_std);

    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Clear all exsiting features in the map.
    map_server.clear();

    b_init_finish = false;

    printf("%lld online reset complete...", online_reset_counter);
    return;
}

// 后端切换函数,需要清除之前的状态,外参数重置
bool MsckfVio::resetCallback(const Parameter_estimate_t &estimste_params,
                            const Parameter_extrinsic_t &extrinsic_params) 
{
    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Clear all exsiting features in the map.
    map_server.clear();

    InitStaticParams(estimste_params, extrinsic_params);

    b_init_finish = false;

    return true;
}

void MsckfVio::publish()
{
    // Convert the IMU frame to the body frame.
    const IMUState &imu_state = state_server.imu_state;
    Eigen::Isometry3d T_w_from_i = Eigen::Isometry3d::Identity();
    T_w_from_i.linear() = quaternionToRotation(imu_state.orientation).transpose();
    T_w_from_i.translation() = imu_state.position;

    // cout << "imu_state.position: " << imu_state.position.transpose() << endl;
    // cout << "ba: " << imu_state.acc_bias.transpose()
    //      << " bg: " << imu_state.gyro_bias.transpose() << endl;

    Eigen::Isometry3d T_w_from_b = IMUState::T_body_from_imu * T_w_from_i *
                                   IMUState::T_body_from_imu.inverse();
    Eigen::Vector3d body_velocity =
        IMUState::T_body_from_imu.linear() * imu_state.velocity;

#ifdef USE_ROS_IMSHOW
    string fixed_frame_id = "world";
    string ned_frame_id = "ned";
    string child_frame_id = "body";
    ros::Time time(curr_image_timestamp);

    //设置ned系与rviz显示的world系之间的关系，若euroc，则两者相同，若ap03，则z朝下
    Eigen::Isometry3d T_fixed_from_ned = Eigen::Isometry3d::Identity();
    Matrix3d R_fixed_from_ned = Matrix3d::Identity();
#ifndef USE_EUROC
    R_fixed_from_ned << 1, 0, 0,
        0, -1, 0,
        0, 0, -1;
#endif
    T_fixed_from_ned.linear() = R_fixed_from_ned;
    T_fixed_from_ned.translation() = Vector3d::Zero();
    tf::Transform T_fixed_from_ned_tf;
    tf::transformEigenToTF(T_fixed_from_ned, T_fixed_from_ned_tf);
    tf_pub.sendTransform(tf::StampedTransform(T_fixed_from_ned_tf, time, fixed_frame_id, ned_frame_id));

    tf::Transform T_w_from_b_tf;
    tf::transformEigenToTF(T_w_from_b, T_w_from_b_tf);
    tf_pub.sendTransform(tf::StampedTransform(T_w_from_b_tf, time, ned_frame_id, child_frame_id));

    // Publish the odometry
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = time;
    odom_msg.header.frame_id = ned_frame_id;
    odom_msg.child_frame_id = child_frame_id;

    tf::poseEigenToMsg(T_w_from_b, odom_msg.pose.pose);
    tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

    // Convert the covariance.
    Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
    Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
    Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
    Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
    Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
    P_imu_pose << P_pp, P_po, P_op, P_oo;

    Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
    H_pose.block<3, 3>(0, 0) = IMUState::T_body_from_imu.linear();
    H_pose.block<3, 3>(3, 3) = IMUState::T_body_from_imu.linear();
    Matrix<double, 6, 6> P_body_pose = H_pose * P_imu_pose * H_pose.transpose();

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            odom_msg.pose.covariance[6 * i + j] = P_body_pose(i, j);
        }
    }
    // Construct the covariance for the velocity.
    Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
    Matrix3d H_vel = IMUState::T_body_from_imu.linear();
    Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            odom_msg.twist.covariance[i * 6 + j] = P_body_vel(i, j);
        }
    }
    odom_pub.publish(odom_msg);

    // Publish the 3D positions of the features that
    // has been initialized.
    pcl::PointCloud<pcl::PointXYZ>::Ptr feature_msg_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    feature_msg_ptr->header.frame_id = ned_frame_id;
    feature_msg_ptr->height = 1;
    for (const auto &item : map_server)
    {
        const auto &feature = item.second;
        if (feature.is_initialized)
        {
            Vector3d feature_position = IMUState::T_body_from_imu.linear() * feature.position;
            feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
        }
    }
    feature_msg_ptr->width = feature_msg_ptr->points.size();
    // cout << "feature_msg_ptr->points.size(): " << feature_msg_ptr->points.size() << endl;
    feature_pub.publish(feature_msg_ptr);

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = odom_msg.header.stamp;
    pose_msg.header.frame_id = odom_msg.header.frame_id;
    pose_msg.pose = odom_msg.pose.pose;

    path.header.stamp = odom_msg.header.stamp;
    path.header.frame_id = ned_frame_id;
    path.poses.push_back(pose_msg);
    // cout << "path.poses size is " << path.poses.size() << endl;
    pub_path.publish(path);

    // Transform the ground truth.
    Eigen::Isometry3d T_w_from_b_gt;
    T_w_from_b_gt.linear() = curr_groundtruth.T_vel.T_w_b.linear();
    T_w_from_b_gt.translation() = curr_groundtruth.T_vel.T_w_b.translation() + Vector3d(0, 0, 1);

    // Ground truth tf.
    string ground_truth_frame_id = "gt";
    bool publish_tf = true;
    if (publish_tf)
    {
        tf::Transform T_w_from_b_gt_tf;
        tf::transformEigenToTF(T_w_from_b_gt, T_w_from_b_gt_tf);
        tf_pub.sendTransform(tf::StampedTransform(T_w_from_b_gt_tf, time, ned_frame_id, ground_truth_frame_id));
    }

    // Ground truth odometry.
    nav_msgs::Odometry mocap_odom_msg;
    mocap_odom_msg.header.stamp = time;
    mocap_odom_msg.header.frame_id = ned_frame_id;
    mocap_odom_msg.child_frame_id = ground_truth_frame_id;

    tf::poseEigenToMsg(T_w_from_b_gt, mocap_odom_msg.pose.pose);
    mocap_odom_pub.publish(mocap_odom_msg);

#endif
    return;
}

} // namespace msckf_vio
