#include "static_initializer.h"

using namespace std;
using namespace Eigen;

namespace initializer {

bool StaticInitializer::tryStaInit(const std::vector<StrImuData>& curr_from_last_imu,
                                FeatureMeasurePtr measure)
{
    // cout << "curr_from_last_imu size is :" << curr_from_last_imu.size() << endl;
    // cout << "measure time is :" << measure->stamp << endl;
    // cout << "measure size is :" << measure->features.size() << endl;

    // return false if this is the 1st image for inclinometer-initializer
    if (0 == staticImgCounter) 
    {
        staticImgCounter++;
        init_features.clear();
        imu_msg_buffer.clear();

        // add features to init_features
        for (const auto& feature : measure->features)
            init_features[feature.id] = Vector2d(feature.u0, feature.v0);

        return false;
    }

    // calculate feature distance of matched features between prev and curr images
    InitFeatures curr_features;
    list<double> feature_dis;
    for (const auto& feature : measure->features) 
    {
        curr_features[feature.id] = Vector2d(feature.u0, feature.v0);
        if (init_features.find(feature.id) != init_features.end()) 
        {
            Vector2d vec2d_c(feature.u0, feature.v0);
            Vector2d vec2d_p = init_features[feature.id];
            feature_dis.push_back((vec2d_c-vec2d_p).norm());
        }
    }

    // return false if number of matched features is small
    if (feature_dis.empty()  || feature_dis.size()<20) 
    {
        staticImgCounter = 0;
        return false;
    }
    
    // ignore outliers rudely
    feature_dis.sort();
    auto itr = feature_dis.end();
    for (int i = 0; i < 19; i++)  
        itr--;

    double maxDis = *itr;

    // classified as static image if maxDis is smaller than threshold, otherwise reset image counter
    if (maxDis < max_feature_dis) 
    {
        staticImgCounter++;
        init_features.swap(curr_features);

        imu_msg_buffer.push_back(curr_from_last_imu);

        if (staticImgCounter < static_Num)  // return false if number of consecitive static images does not reach @static_Num
            return false;
    } 
    else 
    {
        //printf("inclinometer-initializer failed at No.%d static image.",staticImgCounter+1);
        staticImgCounter = 0;
        return false;
    }

    /* reach here means staticImgCounter is equal to static_Num */

    // initialize rotation and gyro bias by imu data between the 1st and the No.static_Num image
    // set take_off_stamp as time of the No.static_Num image
    // set initial imu_state as the state of No.static_Num image
    // earse imu data with timestamp earlier than the No.static_Num image
    initializeGravityAndBias();

    bInit = true;

    return true;
}


void StaticInitializer::initializeGravityAndBias() 
{
    // Initialize gravity and gyro bias.
    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc = Vector3d::Zero();

    int usefulImuSize = 0;
    double last_imu_time;
    for(int i=0; i<imu_msg_buffer.size(); i++)
    {
        std::vector<StrImuData> imu_between_frame;
        imu_between_frame = imu_msg_buffer[i];
        for(int j=0; j<imu_between_frame.size(); j++)
        {
            double imu_time = imu_between_frame[i].timestamp;
            Eigen::Vector3d acc = imu_between_frame[i].acc;
            Eigen::Vector3d gyr = imu_between_frame[i].gyr;
            sum_angular_vel += gyr;
            sum_linear_acc += acc;

            usefulImuSize++;

            last_imu_time = imu_time;
        }
    }

    // Compute gyro bias.
    gyro_bias = sum_angular_vel / usefulImuSize;

    // This is the gravity in the IMU frame.
    Vector3d gravity_imu = sum_linear_acc / usefulImuSize;

    // Initialize the initial orientation, so that the estimation
    // is consistent with the inertial frame.
    double gravity_norm = gravity_imu.norm();
    Vector3d gravity_world(0.0, 0.0, -gravity_norm);

    // Set rotation
    Quaterniond q0_w_i = Quaterniond::FromTwoVectors(gravity_imu, -gravity_world);	
    orientation = q0_w_i.coeffs();  

    // Set other state and timestamp
    state_time = last_imu_time;
    position = Vector3d(0.0, 0.0, 0.0);
    velocity = Vector3d(0.0, 0.0, 0.0);
    acc_bias = Vector3d(0.0, 0.0, 0.0);

    printf("Inclinometer-initializer completed by using %d imu data !!!\n\n",usefulImuSize);

    return;
}


void StaticInitializer::assignInitialState(msckf_vio::IMUState& imu_state) 
{
    if (!bInit) 
    {
        printf("Cannot assign initial state before initialization !!!\n");
        return;
    }
    // Set initial state
    imu_state.time = state_time;
    imu_state.gyro_bias = gyro_bias;
    imu_state.acc_bias = acc_bias;
    imu_state.orientation = orientation;
    imu_state.position = position;
    imu_state.velocity = velocity;

    return;
}

}