#ifndef STATIC_INITIALIZER_H
#define STATIC_INITIALIZER_H

#include <map>
#include <list>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "data_interface.h"
#include "imu_state.h"

using namespace std;

namespace initializer {

class StaticInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    StaticInitializer() = delete;
    StaticInitializer(const double& max_feature_dis_, const int& static_Num_) : 
                    max_feature_dis(max_feature_dis_), static_Num(static_Num_), bInit(false), state_time(0.0)
    {
        staticImgCounter = 0;
        init_features.clear();
        gyro_bias = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
        orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
        imu_msg_buffer.reserve(static_Num);
    }

    // Destructor
    ~StaticInitializer(){}

    // Interface for trying to initialize
    bool tryStaInit(const std::vector<StrImuData>& curr_from_last_imu,
                FeatureMeasurePtr measure);

    // Assign the initial state if initialized successfully
    void assignInitialState(msckf_vio::IMUState& imu_state);

    // If initialized
    bool ifInitialized() 
    {
        return bInit;
    }

private:

    typedef unsigned long long int FeatureIDType;

    // Maximum feature distance allowed bewteen static images
    double max_feature_dis;

    // Number of consecutive image for trigger static initializer
    unsigned int static_Num;

    // Defined type for initialization
    typedef std::map<FeatureIDType, Eigen::Vector2d, std::less<int>,
        Eigen::aligned_allocator<std::pair<const FeatureIDType, Eigen::Vector2d> > > InitFeatures;
    InitFeatures init_features;

    // imu msg buffer
    std::vector< std::vector<StrImuData> > imu_msg_buffer;

    // Counter for static images that will be used in inclinometer-initializer
    unsigned int staticImgCounter;

    // Initialize results
    double state_time;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;
    Eigen::Vector4d orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;

    // Flag indicating if initialized
    bool bInit;

    // initialize rotation and gyro bias by static imu datas
    void initializeGravityAndBias();
};

} // namespace initializer


#endif //STATIC_INITIALIZER_H
