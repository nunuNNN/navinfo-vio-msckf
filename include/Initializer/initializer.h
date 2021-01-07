#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <iostream>
#include <memory>

#include "static_initializer.h"
#include "dynamic_initializer.h"

#include "imu_state.h"
#include "data_interface.h"

using namespace std;

namespace initializer {

class Initializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    Initializer() = delete;
    Initializer(const double& max_feature_dis_, const int& static_Num_, 
        const double& acc_n_, const double& acc_w_, const double& gyr_n_, 
        const double& gyr_w_, const Eigen::Matrix3d& R_c2b, const Eigen::Vector3d& t_bc_b)
    {
        first_init_image = true;

        staticInitPtr.reset(new StaticInitializer(max_feature_dis_, static_Num_));

        dynamicInitPtr.reset(new DynamicInitializer(acc_n_, acc_w_, gyr_n_, gyr_w_, R_c2b, t_bc_b));
    }

    // Destructor
    ~Initializer(){}

    // Interface for trying to initialize
    bool tryIncInit(std::vector<StrImuData>& curr_from_last_imu,
        FeatureMeasurePtr measure, msckf_vio::IMUState& imu_state);

private:

    // Inclinometer-initializer
    std::shared_ptr<StaticInitializer> staticInitPtr;
    // Dynamic initializer
    std::shared_ptr<DynamicInitializer> dynamicInitPtr;

    bool first_init_image;
};

}


#endif //INITIALIZER_H
