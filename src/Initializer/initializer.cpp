#include "initializer.h"

namespace initializer {

bool Initializer::tryIncInit(std::vector<StrImuData>& curr_from_last_imu,
        FeatureMeasurePtr measure, msckf_vio::IMUState& imu_state)
{
    if(first_init_image)
    {
        first_init_image = false;
        curr_from_last_imu.clear();
    }

    if(staticInitPtr->tryStaInit(curr_from_last_imu, measure)) 
    {
        staticInitPtr->assignInitialState(imu_state);
        return true;
    }
    else if (dynamicInitPtr->tryDynInit(curr_from_last_imu, measure)) 
    {
        dynamicInitPtr->assignInitialState(imu_state);
        return true;
    }

    return false;
}

}
