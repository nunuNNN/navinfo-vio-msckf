#pragma once

#include <condition_variable>
#include <list>
#include <memory>
#include <queue>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

// 线程类
#ifndef USE_ROS_IMSHOW
#include "thread_run/thread.h"
#include "thread_run/condition.h"
#else
#include <thread>
#endif

#include "data_interface.h"
#include "msckf_vio/image_processor.h"
#include "msckf_vio/msckf_vio.h"


#ifndef USE_ROS_IMSHOW
class ImageProcessorThread : public Thread
{
public:    
    explicit ImageProcessorThread(
        const std::shared_ptr<msckf_vio::ImageProcessor>& p_image_processor);
    virtual ~ImageProcessorThread();

    void SetModulesInfo(const char* name,const int coreid)
    {
        bindCore(name,coreid);
    }
    
protected: 
    bool    run();

private:
    std::shared_ptr<msckf_vio::ImageProcessor> p_image_processor;

};


class MsckfVioThread : public Thread
{
public:    
    explicit MsckfVioThread(const std::shared_ptr<msckf_vio::MsckfVio>& p_msckf_vio);
    virtual ~MsckfVioThread();

    void SetModulesInfo(const char* name,const int coreid)
    {
        bindCore(name,coreid);
    }
    
protected: 
    bool    run();

private:
    std::shared_ptr<msckf_vio::MsckfVio> p_msckf_vio;

};
#endif


class VioManager
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VioManager();

    ~VioManager();

    void InitVioManager(void (*PublishVIO)(
                        double timestamp,
                        const Eigen::Vector3d &p,
                        const Eigen::Quaterniond &q,
                        float covVx, float covVy, float covVz,
                        uint8_t resetFlag,float rate1, float rate2),
                        void (*PublishPoints)(
                        double timestamp, 
                        const std::vector<int> &curr_init_ids,
                        const std::vector<Eigen::Vector2d> &curr_init_obs,
                        const std::vector<Eigen::Vector3d> &curr_init_pts));

    static VioManager* getInstance();

    void ReleaseVioManager();

#ifndef USE_ROS_IMSHOW
    void WaitForVisionThread();
    void NotifyVisionThread();
#endif
    /********************* push && get data *********************/
    void PushImu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

    void PushImage(double timestamp, const cv::Mat &left_rect_image);

    void PushPVQB(double timestamp, const Eigen::Isometry3d &T_world_from_imu,
                    const Eigen::Vector3d &velocity_in_world, 
                    const Eigen::Vector3d &bias_acc,
                    const Eigen::Vector3d &bias_gyr);

    bool GetCurrData(StrImageData &curr_str_image, vector<StrImuData> &curr_from_last_imu);

    bool GetPvqbByTime(double curImageTime, StrPVQB &str_pvqb);

    
    void PushFeaAndImu(double curImageTime,StrPVQB &str_pvqb, 
                        msckf_vio::Feature_measure_t &str_feature,
                        vector<StrImuData> &curr_from_last_imu);
    bool GetFeaAndImu(Feature_and_PVQB_t &feature_pvq);

private:
    /********************* Singleton Pattern *******************/
    static VioManager* s_pInstance;
#ifndef USE_ROS_IMSHOW
    cond_locker        vision_locker;
#endif

    /***********************  ros thread  **********************/
    void image_processor_thread();
    void msckf_vio_thread();

    /************************ vio data *************************/
    std::queue<StrImageData> images_datas;
    std::queue<StrImuData> imu_msg_buffer;
    std::list<StrPVQB> pvqb_datas;
    //mutex image
    std::mutex mutex_image;
    //mutex pose and velocity
    std::mutex mutex_pvqb;
    //mutex imu
    std::mutex mutex_imu;

    // TODO : add camera id in msg 
    std::deque<Feature_and_PVQB_t> feature_pvq_buffer;
        //mutex pose and velocity
    std::mutex mutex_fea_pvqb;


    // curr && last imu time 用于检查数据输入
    double curr_imu_timestamp = 0;
    double last_imu_timestamp = -1;
    // curr&&last pvq time
    double curr_pvq_timestamp = 0;
    double last_pvq_timestamp = -1;
    // curr&&last image time
    double curr_image_timestamp = 0;
    double last_image_timestamp = -1;

    /************************ thread *************************/
#ifndef USE_ROS_IMSHOW
    std::shared_ptr<ImageProcessorThread> ptr_image_processor_thread;
    std::shared_ptr<MsckfVioThread> ptr_msckf_vio_thread;
#else
    std::thread * ptr_image_processor_thread;
    std::thread * ptr_msckf_vio_thread;
#endif

    /************************ system *************************/
    // vio system
    std::shared_ptr<msckf_vio::ImageProcessor> p_image_processor;
    std::shared_ptr<msckf_vio::MsckfVio> p_msckf_vio;
};
