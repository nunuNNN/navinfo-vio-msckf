#include <iterator>
#include <ctime>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>
#include <vector>

#include "global_param.h"
#include "vio_manager.h"
#include "tic_toc.h"

using namespace std;
using namespace cv;
using namespace Eigen;

VioManager* VioManager::s_pInstance = NULL;
// PublishCallback function
void (*PublishCallbackVio)(uint64_t timestamp, const Vector3d &p,
                        const Quaterniond &q, float covVx, float covVy, float covVz, 
                        uint8_t resetFlag, float rate1, float rate2);
// CurrInitPtsCallback function
void (*CurrInitPtsCallback)(uint64_t timestamp, 
                        const std::vector<int> &curr_init_ids,
                        const std::vector<Eigen::Vector2d> &curr_init_obs,
                        const std::vector<Eigen::Vector3d> &curr_init_pts);

#ifndef USE_ROS_IMSHOW
ImageProcessorThread::ImageProcessorThread(
    const std::shared_ptr<msckf_vio::ImageProcessor>& p_image_processor)
    :Thread(),p_image_processor(p_image_processor)
{
}

ImageProcessorThread::~ImageProcessorThread()
{
}

bool ImageProcessorThread::run()
{
    if (!flag_thread_start[0])
    {
        VioManager::getInstance()->WaitForVisionThread();
        return true;
    }

    StrImageData curr_str_image;
    std::vector<StrImuData> curr_from_last_imu;
    if (!VioManager::getInstance()->GetCurrData(curr_str_image,curr_from_last_imu))
    {
        usleep(1000);
        return true;
    }

    double curr_image_timestamp = curr_str_image.timestamp;

    StrPVQB curr_pose_velocity;
    if (!VioManager::getInstance()->GetPvqbByTime(curr_image_timestamp, curr_pose_velocity))
    {
        usleep(1000);
        return true;
    }

    TicToc t_image_processor;
    p_image_processor->monoCallback(curr_image_timestamp, 
                                    curr_str_image.left_rect_image,
                                    curr_from_last_imu);
    
    msckf_vio::Feature_measure_t curr_features;
    p_image_processor->featureUpdateCallback(curr_features);

    // cout << "t_image_processor run time is : " << t_image_processor.toc() << endl;

    VioManager::getInstance()->PushFeaAndImu(curr_image_timestamp,
                                            curr_pose_velocity,
                                            curr_features,
                                            curr_from_last_imu);

    return true;
}

MsckfVioThread::MsckfVioThread(const std::shared_ptr<msckf_vio::MsckfVio>& p_msckf_vio)
    :Thread(),p_msckf_vio(p_msckf_vio)
{
}

MsckfVioThread::~MsckfVioThread()
{
}

bool MsckfVioThread::run()
{
    Feature_and_PVQB_t feature_pvq;
    if (!VioManager::getInstance()->GetFeaAndImu(feature_pvq))
    {
        usleep(1000);
        return true;       
    }

    TicToc t_msckf_vio;

    double curr_image_timestamp = feature_pvq.curr_time;

    msckf_vio::Feature_measure_t measure = feature_pvq.curr_features;
    msckf_vio::Ground_truth_t groundtruth;
    groundtruth.bias_acc            = feature_pvq.curr_pvqb.bias_acc;
    groundtruth.bias_gyr            = feature_pvq.curr_pvqb.bias_gyr;
    groundtruth.T_vel.v_in_world    = feature_pvq.curr_pvqb.v_in_world;
    groundtruth.T_vel.T_w_b = feature_pvq.curr_pvqb.T_world_from_imu;
    msckf_vio::Translation_velocity_t T_vel_out;

    p_msckf_vio->Process(curr_image_timestamp, 
                            measure,
                            feature_pvq.curr_from_last_imu,
                            groundtruth, 
                            T_vel_out);

    Eigen::Quaterniond q_imu_f_world = Quaterniond(T_vel_out.T_w_b.linear()).normalized();
    Eigen::Vector3d p_in_world = T_vel_out.T_w_b.translation();
    Eigen::Vector3d v_in_world = T_vel_out.v_in_world;
    Eigen::Vector3d v_in_body = T_vel_out.T_w_b.linear().transpose() * v_in_world;

    PublishCallbackVio(curr_image_timestamp * 1e9, p_in_world, q_imu_f_world, 
                        v_in_body.x(), v_in_body.y(), v_in_body.z(), 0, 0, 0);

    // 得到当前帧跟踪上已经完成初始化的点
    std::vector<int> curr_init_ids;
    std::vector<Eigen::Vector2d> curr_init_obs;
    std::vector<Eigen::Vector3d> curr_init_pts;
    
    if(p_msckf_vio->currFeatureInitCallback(curr_init_ids,
                                        curr_init_obs,
                                        curr_init_pts))
    {
        // TODO : 判断得到的三个变量长度是否相等

        CurrInitPtsCallback(curr_image_timestamp * 1e9, curr_init_ids,
                            curr_init_obs, curr_init_pts);
    }
    

    // cout << "t_msckf_vio run time is : " << t_msckf_vio.toc() << endl;
    return true;
}
#endif

/***************************** vio manager ***************************/
VioManager::VioManager()
{
}

VioManager::~VioManager()
{
}

#ifndef USE_ROS_IMSHOW
void VioManager::WaitForVisionThread()
{
    vision_locker.wait();
}

void VioManager::NotifyVisionThread()
{
    vision_locker.broadcast();
}
#endif

void VioManager::InitVioManager(void (*PublishVIO)(
                                    uint64_t timestamp,
                                    const Eigen::Vector3d &p,
                                    const Eigen::Quaterniond &q,
                                    float covVx, float covVy, float covVz,
                                    uint8_t resetFlag, float rate1, float rate2),
                                void (*PublishPoints)(
                                    uint64_t timestamp, 
                                    const std::vector<int> &curr_init_ids,
                                    const std::vector<Eigen::Vector2d> &curr_init_obs,
                                    const std::vector<Eigen::Vector3d> &curr_init_pts))
{
    /************************* out put *************************/
    PublishCallbackVio = PublishVIO;

    CurrInitPtsCallback = PublishPoints;

    /************************** VIO ****************************/
    p_image_processor.reset(new msckf_vio::ImageProcessor());

    // 后端只接收一路数据,初始化时优先选择前视参数
    Parameter_extrinsic_t ex_para;
    ex_para = params_extrinsic;
    p_msckf_vio.reset(new msckf_vio::MsckfVio(params_estimste, ex_para));

    /************************* thread **************************/
#ifndef USE_ROS_IMSHOW
    ptr_image_processor_thread = std::make_shared<ImageProcessorThread>(p_image_processor);
    ptr_msckf_vio_thread = std::make_shared<MsckfVioThread>(p_msckf_vio);

    ptr_image_processor_thread->SetModulesInfo("image", 2);
    ptr_msckf_vio_thread->SetModulesInfo("vio", 2);

    // start thread
    ptr_image_processor_thread->start();
    ptr_msckf_vio_thread->start();
#else
    ptr_image_processor_thread = new std::thread(&VioManager::image_processor_thread, this);
    ptr_msckf_vio_thread = new std::thread(&VioManager::msckf_vio_thread, this);
#endif
}

VioManager* VioManager::getInstance()
{
    if(s_pInstance == NULL)
        s_pInstance = new VioManager();

    return s_pInstance;
}

void VioManager::ReleaseVioManager()
{
#ifndef USE_ROS_IMSHOW
    ptr_image_processor_thread->stop();
    ptr_msckf_vio_thread->stop();

    vision_locker.broadcast();
#endif

    delete s_pInstance;
}

/***************************** push date in system ***************************/
void VioManager::PushImu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
{
    curr_imu_timestamp = timestamp;
    if (timestamp < last_imu_timestamp)
    {
        return;
    }
    last_imu_timestamp = timestamp;

    StrImuData str_imu_data;
    str_imu_data.timestamp = timestamp;
    str_imu_data.acc = acc;
    str_imu_data.gyr = gyr;

    mutex_imu.lock();
    imu_msg_buffer.push(str_imu_data);
    //if imu size is more than 500ms, then pop
    if (imu_msg_buffer.size() > 125)
    {
#ifndef USE_DATASET
        imu_msg_buffer.pop();
#endif
    }
    mutex_imu.unlock();
}

void VioManager::PushImage(double timestamp, const Mat &left_rect_image)
{
    StrImageData image_data;
    image_data.timestamp = timestamp;

    image_data.left_rect_image = left_rect_image.clone();

    mutex_image.lock();
    images_datas.push(image_data);
    //if image size is more than 5, drop!
    if (images_datas.size() > 5)
    {
        images_datas.pop();
    }
    mutex_image.unlock();
}

void VioManager::PushPVQB(double timestamp,
                            const Eigen::Isometry3d &T_world_from_imu,
                            const Eigen::Vector3d &velocity_in_world,
                            const Eigen::Vector3d &bias_acc,
                            const Eigen::Vector3d &bias_gyr)
{
    curr_pvq_timestamp = timestamp;

    if (curr_pvq_timestamp <= last_pvq_timestamp)
    {
        return;
    }
    StrPVQB str_pvqb;
    str_pvqb.timestamp = timestamp;
    str_pvqb.T_world_from_imu = T_world_from_imu;
    str_pvqb.v_in_world = velocity_in_world;
    str_pvqb.bias_acc = bias_acc;
    str_pvqb.bias_gyr = bias_gyr;

    mutex_pvqb.lock();
    pvqb_datas.push_back(str_pvqb);
    // if pvqb_datas is more than 500ms， drop
    if (pvqb_datas.size() > 125)
    {
#ifndef USE_DATASET
        pvqb_datas.erase(pvqb_datas.begin());
#endif
    }
    mutex_pvqb.unlock();

    last_pvq_timestamp = curr_pvq_timestamp;
}

bool VioManager::GetCurrData(StrImageData &curr_str_image, vector<StrImuData> &curr_from_last_imu)
{
    // get the current latest image
    mutex_image.lock();
    if (images_datas.empty())
    {
        // cout << "1 StartThread images_datas is empty!" << endl;
        mutex_image.unlock();
        return false;
    }

    int images_datas_size = images_datas.size();

    // if have too many image in images_datas, remove to only leave the newest image
    while (!images_datas.empty())
    {
        curr_str_image = images_datas.front();
        images_datas.pop();
    }
    mutex_image.unlock();

    curr_image_timestamp = curr_str_image.timestamp;

    // get the IMU between the two frames
    mutex_imu.lock();
    while (!imu_msg_buffer.empty())
    {
        auto imu_msg = imu_msg_buffer.front();

        double imu_time = imu_msg.timestamp;
        if (imu_time < last_image_timestamp)
        {
            imu_msg_buffer.pop();
            continue;
        }
        if (imu_time > curr_image_timestamp)
        {
            break;
        }

        curr_from_last_imu.push_back(imu_msg);

        imu_msg_buffer.pop();
    }
    mutex_imu.unlock();

    last_image_timestamp = curr_image_timestamp;

    return true;
}

bool VioManager::GetPvqbByTime(double curImageTime, StrPVQB &str_pvqb)
{
    mutex_pvqb.lock();

    auto now_it = pvqb_datas.begin();

    if (now_it == pvqb_datas.end())
    {
        // cout << "1 GetPvqbByTime pvqb_datas size is NULL" << endl;
        mutex_pvqb.unlock();
        return false;
    }

    double front_pvq_time = (*now_it).timestamp;
    auto p_end = pvqb_datas.end();
    double back_pvq_time = (*(--p_end)).timestamp;

    for (auto next_it = std::next(now_it); next_it != pvqb_datas.end(); ++now_it, ++next_it)
    {
        if ((*next_it).timestamp < curImageTime)
        {
            continue;
        }

        if ((*now_it).timestamp > curImageTime)
        {
            // cout << "2 GetPvqbByTime image timestamp is too old! pvqb_datas size is "
            //      << fixed << pvqb_datas.size()
            //      << " begin: " << ((pvqb_datas.front())->timestamp)
            //      << " end: " << ((pvqb_datas.back())->timestamp)
            //      << " curImageTime: " << curImageTime << endl;
            mutex_pvqb.unlock();
            return false;
        }

        if ((*now_it).timestamp <= curImageTime && (*next_it).timestamp >= curImageTime)
        {
            double delta_now = curImageTime - (*now_it).timestamp;
            double delta_next = (*next_it).timestamp - curImageTime;
            str_pvqb = delta_now < delta_next ? (*now_it) : (*next_it);
            //delete old elements
            for (auto it = pvqb_datas.begin(); (*it).timestamp < (*now_it).timestamp;)
            {
                it = pvqb_datas.erase(it);
            }
            // cout << "3 GetPvqbByTime Success size: " << pvqb_datas.size()
            //      << " front: " << pvqb_datas.front()->timestamp
            //      << " back: " << pvqb_datas.back()->timestamp << endl;
            mutex_pvqb.unlock();
            return true;
        }
    }
    // cout << "4 GetPvqbByTime failed!" << endl;
    mutex_pvqb.unlock();
    return false;
}


/******************************** 前后端交互数据 ******************************/
void VioManager::PushFeaAndImu(double curImageTime, StrPVQB &str_pvqb, 
                            msckf_vio::Feature_measure_t & str_feature, 
                            std::vector<StrImuData> &curr_from_last_imu)
{
    Feature_and_PVQB_t str_feature_pvqb;
    str_feature_pvqb.curr_time = curImageTime;
    str_feature_pvqb.camera_id = 0;
    str_feature_pvqb.curr_features = str_feature;
    str_feature_pvqb.curr_pvqb = str_pvqb;
    str_feature_pvqb.curr_from_last_imu = curr_from_last_imu;

    mutex_fea_pvqb.lock();
    feature_pvq_buffer.push_back(str_feature_pvqb);
    if (feature_pvq_buffer.size() > 5)
    {
        feature_pvq_buffer.erase(feature_pvq_buffer.begin());
    }
    mutex_fea_pvqb.unlock();
}

bool VioManager::GetFeaAndImu(Feature_and_PVQB_t &feature_pvq)
{
    mutex_fea_pvqb.lock();

    if (feature_pvq_buffer.empty())
    {
        mutex_fea_pvqb.unlock();
        return false;
    }

    feature_pvq = feature_pvq_buffer.front();
    feature_pvq_buffer.pop_front();

    mutex_fea_pvqb.unlock();

    return true;
}

#ifdef USE_ROS_IMSHOW
void VioManager::image_processor_thread()
{
    StrImageData curr_str_image;
    std::vector<StrImuData> curr_from_last_imu;
    if (!GetCurrData(curr_str_image,curr_from_last_imu))
    {
        usleep(1000);
        return;
    }

    double curr_image_timestamp = curr_str_image.timestamp;

    StrPVQB curr_pose_velocity;
    if (!GetPvqbByTime(curr_image_timestamp, curr_pose_velocity))
    {
        usleep(1000);
        return;
    }

    TicToc t_image_processor;
    p_image_processor->monoCallback(curr_image_timestamp, 
                                    curr_str_image.left_rect_image,
                                    curr_from_last_imu);
    
    msckf_vio::Feature_measure_t curr_features;
    p_image_processor->featureUpdateCallback(curr_features);

    // cout << "t_image_processor run time is : " << t_image_processor.toc() << endl;

    PushFeaAndImu(curr_image_timestamp, curr_pose_velocity, curr_features, curr_from_last_imu);
}

void VioManager::msckf_vio_thread()
{
    Feature_and_PVQB_t feature_pvq;
    if (!GetFeaAndImu(feature_pvq))
    {
        usleep(1000);
        return;       
    }

    TicToc t_msckf_vio;

    double curr_image_timestamp = feature_pvq.curr_time;

    msckf_vio::Feature_measure_t measure = feature_pvq.curr_features;
    msckf_vio::Ground_truth_t groundtruth;
    groundtruth.bias_acc            = feature_pvq.curr_pvqb.bias_acc;
    groundtruth.bias_gyr            = feature_pvq.curr_pvqb.bias_gyr;
    groundtruth.T_vel.v_in_world    = feature_pvq.curr_pvqb.v_in_world;
    groundtruth.T_vel.T_w_b = feature_pvq.curr_pvqb.T_world_from_imu;
    msckf_vio::Translation_velocity_t T_vel_out;

    p_msckf_vio->Process(curr_image_timestamp, 
                            measure,
                            feature_pvq.curr_from_last_imu,
                            groundtruth, 
                            T_vel_out);

    Eigen::Quaterniond q_imu_f_world = Quaterniond(T_vel_out.T_w_b.linear()).normalized();
    Eigen::Vector3d p_in_world = T_vel_out.T_w_b.translation();
    Eigen::Vector3d v_in_world = T_vel_out.v_in_world;
    Eigen::Vector3d v_in_body = T_vel_out.T_w_b.linear().transpose() * v_in_world;

    PublishCallbackVio(curr_image_timestamp * 1e9, p_in_world, q_imu_f_world, 
                        v_in_body.x(), v_in_body.y(), v_in_body.z(), 0, 0, 0);

    // 得到当前帧跟踪上已经完成初始化的点
    std::vector<int> curr_init_ids;
    std::vector<Eigen::Vector2d> curr_init_obs;
    std::vector<Eigen::Vector3d> curr_init_pts;
    
    if(p_msckf_vio->currFeatureInitCallback(curr_init_ids,
                                        curr_init_obs,
                                        curr_init_pts))
    {
        // TODO : 判断得到的三个变量长度是否相等

        CurrInitPtsCallback(curr_image_timestamp * 1e9, curr_init_ids,
                            curr_init_obs, curr_init_pts);
    }
    
}
#endif
