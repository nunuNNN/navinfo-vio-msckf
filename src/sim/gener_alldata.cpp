//
// Created by hyj on 17-6-22.
//

#include <fstream>
#include <sys/stat.h>
#include <map>
#include <opencv2/opencv.hpp>
#include "imu.h"
#include "utilities.h"

// msckf_vio
#include "global_param.h"
#include "data_interface.h"
#include <msckf_vio/msckf_vio.h>

using namespace Eigen;
using namespace msckf_vio;

using Point = Eigen::Vector4d;
// using Points = std::vector<Point, Eigen::aligned_allocator<Point> >;
using Points = std::map<int, Point>;
using Line = std::pair<Eigen::Vector4d, Eigen::Vector4d>;
using Lines = std::vector<Line, Eigen::aligned_allocator<Line> >;

void CreatePointsLines(Points& points, Lines& lines);
Vector2d cam2pixel(const Vector2d &p, const Vector4d &cam_int);
void drawFeaturesStereo(Param &param);

// typedef struct
// {
//     uint32_t id;
//     float u0;
//     float v0;
//     float u1;
//     float v1;
// } Stereo_feature_t;

// typedef struct
// {
//     double stamp;
//     std::vector<Stereo_feature_t> features;
// } Feature_measure_t;

Feature_measure_t prev_features;
Feature_measure_t curr_features;

// vio BackEnd system
std::shared_ptr<msckf_vio::MsckfVio> p_msckf_vio; 


int main(){
    // 建立keyframe文件夹
    // mkdir("keyframe", 0777);

    // 生成3d points
    Points points;
    Lines lines;
    CreatePointsLines(points, lines);

    // IMU model
    Param params;
    IMU imuGen(params);
    bool first_frame = true;

    // vio BackEnd system 初始化
    float f_forw = 574.247559;
    float cx_forw = 242.220688;
    float cy_forw = 321.135437;
    float baseline_forw = 0.040203;
    float f_down = 0;
    float cx_down = 0;
    float cy_down = 0;
    float baseline_down = 0;
    InitParams(f_forw, cx_forw, cy_forw, baseline_forw,
                     f_down, cx_down, cy_down, baseline_down);

    p_msckf_vio = std::make_shared<msckf_vio::MsckfVio>(0);

    // create imu data
    // imu pose gyro acc
    std::vector< MotionData > imudata;
    std::vector< MotionData > imudata_noise;
    for (float t = params.t_start; t<params.t_end;) 
    {
        MotionData data = imuGen.MotionModel(t);
        imudata.push_back(data);

        // add imu noise
        MotionData data_noise = data;
        imuGen.addIMUnoise(data_noise);
        imudata_noise.push_back(data_noise);

        /************************** feed imu *****************************/
        p_msckf_vio->imuCallback(t, data.imu_acc, data.imu_gyro);
        /*****************************************************************/

        t += 1.0/params.imu_frequency;
    }
    imuGen.init_velocity_ = imudata[0].imu_velocity;
    imuGen.init_twb_ = imudata.at(0).twb;
    imuGen.init_Rwb_ = imudata.at(0).Rwb;
    save_Pose("imu_pose.txt", imudata);
    // save_Pose("imu_pose_noise.txt", imudata_noise);

    imuGen.testImu("imu_pose.txt", "imu_int_pose.txt");     // test the imu data, integrate the imu data to generate the imu trajecotry
    // imuGen.testImu("imu_pose_noise.txt", "imu_int_pose_noise.txt");

    // cam pose
    std::vector< MotionData > camdata;
    std::vector< MotionData > camdataRight;
    for (float t = params.t_start; t<params.t_end;) 
    {
        MotionData imu = imuGen.MotionModel(t);   // imu body frame to world frame motion
        MotionData cam;
        MotionData camRight;

        cam.timestamp = imu.timestamp;
        cam.Rwb = imu.Rwb * params.R_bc;    // cam frame in world frame
        cam.twb = imu.twb + imu.Rwb * params.t_bc; //  Tcw = Twb * Tbc ,  t = Rwb * tbc + twb
        camdata.push_back(cam);

        camRight.timestamp = imu.timestamp;
        camRight.Rwb = imu.Rwb * params.R_bc_right;    // cam frame in world frame
        camRight.twb = imu.twb + imu.Rwb * params.t_bc_right; //  Tcw = Twb * Tbc ,  t = Rwb * tbc + twb
        camdataRight.push_back(camRight);

        t += 1.0/params.cam_frequency;
    }
    // save_Pose("cam_pose.txt",camdata);
    // save_Pose_asTUM("cam_pose_tum.txt",camdata);

    // points obs in image
    double outTime = params.t_start;
    std::vector< MotionData > outpath;
    for(int n = 0; n < camdata.size(); ++n)
    {
        MotionData data = camdata[n];
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block(0, 0, 3, 3) = data.Rwb;
        Twc.block(0, 3, 3, 1) = data.twb;
        data = camdataRight[n];
        Eigen::Matrix4d TwcRight = Eigen::Matrix4d::Identity();
        TwcRight.block(0, 0, 3, 3) = data.Rwb;
        TwcRight.block(0, 3, 3, 1) = data.twb;

        // 遍历所有的特征点，看哪些特征点在视野里
        std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > points_cam;    // ３维点在当前cam视野里
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > features_cam;  // 对应的２维图像坐标
        std::vector<Vector3d> points_cur_camRight;    // ３维点在当前帧cam视野里
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > features_camRight;  // 对应的２维图像坐标
        std::map<int, Point>::iterator it; it = points.begin();
        while (it != points.end()) {
            Eigen::Vector4d pw = it->second;          // 最后一位存着feature id
            pw[3] = 1;                               //改成齐次坐标最后一位
            Eigen::Vector4d pc1 = Twc.inverse() * pw; // T_wc.inverse() * Pw  -- > point in cam frame
            Eigen::Vector4d pc1Right = TwcRight.inverse() * pw; // T_wc.inverse() * Pw  -- > point in cam frame

            if(pc1(2) < 0 || pc1Right(2) < 0) 
                continue; // z必须大于０,在摄像机坐标系前方

            Eigen::Vector2d obs(pc1(0)/pc1(2), pc1(1)/pc1(2)) ;
            Eigen::Vector2d obs_piexl = cam2pixel(obs, params.cam_intrinsics);
            points_cam.push_back(it->second);
            features_cam.push_back(obs_piexl);

            Eigen::Vector2d obsRight(pc1Right(0)/pc1Right(2), pc1Right(1)/pc1Right(2)) ;
            Eigen::Vector2d obsRight_piexl = cam2pixel(obsRight, params.cam_intrinsics);
            features_camRight.push_back(obsRight_piexl); 

            points_cur_camRight.push_back(Vector3d(pc1Right(0), pc1Right(1), pc1Right(2)));

            /*************************将特征点封装成数据结构体**************************/
            Stereo_feature_t tmp;
            tmp.id = it->first;
            tmp.u0 = obs.x();
            tmp.v0 = obs.y();
            tmp.u1 = obsRight.x();
            tmp.v1 = obsRight.y();
            curr_features.features.push_back(tmp);

            it++;
        }
        curr_features.stamp = outTime;
        if(first_frame)
        { 
            first_frame = false;
            prev_features = curr_features;
            curr_features.features.clear();
            outTime += 1.0/params.cam_frequency;
            continue;
        }

        /*********************** 对数据封装,进行vio仿真 ************************/
        // 调用优化函数，进行仿真优化
        // std::cout << "time is : " << outTime << ";  " 
        //         << "size of features : " << curr_features.features.size()
        //         << std::endl;
        // 对数据进行封装 Ground_truth(imu在world系下状态) && T_vel_out
        // 在这里在做一次imu轨迹生成,就可以不用利用cam轨迹转换到imu,为了省事,因为是同一个轨迹发生器生成
        MotionData imu = imuGen.MotionModel(outTime);   // imu body frame to world frame motion

        Ground_truth_t groundtruth;
        groundtruth.T_vel.T_w_b.linear() = imu.Rwb;
        groundtruth.T_vel.T_w_b.translation() = imu.twb;
        groundtruth.T_vel.v_in_world = imu.imu_velocity;
        groundtruth.bias_acc = imu.imu_acc_bias;
        groundtruth.bias_gyr = imu.imu_gyro_bias;

        Translation_velocity_t T_vel_out;
        // 将数据传入vio后端
        p_msckf_vio->Process(outTime, curr_features, groundtruth, T_vel_out);

        MotionData out_path;
        out_path.timestamp = outTime;
        out_path.Rwb = T_vel_out.T_w_b.linear();
        out_path.twb = T_vel_out.T_w_b.translation();
        out_path.imu_velocity = T_vel_out.v_in_world;
        outpath.push_back(out_path);

        // 特征点可视化
        drawFeaturesStereo(params);
        /********************************************************************/

        // 将当前特征点保存成为上一帧数据
        prev_features = curr_features;
        outTime += 1.0/params.cam_frequency;
        curr_features.features.clear();

        // save points
        // std::stringstream filename1;
        // filename1<<"keyframe/all_points_"<<n<<".txt";
        // save_features(filename1.str(),points_cam,features_cam);
    }
    save_Pose("out_path.txt", outpath);

    return 0;
}

void CreatePointsLines(Points& points, Lines& lines)
{
    std::ifstream f;
    f.open("house_model/house.txt");

    int featureID = 0;
    while(!f.eof())
    {
        std::string s;
        std::getline(f,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double x,y,z;
            ss >> x;
            ss >> y;
            ss >> z;
            Eigen::Vector4d pt0( x, y, z, 1 );
            ss >> x;
            ss >> y;
            ss >> z;
            Eigen::Vector4d pt1( x, y, z, 1 );

            bool isHistoryPoint = false;
            std::map<int, Point>::iterator it; it = points.begin();
            while (it != points.end()) {
                Eigen::Vector4d pt = it->second;
                if(pt == pt0)
                {
                    isHistoryPoint = true;
                }
                it++;
            }
            if(!isHistoryPoint) {
                // points.push_back(pt0);
                points[featureID] = pt0;
                featureID++;
            }

            isHistoryPoint = false;
            std::map<int, Point>::iterator it1; it1 = points.begin();
            while (it1 != points.end()) {
                Eigen::Vector4d pt = it1->second;
                if(pt == pt1)
                {
                    isHistoryPoint = true;
                }
                it1++;
            }
            if(!isHistoryPoint) {
                // points.push_back(pt1);
                points[featureID] = pt1;
                featureID++;
            }

            // pt0 = Twl * pt0;
            // pt1 = Twl * pt1;
            lines.emplace_back(pt0, pt1);   // lines
        }
    }

    // create more 3d points, you can comment this code
    int n = points.size();
    for (int j = 0; j < n; ++j) {
        Eigen::Vector4d p = points[j] + Eigen::Vector4d(0.5,0.5,-0.5,0);
        // points.push_back(p);
        points[n+j] = p;
    }

    // save points
    // save_points("all_points.txt", points);
}

Vector2d cam2pixel(const Vector2d &p, const Vector4d &cam_int)
{
    Vector2d res(p(0)*cam_int(0)+cam_int(1), p(1)*cam_int(2)+cam_int(3));
    return res;
}


void drawFeaturesStereo(Param &param) {
    // Colors for different features.
    cv::Scalar tracked(0, 255, 0);
    cv::Scalar new_feature(0, 255, 255);

    // Create an output image.
    int img_height = 640;
    int img_width = 480;
    cv::Mat out_img(img_height, img_width*2, CV_8U, cv::Scalar(255));

    cv::Point pt1(img_width, 0);
    cv::Point pt2(img_width, img_height);
    cv::line(out_img, pt1, pt2, cv::Scalar(0, 0, 0));

    // Collect features ids in the previous frame.
    std::vector<uint32_t> prev_ids(0);
    for (const auto& feature : prev_features.features)
        prev_ids.push_back(feature.id);

    // Collect feature points in the previous frame.
    std::map<uint32_t, cv::Point2f> prev_cam0_points;
    std::map<uint32_t, cv::Point2f> prev_cam1_points;
    for (const auto& feature : prev_features.features) {
        Eigen::Vector2d cam0 = Vector2d(feature.u0, feature.v0);
        Eigen::Vector2d cam1 = Vector2d(feature.u1, feature.v1);
        Eigen::Vector2d tmp0 = cam2pixel(cam0, param.cam_intrinsics);
        Eigen::Vector2d tmp1 = cam2pixel(cam1, param.cam_intrinsics);
        prev_cam0_points[feature.id] = cv::Point2f(tmp0.x(), tmp0.y());
        prev_cam1_points[feature.id] = cv::Point2f(tmp1.x(), tmp1.y());
    }

    // Collect feature points in the current frame.
    std::map<uint32_t, cv::Point2f> curr_cam0_points;
    std::map<uint32_t, cv::Point2f> curr_cam1_points;
    for (const auto& feature : curr_features.features) {
        Eigen::Vector2d cam0 = Vector2d(feature.u0, feature.v0);
        Eigen::Vector2d cam1 = Vector2d(feature.u1, feature.v1);
        Eigen::Vector2d tmp0 = cam2pixel(cam0, param.cam_intrinsics);
        Eigen::Vector2d tmp1 = cam2pixel(cam1, param.cam_intrinsics);
        curr_cam0_points[feature.id] = cv::Point2f(tmp0.x(), tmp0.y());
        curr_cam1_points[feature.id] = cv::Point2f(tmp1.x(), tmp1.y());
    }

    // Draw tracked features.
    for (const auto& id : prev_ids) {
        if (prev_cam0_points.find(id) != prev_cam0_points.end() && curr_cam0_points.find(id) != curr_cam0_points.end()) {
            cv::Point2f prev_pt0 = prev_cam0_points[id];
            cv::Point2f prev_pt1 = prev_cam1_points[id] + cv::Point2f(img_width, 0.0);
            cv::Point2f curr_pt0 = curr_cam0_points[id];
            cv::Point2f curr_pt1 = curr_cam1_points[id] + cv::Point2f(img_width, 0.0);

            cv::circle(out_img, curr_pt0, 3, tracked, -1);
            cv::circle(out_img, curr_pt1, 3, tracked, -1);
            cv::line(out_img, prev_pt0, curr_pt0, tracked, 1);
            cv::line(out_img, prev_pt1, curr_pt1, tracked, 1);

            prev_cam0_points.erase(id);
            prev_cam1_points.erase(id);
            curr_cam0_points.erase(id);
            curr_cam1_points.erase(id);
        }
    }

    // Draw new features.
    for (const auto& new_cam0_point : curr_cam0_points) {
        cv::Point2f pt0 = new_cam0_point.second;
        cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] +
        cv::Point2f(img_width, 0.0);

        cv::circle(out_img, pt0, 3, new_feature, -1);
        cv::circle(out_img, pt1, 3, new_feature, -1);
    }

    cv::imshow("Feature", out_img);
    cv::waitKey(50);

    return;
}
