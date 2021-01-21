#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <stdlib.h>
#include <unistd.h>

#include <Eigen/Core>

#include "interface_msckf_vio.h"

#include <opencv2/opencv.hpp>

#include <ros/ros.h>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace ros;

string sData_path = "/home/ld/vio_space/src/navio_update/dataset/EuRoC/mav0/";

string sConfig_path = "/home/ld/vio_space/src/navinfo-vio-msckf/build_pc/config/";
string str_file_ground_truth = sConfig_path + "MH_01_ground_truth.txt";

Mat M1l, M2l;
Mat M1r, M2r;

int width = 0;
int height = 0;
float fx = 0.0;
float fy = 0.0;
float cx = 0.0;
float cy = 0.0;

float fb = 0.0;

float baseline = 0.0;

Eigen::Vector3d p_world = Eigen::Vector3d::Zero();
Quaterniond q_w_from_b = Quaterniond::Identity();

ofstream of_pose_output;

double last_publish_time = -1.0;

//EuRoC imu bias
// Vector3d acc_bias(-0.020544, 0.124837, 0.0618);
// Vector3d gyr_bias(-0.001806, 0.02094, 0.07687);
Vector3d acc_bias = Vector3d::Zero();
Vector3d gyr_bias = Vector3d::Zero();


void LoadConfigParam()
{
    cout << "1 LoadConfigParam" << endl;
    // Read rectification parameters
    string param_file = sConfig_path + "EuRoC.yaml";
    cout << "2 LoadConfigParam param_file: " << param_file << endl;
    FileStorage fsSettings(param_file, FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings, file: " << param_file << endl;
        return;
    }
    cout << "2 LoadConfigParam" << endl;

    Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    fb = fsSettings["Camera.bf"];

    // cout << "K_l:\n"
    //      << K_l << "\nK_r:\n"
    //      << K_r << "\nP_l:\n"
    //      << P_l << "\nP_r:\n"
    //      << P_r << "\nR_l:\n"
    //      << R_l << "\nR_r:\n"
    //      << R_r << "\nD_l:\n"
    //      << D_l << "\nD_r:\n"
    //      << D_r << endl
    //      << endl;

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() || rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return;
    }

    initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), Size(cols_l, rows_l), CV_32F, M1l, M2l);

    initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), Size(cols_r, rows_r), CV_32F, M1r, M2r);

    // set params
    width = cols_r;
    height = rows_r;
    // cout << "P_r:\n"
    //      << P_r << endl;
    fx = P_r.at<double>(0, 0);
    fy = P_r.at<double>(1, 1);
    cx = P_r.at<double>(0, 2);
    cy = P_r.at<double>(1, 2);

    baseline = fb / fx;

    cout << "LoadConfigParam width:" << fixed << width << " height:" << height
         << " fx:" << fx << " fy:" << fy
         << " cx:" << cx << " cy:" << cy
         << " fb:" << fb << " baseline:" << baseline << endl
         << endl;

    if (fx != fy)
    {
        cerr << "ERROR: Should: fx == fy" << endl;
    }
}

void PubImuData()
{
    string sImu_data_file = sConfig_path + "MH_01_imu0.txt";
    cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << endl;

    ifstream fsImu;
    fsImu.open(sImu_data_file.c_str());
    if (!fsImu.is_open())
    {
        cerr << "Failed to open imu file! " << sImu_data_file << endl;
        return;
    }

    std::string sImu_line;
    double dStampNSec = 0.0;
    Vector3d vAcc;
    Vector3d vGyr;

    while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
    {
        std::istringstream ssImuData(sImu_line);
        ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
        // cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;

        VISION_MsckfVio_SendImu(dStampNSec, vAcc - acc_bias, vGyr - gyr_bias);

        // usleep(5000);
    }
    fsImu.close();
}

void PublishGroundTruth()
{
    ifstream fs_ground_truth;

    fs_ground_truth.open(str_file_ground_truth.c_str());
    if (!fs_ground_truth.is_open())
    {
        cout << "1 PublishGroundTruth str_file_ground_truth is not open! " << str_file_ground_truth << endl;
        return;
    }
    while (!fs_ground_truth.eof())
    {
        string s;
        getline(fs_ground_truth, s);

        string::size_type idx;
        idx = s.find("#");

        if (!s.empty() && idx == string::npos)
        {
            stringstream ss;
            ss << s;

            double timestamp_ns;

            Vector3d p_wb;
            Quaterniond q_wb;
            Vector3d v_w;
            Vector3d b_a;
            Vector3d b_g;

            ss >> timestamp_ns >> p_wb.x() >> p_wb.y() >> p_wb.z() >>
                q_wb.w() >> q_wb.x() >> q_wb.y() >> q_wb.z() >>
                v_w.x() >> v_w.y() >> v_w.z() >>
                b_g.x() >> b_g.y() >> b_g.z() >>
                b_a.x() >> b_a.y() >> b_a.z();

            Eigen::Isometry3d T_w_from_b = Eigen::Isometry3d::Identity();
            T_w_from_b.rotate(q_wb);
            T_w_from_b.pretranslate(p_wb);
            // cout << "2 PublishGroundTruth " << fixed << timestamp_ns << endl;
            VISION_MsckfVio_SendPVQB(timestamp_ns, T_w_from_b, v_w, b_a, b_g);

            // usleep(10000);
        }
    }
    fs_ground_truth.close();
}

void getInitPose(double timestamp, const string &str_file_ground_truth, double &pose_stamp_out, Eigen::Isometry3d &T_w_from_b)
{
    ifstream fs_ground_truth;
    fs_ground_truth.open(str_file_ground_truth.c_str());

    if (!fs_ground_truth.is_open())
    {
        cout << "1 getInitPose str_file_ground_truth is not open! " << str_file_ground_truth << endl;
        return;
    }
    while (!fs_ground_truth.eof())
    {
        string s;
        getline(fs_ground_truth, s);

        string::size_type idx;
        idx = s.find("#");
        if (!s.empty() && idx == string::npos)
        {
            stringstream ss;
            ss << s;

            double t;

            Vector3d p_wb;
            Quaterniond q_wb;
            ss >> t >> p_wb.x() >> p_wb.y() >> p_wb.z() >> q_wb.w() >> q_wb.x() >> q_wb.y() >> q_wb.z();

            if (t <= timestamp)
            {
                T_w_from_b.rotate(q_wb);
                T_w_from_b.pretranslate(p_wb);
            }
            else
            {
                // cout << "2 getInitPose find timestamp is " << fixed << pose_stamp_out
                //      << " need < timestamp: " << timestamp
                //      << " p: " << p_wb.transpose()
                //      << " q: " << q_wb.coeffs().transpose()
                //      << endl;
                break;
            }

            pose_stamp_out = t;
        }
    }
    fs_ground_truth.close();
}

void PubImageData()
{
    string sImage_file = sConfig_path + "MH_01_cam0.txt";

    cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

    ifstream fsImage;
    fsImage.open(sImage_file.c_str());
    if (!fsImage.is_open())
    {
        cerr << "Failed to open image file! " << sImage_file << endl;
        return;
    }

    std::string sImage_line;
    double dStampNSec;
    string sImgFileName;

    // image size
    int width, height;

    // namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
    {
        std::istringstream ssImuData(sImage_line);
        ssImuData >> dStampNSec >> sImgFileName;
        // cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
        if (dStampNSec < 1403636601000000000)
        {
            continue;
        }
        if (dStampNSec > 1403638524927830000)
        {
            // return;
        }
        string left_image_path = sData_path + "cam0/data/" + sImgFileName;

        Mat imLeft = imread(left_image_path.c_str(), 0);
        if (imLeft.empty())
        {
            cerr << "image is empty! path: " << left_image_path << endl;
            return;
        }

        // cout << "left size: " << imLeft.cols << " " << imLeft.rows << endl;

        string right_image_path = sData_path + "cam1/data/" + sImgFileName;

        Mat imRight = imread(right_image_path.c_str(), 0);
        if (imRight.empty())
        {
            cerr << "image is empty! path: " << right_image_path << endl;
            return;
        }

        Mat left_image_rectified;
        remap(imLeft, left_image_rectified, M1l, M2l, INTER_LINEAR);
        // cout << "left_image_rectified size: " << left_image_rectified.cols << " " << left_image_rectified.rows << endl;

        Mat right_image_rectified;
        remap(imRight, right_image_rectified, M1r, M2r, INTER_LINEAR);

        // imshow("left_image_rectified", left_image_rectified);
        // imshow("right_image_rectified", right_image_rectified);
        // cvWaitKey(0);

        width = left_image_rectified.cols;
        height = left_image_rectified.rows;
        if (width != left_image_rectified.cols || width != right_image_rectified.cols || height != left_image_rectified.rows || height != right_image_rectified.rows)
        {
            cout << "width and height is not right!" << endl;
            return;
        }

        // send left and right images after rectified, and the depth image
        VISION_MsckfVio_SendMonoImage(dStampNSec, left_image_rectified);

        usleep(50 * 1000);
        // cvWaitKey(0);
    }
    fsImage.close();
}

void PublishMsckfVio(
    uint64_t timestamp,
    const Eigen::Vector3d &p,
    const Eigen::Quaterniond &q,
    float covVx, float covVy, float covVz,
    uint8_t resetFlag, float rate1, float rate2)
{
    // cout << "PublishMsckfVio timestamp: " << fixed << timestamp
    //      << " p: " << v_last_from_curr_in_last_imu.transpose()
    //      << " q: " << q_last_imu_from_curr_imu.coeffs().transpose() << endl;

    if (last_publish_time < 0)
    {
        last_publish_time = timestamp * 1e-9;
    }
    of_pose_output << fixed << timestamp
                   << "," << p.x() 
                   << "," << p.y()
                   << "," << p.z()
                   << "," << q.w()
                   << "," << q.x()
                   << "," << q.y()
                   << "," << q.z()
                   << endl;
    last_publish_time = timestamp * 1e-9;
}

void PublishPoints(uint64_t timestamp, 
        const std::vector<int> &curr_init_ids,
        const std::vector<Eigen::Vector2d> &curr_init_obs,
        const std::vector<Eigen::Vector3d> &curr_init_pts)
{
    // cout << "PublishPoints timestamp: " << fixed << timestamp << endl;
    // cout << "ids size id : " << curr_init_ids.size() << endl;
    // cout << "obs size id : " << curr_init_obs.size() << endl;
    // cout << "pts size id : " << curr_init_pts.size() << endl;
}


int main(int argc, char **argv)
{
    cout << "1 main..." << endl;
    init(argc, argv, "STEREO_VO");
    console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, console::levels::Debug);

    start();
    NodeHandle nh("~");

    of_pose_output.open("./publish_pose.txt", ios::out | ios::trunc);
    if (!of_pose_output.is_open())
    {
        cerr << "of_pose_output is not open!" << endl;
        return -1;
    }
    of_pose_output << "timestamp, px, py, pz, qx, qy, qz, qw" << endl;

    LoadConfigParam();
    VISION_MsckfVio_Init(fx, cx, cy, baseline, PublishMsckfVio, PublishPoints);

    thread thd_pub_pose(PublishGroundTruth);
    thd_pub_pose.join();

    std::thread thd_PubImuData(PubImuData);
    thd_PubImuData.join();

    std::thread thd_PubImageData(PubImageData);
    thd_PubImageData.join();

    printf("main finished.\n");

    // while (1)
    {
        sleep(5);
        cout << "--------------------------------------------" << endl;
    }
    VISION_MsckfVio_Stop();
    of_pose_output.close();

    spin();
    return 0;
}
