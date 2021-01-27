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

using namespace std;
using namespace Eigen;
using namespace cv;

string sData_path = "/home/ld/Downloads/dataset/navinfo/";
string sConfig_path = "/home/ld/vio_space/src/navinfo-vio-msckf/build_pc/config/";

int width = 0;
int height = 0;
float fx = 0.0;
float fy = 0.0;
float cx = 0.0;
float cy = 0.0;

Eigen::Vector3d p_world = Eigen::Vector3d::Zero();
Quaterniond q_w_from_b = Quaterniond::Identity();

ofstream of_pose_output;

double last_publish_time = -1.0;


void LoadConfigParam()
{
    cout << "1 LoadConfigParam" << endl;
    // Read rectification parameters
    string param_file = sConfig_path + "NavInfo.yaml";
    cout << "2 LoadConfigParam param_file: " << param_file << endl;
    FileStorage fsSettings(param_file, FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings, file: " << param_file << endl;
        return;
    }
    cout << "2 LoadConfigParam" << endl;

    fsSettings["Camera.fx"] >> fx;
    fsSettings["Camera.fy"] >> fy;
    fsSettings["Camera.cx"] >> cx;
    fsSettings["Camera.cy"] >> cy;

    height = fsSettings["Camera.height"];
    width = fsSettings["Camera.width"];

    cout << "LoadConfigParam width:" << fixed << width << " height:" << height
         << " fx:" << fx << " fy:" << fy
         << " cx:" << cx << " cy:" << cy
         << endl;
}

void PubImuData()
{
    string sImu_data_file = sData_path + "imu.txt";
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

        VISION_MsckfVio_SendImu(dStampNSec * 1e-3, vAcc, vGyr);

        // usleep(5000);
    }
    fsImu.close();
}

void PubImageData()
{
    string sImage_file = sData_path +  "cam.txt";

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

    // namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
    {
        std::istringstream ssImuData(sImage_line);
        ssImuData >> dStampNSec >> sImgFileName;
        // cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
        if (dStampNSec < 1294562431490)
        {
            continue;
        }
        if (dStampNSec > 1294563024790)
        {
            // return;
        }
        string left_image_path = sData_path + "cam0/" + sImgFileName;

        Mat imLeft = imread(left_image_path.c_str(), 0);
        if (imLeft.empty())
        {
            cerr << "image is empty! path: " << left_image_path << endl;
            return;
        }

        // 将RGB或RGBA图像转为灰度图像
        bool mbRGB = true;
        if(imLeft.channels()==3)
        {
            if(mbRGB)
                cvtColor(imLeft,imLeft,CV_RGB2GRAY);
            else
                cvtColor(imLeft,imLeft,CV_BGR2GRAY);
        }
        else if(imLeft.channels()==4)
        {
            if(mbRGB)
                cvtColor(imLeft,imLeft,CV_RGBA2GRAY);

            else
                cvtColor(imLeft,imLeft,CV_BGRA2GRAY);
        }

        // Mat left_image_rectified;
        // remap(imLeft, left_image_rectified, M1l, M2l, INTER_LINEAR);

        // imshow("imLeft", imLeft);
        // cvWaitKey(5);

        if (width != imLeft.cols || height != imLeft.rows)
        {
            cout << "width and height is not right!" << endl;
            return;
        }

        // send left and right images after rectified, and the depth image
        VISION_MsckfVio_SendMonoImage(dStampNSec * 1e-3, imLeft);

        usleep(100 * 1000);
    }
    fsImage.close();
}

void PublishMsckfVio(
    double timestamp,
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
        last_publish_time = timestamp;
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
    last_publish_time = timestamp;
}

void PublishPoints(double timestamp, 
        const std::vector<int> &curr_init_ids,
        const std::vector<Eigen::Vector2d> &curr_init_obs,
        const std::vector<Eigen::Vector3d> &curr_init_pts)
{
    // cout << "PublishPoints timestamp: " << fixed << timestamp << endl;
    // cout << "ids size id : " << curr_init_ids.size() << endl;
    // cout << "obs size id : " << curr_init_obs.size() << endl;
    // cout << "pts size id : " << curr_init_pts.size() << endl;
}


int main()
{
    cout << "1 main..." << endl;

    cpu_set_t mask;
    unsigned long cpuid = 2;
    CPU_ZERO(&mask);
    CPU_SET(cpuid, &mask);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) < 0)
    {
        perror("sched_setaffinity err:");
    }

    of_pose_output.open("./publish_pose.txt", ios::out | ios::trunc);
    if (!of_pose_output.is_open())
    {
        cerr << "of_pose_output is not open!" << endl;
        return -1;
    }
    // of_pose_output << "timestamp, px, py, pz, qx, qy, qz, qw" << endl;

    LoadConfigParam();
    VISION_MsckfVio_Init(fx, cx, cy, PublishMsckfVio, PublishPoints);


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
    return 0;
}
