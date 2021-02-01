#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <Eigen/Dense>

#include <msckf_vio/image_processor.h>


#include "global_param.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace msckf_vio
{
ImageProcessor::ImageProcessor()
    : is_first_img(true),
      prev_features_ptr(new GridFeatures()),
      curr_features_ptr(new GridFeatures()),
      camera_id(0)
{
#ifdef USE_ROS_IMSHOW
    ros::NodeHandle nh("~");
    image_transport::ImageTransport it(nh);
    debug_stereo_pub = it.advertise("/firefly_sbx/image_processor/debug_stereo_image", 1);
#endif
    initialize();
}

ImageProcessor::~ImageProcessor()
{
    destroyAllWindows();
    //printf("Feature lifetime statistics:");
    //featureLifetimeStatistics();
    return;
}

bool ImageProcessor::initialize()
{
    cout << " ImageProcessor::initialize camera_id: " << camera_id << endl;

    cam0_intrinsics[0] = params_camera.m_f;
    cam0_intrinsics[1] = params_camera.m_f;
    cam0_intrinsics[2] = params_camera.m_cx;
    cam0_intrinsics[3] = params_camera.m_cy;

    cam0_distortion_coeffs[0] = params_camera.m_k1;
    cam0_distortion_coeffs[1] = params_camera.m_k2;
    cam0_distortion_coeffs[2] = params_camera.m_p1;
    cam0_distortion_coeffs[3] = params_camera.m_p2;

    // Eigen::Isometry3d -> opencv -- Transform cam0 frames with imu frames.
    Eigen::Matrix3d R_imu_from_cam0_e = T_cam0_from_imu.linear().transpose();
    // eigen2cv(R_imu_from_cam0_e, R_imu_from_cam0);
    Eigen::Vector3d t_imu_from_cam0_e = T_cam0_from_imu.inverse().translation();
    // eigen2cv(t_imu_from_cam0_e, t_imu_from_cam0);
    for (int i=0; i<3; i++)
    {
        t_imu_from_cam0(i) = t_imu_from_cam0_e(i);
        for (int j=0; j<3; j++)
        {
            R_imu_from_cam0(i,j) = R_imu_from_cam0_e(i,j);
        }
    }

#ifndef USE_OPENCV3
    detector = cv::FastFeatureDetector(processor_config.fast_threshold);
#else
    detector_ptr = FastFeatureDetector::create(processor_config.fast_threshold);
#endif

    printf("===========================================\n");
    printf("cam0_intrinscs: %f, %f, %f, %f\n",
           cam0_intrinsics[0], cam0_intrinsics[1],
           cam0_intrinsics[2], cam0_intrinsics[3]);
    printf("cam0_distortion_coefficients: %f, %f, %f, %f\n",
           cam0_distortion_coeffs[0], cam0_distortion_coeffs[1],
           cam0_distortion_coeffs[2], cam0_distortion_coeffs[3]);

    cout << "R_imu_from_cam0: " << R_imu_from_cam0 << endl;
    cout << "t_imu_from_cam0: " << t_imu_from_cam0.t() << endl;

    printf("grid_row: %d\n",
           processor_config.grid_row);
    printf("grid_col: %d\n",
           processor_config.grid_col);
    printf("grid_min_feature_num: %d\n",
           processor_config.grid_min_feature_num);
    printf("grid_max_feature_num: %d\n",
           processor_config.grid_max_feature_num);
    printf("pyramid_levels: %d\n",
           processor_config.pyramid_levels);
    printf("patch_size: %d\n",
           processor_config.patch_size);
    printf("fast_threshold: %d\n",
           processor_config.fast_threshold);
    printf("max_iteration: %d\n",
           processor_config.max_iteration);
    printf("track_precision: %f\n",
           processor_config.track_precision);
    printf("ransac_threshold: %f\n",
           processor_config.ransac_threshold);
    printf("stereo_threshold: %f\n",
           processor_config.stereo_threshold);
    printf("===========================================\n");

    return true;
}

void ImageProcessor::monoCallback(double timestamp, 
                                const Mat &cam0_img,
                                std::vector<StrImuData> &curr_from_last_imu)
{
    last_cam_timestamp = curr_cam_timestamp;
    prev_features_ptr = curr_features_ptr;
    std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

    // Initialize the current features to empty vectors.
    curr_features_ptr.reset(new GridFeatures());
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
    {
        (*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
    }


    curr_cam_timestamp = timestamp;
    // cout << "curr_cam_timestamp: " << fixed << curr_cam_timestamp << endl;

    // 复制imu数据
    imu_curr_from_last = curr_from_last_imu;

    // Get the current image.
    curr_cam0_img = cam0_img.clone();
    // Build the image pyramids once since they're used at multiple places
    createImagePyramids();

    // Detect features in the first frame.
    if (is_first_img)
    {
        initializeFirstFrame();
        is_first_img = false;

        // Draw results.
        drawFeaturesMono();
    }
    else
    {
        // Track the feature in the previous image.
        trackFeatures();

        // Add new features into the current image.
        addNewFeatures();

        // Add new features into the current image.
        pruneGridFeatures();

        // Draw results.
        drawFeaturesMono();
    }

    //updateFeatureLifetime();

    return;
}

void ImageProcessor::createImagePyramids()
{
    buildOpticalFlowPyramid(
        curr_cam0_img, curr_cam0_pyramid_,
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels, true, BORDER_REFLECT_101,
        BORDER_CONSTANT, false);
}

void ImageProcessor::initializeFirstFrame()
{
    // Size of each grid.
    Mat img = curr_cam0_img.clone();
    static int grid_height = img.rows / processor_config.grid_row;
    static int grid_width = img.cols / processor_config.grid_col;

    // Detect new features on the frist image.
    vector<KeyPoint> new_features(0);

    // cout << "cam0_curr_img.img " << img.cols << " " << img.rows
    //      << " " << img.type() << endl;
#ifndef USE_OPENCV3
    detector.detect(img, new_features);
#else
    detector_ptr->detect(img, new_features);
#endif

    cout << "new_features size: " << new_features.size() << endl;

    vector<cv::Point2f> cam0_inliers(0);
    vector<float> response_inliers(0);
    for (int i = 0; i < new_features.size(); ++i)
    {
        cam0_inliers.push_back(new_features[i].pt);
        response_inliers.push_back(new_features[i].response);
    }

    // Group the features into grids
    GridFeatures grid_new_features;
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col;
         ++code)
    {
        grid_new_features[code] = vector<FeatureMetaData>(0);
    }

    for (int i = 0; i < cam0_inliers.size(); ++i)
    {
        const cv::Point2f &cam0_point = cam0_inliers[i];
        const float &response = response_inliers[i];

        int row = static_cast<int>(cam0_point.y / grid_height);
        int col = static_cast<int>(cam0_point.x / grid_width);
        int code = row * processor_config.grid_col + col;

        FeatureMetaData new_feature;
        new_feature.response = response;
        new_feature.cam0_point = cam0_point;
        grid_new_features[code].push_back(new_feature);
    }

    // Sort the new features in each grid based on its response.
    for (auto &item : grid_new_features)
    {
        std::sort(item.second.begin(), item.second.end(),
                  &ImageProcessor::featureCompareByResponse);
    }
    // Collect new features within each grid with high response.
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col;
         ++code)
    {
        vector<FeatureMetaData> &features_this_grid = (*curr_features_ptr)[code];
        vector<FeatureMetaData> &new_features_this_grid = grid_new_features[code];

        for (int k = 0; k < processor_config.grid_min_feature_num &&
                        k < new_features_this_grid.size();
             ++k)
        {
            features_this_grid.push_back(new_features_this_grid[k]);
            features_this_grid.back().id = next_feature_id++;
            features_this_grid.back().lifetime = 1;
        }
    }

    return;
}

void ImageProcessor::predictFeatureTracking(
    const vector<cv::Point2f> &input_pts,
    const cv::Matx33f &R_curr_from_prev,
    const cv::Vec4d &intrinsics,
    vector<cv::Point2f> &compensated_pts)
{
    // Return directly if there are no input features.
    if (input_pts.size() == 0)
    {
        compensated_pts.clear();

        return;
    }
    compensated_pts.resize(input_pts.size());

    // Intrinsic matrix.
    cv::Matx33f K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);
    cv::Matx33f H = K * R_curr_from_prev * K.inv();

    for (int i = 0; i < input_pts.size(); ++i)
    {
        cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
        cv::Vec3f p2 = H * p1;
        compensated_pts[i].x = p2[0] / p2[2];
        compensated_pts[i].y = p2[1] / p2[2];
    }

    return;
}

void ImageProcessor::trackFeatures()
{
    // Size of each grid.
    static int grid_height = curr_cam0_img.rows / processor_config.grid_row;
    static int grid_width = curr_cam0_img.cols / processor_config.grid_col;

    // Compute a rough relative rotation which takes a vector
    // from the previous frame to the current frame.
    Matx33f cam0_R_curr_from_prev;
    integrateImuData(cam0_R_curr_from_prev);

    // Organize the features in the previous image.
    vector<FeatureIDType> prev_ids(0);
    vector<int> prev_lifetime(0);
    vector<Point2f> prev_cam0_points(0);

    for (const auto &item : *prev_features_ptr)
    {
        for (const auto &prev_feature : item.second)
        {
            prev_ids.push_back(prev_feature.id);
            prev_lifetime.push_back(prev_feature.lifetime);
            prev_cam0_points.push_back(prev_feature.cam0_point);
        }
    }

    // Number of the features before tracking.
    before_tracking = prev_cam0_points.size();

    // Abort tracking if there is no features in
    // the previous frame.
    if (prev_ids.size() == 0)
    {
        return;
    }

    // Track features using LK optical flow method.
    vector<Point2f> curr_cam0_points(0);
    vector<unsigned char> track_inliers(0);

    predictFeatureTracking(prev_cam0_points, cam0_R_curr_from_prev, cam0_intrinsics, curr_cam0_points);

    calcOpticalFlowPyrLK(prev_cam0_pyramid_, curr_cam0_pyramid_,
                         prev_cam0_points, curr_cam0_points,
                         track_inliers, noArray(),
                         Size(processor_config.patch_size, processor_config.patch_size),
                         processor_config.pyramid_levels,
                         TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                                      processor_config.max_iteration,
                                      processor_config.track_precision),
                         cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < curr_cam0_points.size(); ++i)
    {
        if (track_inliers[i] == 0)
        {
            continue;
        }
        if (curr_cam0_points[i].y < 0 ||
            curr_cam0_points[i].y > curr_cam0_img.rows - 1 ||
            curr_cam0_points[i].x < 0 ||
            curr_cam0_points[i].x > curr_cam0_img.cols - 1)
        {
            track_inliers[i] = 0;
        }
    }

    // Collect the tracked points.
    vector<FeatureIDType> prev_tracked_ids(0);
    vector<int> prev_tracked_lifetime(0);
    vector<Point2f> prev_tracked_cam0_points(0);
    vector<Point2f> curr_tracked_cam0_points(0);

    removeUnmarkedElements(prev_ids, track_inliers, prev_tracked_ids);
    removeUnmarkedElements(prev_lifetime, track_inliers, prev_tracked_lifetime);
    removeUnmarkedElements(prev_cam0_points, track_inliers, prev_tracked_cam0_points);
    removeUnmarkedElements(curr_cam0_points, track_inliers, curr_tracked_cam0_points);

    // Number of features left after tracking.
    after_tracking = curr_tracked_cam0_points.size();

    // RANSAC on temporal image pairs of cam0.
    vector<int> cam0_ransac_inliers(0);
    twoPointRansac(prev_tracked_cam0_points, curr_tracked_cam0_points,
                   cam0_R_curr_from_prev, cam0_intrinsics,
                   cam0_distortion_coeffs, processor_config.ransac_threshold,
                   0.99, cam0_ransac_inliers);

    // Number of features after ransac.
    after_ransac = 0;

    for (int i = 0; i < cam0_ransac_inliers.size(); ++i)
    {
        if (cam0_ransac_inliers[i] == 0)
        {
            continue;
        }

        int row = static_cast<int>(curr_tracked_cam0_points[i].y / grid_height);
        int col = static_cast<int>(curr_tracked_cam0_points[i].x / grid_width);
        int code = row * processor_config.grid_col + col;

        (*curr_features_ptr)[code].push_back(FeatureMetaData());

        FeatureMetaData &grid_new_feature = (*curr_features_ptr)[code].back();
        grid_new_feature.id = prev_tracked_ids[i];
        grid_new_feature.lifetime = ++prev_tracked_lifetime[i];
        grid_new_feature.cam0_point = curr_tracked_cam0_points[i];

        ++after_ransac;
    }


    // Compute the tracking rate.
    int prev_feature_num = 0;
    for (const auto &item : *prev_features_ptr)
    {
        prev_feature_num += item.second.size();
    }

    int curr_feature_num = 0;
    for (const auto &item : *curr_features_ptr)
    {
        curr_feature_num += item.second.size();
    }

    printf("\033[0;32m candidates: %d; raw track: %d; after ransac: %d; ransac: %d/%d=%f\033[0m\n",
           before_tracking, after_tracking, after_ransac,
           curr_feature_num, prev_feature_num,
           static_cast<double>(curr_feature_num) /
               (static_cast<double>(prev_feature_num) + 1e-5));

    return;
}

void ImageProcessor::undistortPoints(
    const std::vector<cv::Point2f> &pts_in,
    const cv::Vec4d &intrinsics,
    const cv::Vec4d &distortion_coeffs,
    std::vector<cv::Point2f> &pts_out,
    const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics)
{
    if (pts_in.size() == 0)
    {
        return;
    }

    const cv::Matx33d K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);

    const cv::Matx33d K_new(
        new_intrinsics[0], 0.0, new_intrinsics[2],
        0.0, new_intrinsics[1], new_intrinsics[3],
        0.0, 0.0, 1.0);

    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                        rectification_matrix, K_new);
}

std::vector<cv::Point2f> ImageProcessor::distortPoints(
    const std::vector<cv::Point2f> &pts_in,
    const cv::Vec4d &intrinsics,
    const cv::Vec4d &distortion_coeffs)
{
    const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
                        0.0, intrinsics[1], intrinsics[3],
                        0.0, 0.0, 1.0);

    std::vector<cv::Point2f> pts_out;

    std::vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                      distortion_coeffs, pts_out);

    return pts_out;
}

void ImageProcessor::addNewFeatures()
{
    const Mat &curr_img = curr_cam0_img;

    // Size of each grid.
    static int grid_height = curr_cam0_img.rows / processor_config.grid_row;
    static int grid_width = curr_cam0_img.cols / processor_config.grid_col;

    // Create a mask to avoid redetecting existing features.
    Mat mask(curr_img.rows, curr_img.cols, CV_8U, Scalar(1));

    for (const auto &features : *curr_features_ptr)
    {
        for (const auto &feature : features.second)
        {
            const int y = static_cast<int>(feature.cam0_point.y);
            const int x = static_cast<int>(feature.cam0_point.x);

            int up_lim = y - 2, bottom_lim = y + 3,
                left_lim = x - 2, right_lim = x + 3;
            if (up_lim < 0)
            {
                up_lim = 0;
            }
            if (bottom_lim > curr_img.rows)
            {
                bottom_lim = curr_img.rows;
            }
            if (left_lim < 0)
            {
                left_lim = 0;
            }
            if (right_lim > curr_img.cols)
            {
                right_lim = curr_img.cols;
            }

            Range row_range(up_lim, bottom_lim);
            Range col_range(left_lim, right_lim);
            int row_lim = right_lim - left_lim;
            int col_lim = bottom_lim - up_lim;
            Mat tmp(row_lim, col_lim, CV_8U, Scalar(0));
            mask(row_range, col_range) = tmp;
        }
    }

    // Detect new features.
    vector<KeyPoint> new_features(0);
#ifndef USE_OPENCV3
    detector.detect(curr_img, new_features, mask);
#else
    detector_ptr->detect(curr_img, new_features, mask);
#endif

    // Collect the new detected features based on the grid.
    // Select the ones with top response within each grid afterwards.
    vector<vector<KeyPoint>> new_feature_sieve(
        processor_config.grid_row * processor_config.grid_col);
    for (const auto &feature : new_features)
    {
        int row = static_cast<int>(feature.pt.y / grid_height);
        int col = static_cast<int>(feature.pt.x / grid_width);
        new_feature_sieve[row * processor_config.grid_col + col].push_back(feature);
    }

    new_features.clear();
    for (auto &item : new_feature_sieve)
    {
        if (item.size() > processor_config.grid_max_feature_num)
        {
            std::sort(item.begin(), item.end(),
                      &ImageProcessor::keyPointCompareByResponse);
            item.erase(item.begin() + processor_config.grid_max_feature_num, item.end());
        }
        new_features.insert(new_features.end(), item.begin(), item.end());
    }

    int detected_new_features = new_features.size();


    vector<cv::Point2f> cam0_inliers(0);
    vector<float> response_inliers(0);
    for (int i = 0; i < new_features.size(); ++i)
    {
        cam0_inliers.push_back(new_features[i].pt);
        response_inliers.push_back(new_features[i].response);
    }

    int matched_new_features = cam0_inliers.size();

    if (matched_new_features < 5 &&
        static_cast<double>(matched_new_features) /
                static_cast<double>(detected_new_features) <
            0.1)
    {
        printf("Images at seems unsynced...");
    }

    // Group the features into grids
    GridFeatures grid_new_features;
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col;
         ++code)
    {
        grid_new_features[code] = vector<FeatureMetaData>(0);
    }

    for (int i = 0; i < cam0_inliers.size(); ++i)
    {
        const cv::Point2f &cam0_point = cam0_inliers[i];
        const float &response = response_inliers[i];

        int row = static_cast<int>(cam0_point.y / grid_height);
        int col = static_cast<int>(cam0_point.x / grid_width);
        int code = row * processor_config.grid_col + col;

        FeatureMetaData new_feature;
        new_feature.response = response;
        new_feature.cam0_point = cam0_point;
        grid_new_features[code].push_back(new_feature);
    }

    // Sort the new features in each grid based on its response.
    for (auto &item : grid_new_features)
    {
        std::sort(item.second.begin(), item.second.end(),
                  &ImageProcessor::featureCompareByResponse);
    }

    int new_added_feature_num = 0;
    // Collect new features within each grid with high response.
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col;
         ++code)
    {
        vector<FeatureMetaData> &features_this_grid = (*curr_features_ptr)[code];
        vector<FeatureMetaData> &new_features_this_grid = grid_new_features[code];

        if (features_this_grid.size() >=
            processor_config.grid_min_feature_num)
        {
            continue;
        }

        int vacancy_num = processor_config.grid_min_feature_num -
                          features_this_grid.size();
        for (int k = 0; k < vacancy_num && k < new_features_this_grid.size(); ++k)
        {
            features_this_grid.push_back(new_features_this_grid[k]);
            features_this_grid.back().id = next_feature_id++;
            features_this_grid.back().lifetime = 1;

            ++new_added_feature_num;
        }
    }

    // printf("\033[0;33m detected: %d; matched: %d; new added feature: %d\033[0m\n",
    //        detected_new_features, matched_new_features, new_added_feature_num);

    return;
}

void ImageProcessor::pruneGridFeatures()
{
    for (auto &item : *curr_features_ptr)
    {
        auto &grid_features = item.second;
        // Continue if the number of features in this grid does
        // not exceed the upper bound.
        if (grid_features.size() <= processor_config.grid_max_feature_num)
        {
            continue;
        }

        std::sort(grid_features.begin(), grid_features.end(),
                  &ImageProcessor::featureCompareByLifetime);
        grid_features.erase(grid_features.begin() +
                                processor_config.grid_max_feature_num,
                            grid_features.end());
    }
    return;
}

void ImageProcessor::integrateImuData(Matx33f& cam0_R_p_c) 
{
    // cout << "last_cam_timestamp : " << last_cam_timestamp << endl;
    // cout << "curr_cam_timestamp : " << curr_cam_timestamp << endl;
    
    int sizeooo = imu_curr_from_last.size();

    // Find the start and the end limit within the imu msg buffer.
    Vec3d mean_ang_vel(0.0, 0.0, 0.0);
    int num_imu_iter = 0;

    for(int i=0; i<imu_curr_from_last.size(); i++)
    {
        auto imu_msg = imu_curr_from_last[i];
        mean_ang_vel += Vec3d(imu_msg.gyr.x(), imu_msg.gyr.y(), imu_msg.gyr.z());
        num_imu_iter++;
    }

    // cout << "num_imu_iter : " << num_imu_iter <<endl;

    if (num_imu_iter > 0)
    {
        mean_ang_vel *= 1.0f / num_imu_iter;
    }

    // Transform the mean angular velocity from the IMU
    // frame to the cam0 frames.
    Vec3f cam0_mean_ang_vel = R_imu_from_cam0.t() * mean_ang_vel;

    // Compute the relative rotation.
    double dtime = curr_cam_timestamp - last_cam_timestamp;
    Rodrigues(cam0_mean_ang_vel*dtime, cam0_R_p_c);
    cam0_R_p_c = cam0_R_p_c.t();

    return;
}

void ImageProcessor::rescalePoints(
    vector<Point2f> &pts1, vector<Point2f> &pts2,
    float &scaling_factor)
{

    scaling_factor = 0.0f;

    for (int i = 0; i < pts1.size(); ++i)
    {
        scaling_factor += sqrt(pts1[i].dot(pts1[i]));
        scaling_factor += sqrt(pts2[i].dot(pts2[i]));
    }

    scaling_factor = (pts1.size() + pts2.size()) /
                     scaling_factor * sqrt(2.0f);

    for (int i = 0; i < pts1.size(); ++i)
    {
        pts1[i] *= scaling_factor;
        pts2[i] *= scaling_factor;
    }

    return;
}

void ImageProcessor::twoPointRansac(
    const vector<Point2f> &pts1, const vector<Point2f> &pts2,
    const cv::Matx33f &R_curr_from_prev, const cv::Vec4d &intrinsics,
    const cv::Vec4d &distortion_coeffs,
    const double &inlier_error,
    const double &success_probability,
    vector<int> &inlier_markers)
{

    // Check the size of input point size.
    if (pts1.size() != pts2.size())
    {
        printf("Sets of different size (%lu and %lu) are used...",
               pts1.size(), pts2.size());
    }

    double norm_pixel_unit = 2.0 / (intrinsics[0] + intrinsics[1]);
    int iter_num = static_cast<int>(
        ceil(log(1 - success_probability) / log(1 - 0.7 * 0.7)));

    // Initially, mark all points as inliers.
    inlier_markers.clear();
    inlier_markers.resize(pts1.size(), 1);

    // Undistort all the points.
    std::vector<cv::Point2f> pts1_undistorted(pts1.size());
    std::vector<cv::Point2f> pts2_undistorted(pts2.size());
    undistortPoints(pts1, intrinsics, distortion_coeffs, pts1_undistorted);
    undistortPoints(pts2, intrinsics, distortion_coeffs, pts2_undistorted);

    // Compenstate the points in the previous image with
    // the relative rotation.
    for (auto &pt : pts1_undistorted)
    {
        Vec3f pt_h(pt.x, pt.y, 1.0f);
        //Vec3f pt_hc = dR * pt_h;
        Vec3f pt_hc = R_curr_from_prev * pt_h;
        pt.x = pt_hc[0];
        pt.y = pt_hc[1];
    }

    // Normalize the points to gain numerical stability.
    float scaling_factor = 0.0f;
    rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
    norm_pixel_unit *= scaling_factor;

    // Compute the difference between previous and current points,
    // which will be used frequently later.
    vector<Point2d> pts_diff(pts1_undistorted.size());
    for (int i = 0; i < pts1_undistorted.size(); ++i)
    {
        pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];
    }

    // Mark the point pairs with large difference directly.
    // BTW, the mean distance of the rest of the point pairs
    // are computed.
    double mean_pt_distance = 0.0;
    int raw_inlier_cntr = 0;
    for (int i = 0; i < pts_diff.size(); ++i)
    {
        double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
        // 25 pixel distance is a pretty large tolerance for normal motion.
        // However, to be used with aggressive motion, this tolerance should
        // be increased significantly to match the usage.
        if (distance > 50.0 * norm_pixel_unit)
        {
            inlier_markers[i] = 0;
        }
        else
        {
            mean_pt_distance += distance;
            ++raw_inlier_cntr;
        }
    }
    mean_pt_distance /= raw_inlier_cntr;

    // If the current number of inliers is less than 3, just mark
    // all input as outliers. This case can happen with fast
    // rotation where very few features are tracked.
    if (raw_inlier_cntr < 3)
    {
        for (auto &marker : inlier_markers)
        {
            marker = 0;
        }

        return;
    }

    // Before doing 2-point RANSAC, we have to check if the motion
    // is degenerated, meaning that there is no translation between
    // the frames, in which case, the model of the RANSAC does not
    // work. If so, the distance between the matched points will
    // be almost 0.
    //if (mean_pt_distance < inlier_error*norm_pixel_unit) {
    if (mean_pt_distance < norm_pixel_unit)
    {
        //ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
        for (int i = 0; i < pts_diff.size(); ++i)
        {
            if (inlier_markers[i] == 0)
            {
                continue;
            }
            if (sqrt(pts_diff[i].dot(pts_diff[i])) >
                inlier_error * norm_pixel_unit)
                inlier_markers[i] = 0;
        }
        return;
    }

    // In the case of general motion, the RANSAC model can be applied.
    // The three column corresponds to tx, ty, and tz respectively.
    MatrixXd coeff_t(pts_diff.size(), 3);
    for (int i = 0; i < pts_diff.size(); ++i)
    {
        coeff_t(i, 0) = pts_diff[i].y;
        coeff_t(i, 1) = -pts_diff[i].x;
        coeff_t(i, 2) = pts1_undistorted[i].x * pts2_undistorted[i].y -
                        pts1_undistorted[i].y * pts2_undistorted[i].x;
    }

    vector<int> raw_inlier_idx;
    for (int i = 0; i < inlier_markers.size(); ++i)
    {
        if (inlier_markers[i] != 0)
            raw_inlier_idx.push_back(i);
    }

    vector<int> best_inlier_set;
    double best_error = 1e10;
    // random_numbers::RandomNumberGenerator random_gen;
    srand(1);

    for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx)
    {
        // Randomly select two point pairs.
        // Although this is a weird way of selecting two pairs, but it
        // is able to efficiently avoid selecting repetitive pairs.
        int select_idx1 = rand() % (raw_inlier_idx.size() - 1);
        int select_idx_diff = rand() % (raw_inlier_idx.size() - 2) + 1;
        int select_idx2 = select_idx1 + select_idx_diff < raw_inlier_idx.size() ? select_idx1 + select_idx_diff : select_idx1 + select_idx_diff - raw_inlier_idx.size();

        int pair_idx1 = raw_inlier_idx[select_idx1];
        int pair_idx2 = raw_inlier_idx[select_idx2];

        // Construct the model;
        Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
        Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
        Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
        vector<double> coeff_l1_norm(3);
        coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
        coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
        coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
        int base_indicator = min_element(coeff_l1_norm.begin(),
                                         coeff_l1_norm.end()) -
                             coeff_l1_norm.begin();

        Vector3d model(0.0, 0.0, 0.0);
        if (base_indicator == 0)
        {
            Matrix2d A;
            A << coeff_ty, coeff_tz;
            Vector2d solution = A.inverse() * (-coeff_tx);
            model(0) = 1.0;
            model(1) = solution(0);
            model(2) = solution(1);
        }
        else if (base_indicator == 1)
        {
            Matrix2d A;
            A << coeff_tx, coeff_tz;
            Vector2d solution = A.inverse() * (-coeff_ty);
            model(0) = solution(0);
            model(1) = 1.0;
            model(2) = solution(1);
        }
        else
        {
            Matrix2d A;
            A << coeff_tx, coeff_ty;
            Vector2d solution = A.inverse() * (-coeff_tz);
            model(0) = solution(0);
            model(1) = solution(1);
            model(2) = 1.0;
        }

        // Find all the inliers among point pairs.
        VectorXd error = coeff_t * model;

        vector<int> inlier_set;
        for (int i = 0; i < error.rows(); ++i)
        {
            if (inlier_markers[i] == 0)
                continue;
            if (std::abs(error(i)) < inlier_error * norm_pixel_unit)
                inlier_set.push_back(i);
        }

        // If the number of inliers is small, the current
        // model is probably wrong.
        if (inlier_set.size() < 0.2 * pts1_undistorted.size())
        {
            continue;
        }

        // Refit the model using all of the possible inliers.
        VectorXd coeff_tx_better(inlier_set.size());
        VectorXd coeff_ty_better(inlier_set.size());
        VectorXd coeff_tz_better(inlier_set.size());
        for (int i = 0; i < inlier_set.size(); ++i)
        {
            coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
            coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
            coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
        }

        Vector3d model_better(0.0, 0.0, 0.0);
        if (base_indicator == 0)
        {
            MatrixXd A(inlier_set.size(), 2);
            A << coeff_ty_better, coeff_tz_better;
            Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
            model_better(0) = 1.0;
            model_better(1) = solution(0);
            model_better(2) = solution(1);
        }
        else if (base_indicator == 1)
        {
            MatrixXd A(inlier_set.size(), 2);
            A << coeff_tx_better, coeff_tz_better;
            Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
            model_better(0) = solution(0);
            model_better(1) = 1.0;
            model_better(2) = solution(1);
        }
        else
        {
            MatrixXd A(inlier_set.size(), 2);
            A << coeff_tx_better, coeff_ty_better;
            Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
            model_better(0) = solution(0);
            model_better(1) = solution(1);
            model_better(2) = 1.0;
        }

        // Compute the error and upate the best model if possible.
        VectorXd new_error = coeff_t * model_better;

        double this_error = 0.0;
        for (const auto &inlier_idx : inlier_set)
            this_error += std::abs(new_error(inlier_idx));
        this_error /= inlier_set.size();

        if (inlier_set.size() > best_inlier_set.size())
        {
            best_error = this_error;
            best_inlier_set = inlier_set;
        }
    }

    // Fill in the markers.
    inlier_markers.clear();
    inlier_markers.resize(pts1.size(), 0);
    for (const auto &inlier_idx : best_inlier_set)
    {
        inlier_markers[inlier_idx] = 1;
    }

    //printf("inlier ratio: %lu/%lu\n",
    //    best_inlier_set.size(), inlier_markers.size());

    return;
}

void ImageProcessor::featureUpdateCallback(Feature_measure_t &curr_features)
{
    // Publish features.
    std::vector<FeatureIDType> curr_ids(0);
    std::vector<cv::Point2f> curr_cam0_points(0);

    for (const auto &grid_features : (*curr_features_ptr))
    {
        for (const auto &feature : grid_features.second)
        {
            curr_ids.push_back(feature.id);
            curr_cam0_points.push_back(feature.cam0_point);
        }
    }

    std::vector<cv::Point2f> curr_cam0_points_undistorted(0);

    undistortPoints(curr_cam0_points, cam0_intrinsics,
                    cam0_distortion_coeffs, curr_cam0_points_undistorted);

    Feature_measure_t feature_msg;
    feature_msg.features.resize(curr_ids.size());

    feature_msg.stamp = curr_cam_timestamp;
    // cout << "curr_cam_timestamp : " << curr_cam_timestamp << endl;
    // cout << "curr_ids.size() : " << curr_ids.size() << endl;
    for (uint32_t i = 0; i < curr_ids.size(); ++i)
    {
        feature_msg.features[i].id = (uint32_t)curr_ids[i];
        feature_msg.features[i].u0 = curr_cam0_points_undistorted[i].x;
        feature_msg.features[i].v0 = curr_cam0_points_undistorted[i].y;
        // cout << "u0 : " << curr_cam0_points_undistorted[i].x << endl;
        // cout << "v0 : " << curr_cam0_points_undistorted[i].y << endl;
    }
    curr_features = feature_msg;
}

void ImageProcessor::drawFeaturesMono()
{
    // Colors for different features.
    Scalar tracked(0, 255, 0);
    Scalar new_feature(0, 255, 255);

    static int grid_height = curr_cam0_img.rows / processor_config.grid_row;
    static int grid_width = curr_cam0_img.cols / processor_config.grid_col;

    // Create an output image.
    int img_height = curr_cam0_img.rows;
    int img_width = curr_cam0_img.cols;
    Mat out_img(img_height, img_width, CV_8UC3);
    cvtColor(curr_cam0_img, out_img, CV_GRAY2RGB);

    // Draw grids on the image.
    for (int i = 1; i < processor_config.grid_row; ++i)
    {
        Point pt1(0, i * grid_height);
        Point pt2(img_width, i * grid_height);
        line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i)
    {
        Point pt1(i * grid_width, 0);
        Point pt2(i * grid_width, img_height);
        line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    vector<FeatureIDType> prev_ids(0);
    for (const auto &grid_features : *prev_features_ptr)
        for (const auto &feature : grid_features.second)
            prev_ids.push_back(feature.id);

    // Collect feature points in the previous frame.
    map<FeatureIDType, Point2f> prev_cam0_points;
    for (const auto &grid_features : *prev_features_ptr)
        for (const auto &feature : grid_features.second)
            prev_cam0_points[feature.id] = feature.cam0_point;

    // Collect feature points in the current frame.
    map<FeatureIDType, Point2f> curr_cam0_points;
    for (const auto &grid_features : *curr_features_ptr)
        for (const auto &feature : grid_features.second)
            curr_cam0_points[feature.id] = feature.cam0_point;

    // Draw tracked features.
    for (const auto &id : prev_ids)
    {
        if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
            curr_cam0_points.find(id) != curr_cam0_points.end())
        {
            cv::Point2f prev_pt0 = prev_cam0_points[id];
            cv::Point2f curr_pt0 = curr_cam0_points[id];

            circle(out_img, curr_pt0, 5, tracked, -1);
            line(out_img, prev_pt0, curr_pt0, tracked, 1);

            prev_cam0_points.erase(id);
            curr_cam0_points.erase(id);
        }
    }

    // Draw new features.
    for (const auto &new_cam0_point : curr_cam0_points)
    {
        cv::Point2f pt0 = new_cam0_point.second;
        circle(out_img, pt0, 5, new_feature, -1);
    }

#ifdef USE_ROS_IMSHOW
    std_msgs::Header img_header;
    img_header.stamp = ros::Time(curr_cam_timestamp);
    cv_bridge::CvImage debug_image(img_header, "bgr8", out_img);
    debug_stereo_pub.publish(debug_image.toImageMsg());
#endif

#ifndef CROSS_COMPILE
    imshow("Feature", out_img);
    waitKey(5);
#endif

    return;
}

void ImageProcessor::updateFeatureLifetime()
{
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col;
         ++code)
    {
        vector<FeatureMetaData> &features = (*curr_features_ptr)[code];
        for (const auto &feature : features)
        {
            if (feature_lifetime.find(feature.id) == feature_lifetime.end())
            {
                feature_lifetime[feature.id] = 1;
            }
            else
            {
                ++feature_lifetime[feature.id];
            }
        }
    }

    return;
}

void ImageProcessor::featureLifetimeStatistics()
{

    map<int, int> lifetime_statistics;
    for (const auto &data : feature_lifetime)
    {
        if (lifetime_statistics.find(data.second) ==
            lifetime_statistics.end())
        {
            lifetime_statistics[data.second] = 1;
        }
        else
        {
            ++lifetime_statistics[data.second];
        }
    }

    for (const auto &data : lifetime_statistics)
    {
        cout << data.first << " : " << data.second << endl;
    }

    return;
}

} // end namespace msckf_vio
