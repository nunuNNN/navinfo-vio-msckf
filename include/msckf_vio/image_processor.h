#ifndef MSCKF_VIO_IMAGE_PROCESSOR_H
#define MSCKF_VIO_IMAGE_PROCESSOR_H

#include <vector>
#include <queue>
#include <map>
#include <memory>
#include <mutex>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "data_interface.h"

#ifdef USE_ROS_IMSHOW
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#endif

namespace msckf_vio
{
/*
 * @brief ImageProcessor Detects and tracks features
 *    in image sequences.
 */
class ImageProcessor
{
public:
    // Constructor
    ImageProcessor();
    // Disable copy and assign constructors.
    ImageProcessor(const ImageProcessor &) = delete;
    ImageProcessor operator=(const ImageProcessor &) = delete;

    // Destructor
    ~ImageProcessor();

    /*
   * @brief monoCallback Callback function for the mono images.
   * @param cam0_img left image.
   */
    void monoCallback(double timestamp, 
                    const cv::Mat &cam0_img, 
                    std::vector<StrImuData> &curr_from_last_imu);

    /*
     * @brief featureUpdateCallback
     *    Publish the features on the current image including
     *    both the tracked and newly detected ones.
    */
    void featureUpdateCallback(Feature_measure_t &curr_features);

    typedef std::shared_ptr<ImageProcessor> Ptr;
    typedef std::shared_ptr<const ImageProcessor> ConstPtr;

private:

#ifdef USE_ROS_IMSHOW
    image_transport::Publisher debug_stereo_pub;
#endif
    cv::Mat curr_cam0_img;

    double curr_cam_timestamp;
    double last_cam_timestamp;

    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    std::vector<StrImuData> imu_curr_from_last;


    /*
   * @brief FeatureIDType An alias for unsigned long long int.
   */
    typedef unsigned long long int FeatureIDType;

    /*
   * @brief FeatureMetaData Contains necessary information
   *    of a feature for easy access.
   */
    struct FeatureMetaData
    {
        FeatureIDType id;
        float response;
        int lifetime;
        cv::Point2f cam0_point;
    };

    /*
   * @brief GridFeatures Organize features based on the grid
   *    they belong to. Note that the key is encoded by the
   *    grid index.
   */
    typedef std::map<int, std::vector<FeatureMetaData>> GridFeatures;

    // Indicate if this is the first image message.
    bool is_first_img;


    // ID for the next new feature.
    FeatureIDType next_feature_id;

    // Feature detector
#ifndef USE_OPENCV3
    cv::FastFeatureDetector detector;
#else
    cv::Ptr<cv::Feature2D> detector_ptr;
#endif

    // Pyramids for previous and current image
    std::vector<cv::Mat> prev_cam0_pyramid_;
    std::vector<cv::Mat> curr_cam0_pyramid_;

    // Features in the previous and current image.
    std::shared_ptr<GridFeatures> prev_features_ptr;
    std::shared_ptr<GridFeatures> curr_features_ptr;

    // Number of features after each outlier removal step.
    int before_tracking;
    int after_tracking;
    int after_matching;
    int after_ransac;

    // Debugging
    std::map<FeatureIDType, int> feature_lifetime;

    // Camera calibration parameters
    cv::Vec4d cam0_intrinsics;
    cv::Vec4d cam0_distortion_coeffs;

    // Take a vector from cam0 frame to the IMU frame.
    cv::Matx33d R_imu_from_cam0;
    cv::Vec3d t_imu_from_cam0;

    int camera_id;

private:
    // Initialize the object.
    bool initialize();

    /*
   * @brief keyPointCompareByResponse
   *    Compare two keypoints based on the response.
   */
    static bool keyPointCompareByResponse(
        const cv::KeyPoint &pt1,
        const cv::KeyPoint &pt2)
    {
        // Keypoint with higher response will be at the
        // beginning of the vector.
        return pt1.response > pt2.response;
    }
    /*
   * @brief featureCompareByResponse
   *    Compare two features based on the response.
   */
    static bool featureCompareByResponse(
        const FeatureMetaData &f1,
        const FeatureMetaData &f2)
    {
        // Features with higher response will be at the
        // beginning of the vector.
        return f1.response > f2.response;
    }
    /*
   * @brief featureCompareByLifetime
   *    Compare two features based on the lifetime.
   */
    static bool featureCompareByLifetime(
        const FeatureMetaData &f1,
        const FeatureMetaData &f2)
    {
        // Features with longer lifetime will be at the
        // beginning of the vector.
        return f1.lifetime > f2.lifetime;
    }

    /*
   * @initializeFirstFrame
   *    Initialize the image processing sequence, which is
   *    bascially detect new features on the first set of
   *    stereo images.
   */
    void initializeFirstFrame();

    /*
   * @brief trackFeatures
   *    Tracker features on the newly received stereo images.
   */
    void trackFeatures();

    /*
   * @addNewFeatures
   *    Detect new features on the image to ensure that the
   *    features are uniformly distributed on the image.
   */
    void addNewFeatures();

    /*
   * @brief pruneGridFeatures
   *    Remove some of the features of a grid in case there are
   *    too many features inside of that grid, which ensures the
   *    number of features within each grid is bounded.
   */
    void pruneGridFeatures();

    /*
   * @brief drawFeaturesMono
   *    Draw tracked and newly detected features on the left
   *    image only.
   */
    void drawFeaturesMono();

    /*
   * @brief createImagePyramids
   *    Create image pyramids used for klt tracking.
   */
    void createImagePyramids();

  /*
   * @brief integrateImuData Integrates the IMU gyro readings
   *    between the two consecutive images, which is used for
   *    both tracking prediction and 2-point RANSAC.
   * @return cam0_R_p_c: a rotation matrix which takes a vector
   *    from previous cam0 frame to current cam0 frame.
   */
  void integrateImuData(cv::Matx33f& cam0_R_p_c);

    /*
   * @brief predictFeatureTracking Compensates the rotation
   *    between consecutive camera frames so that feature
   *    tracking would be more robust and fast.
   * @param input_pts: features in the previous image to be tracked.
   * @param R_p_c: a rotation matrix takes a vector in the previous
   *    camera frame to the current camera frame.
   * @param intrinsics: intrinsic matrix of the camera.
   * @return compensated_pts: predicted locations of the features
   *    in the current image based on the provided rotation.
   *
   * Note that the input and output points are of pixel coordinates.
   */
    void predictFeatureTracking(
        const std::vector<cv::Point2f> &input_pts,
        const cv::Matx33f &R_p_c,
        const cv::Vec4d &intrinsics,
        std::vector<cv::Point2f> &compenstated_pts);

    /*
   * @brief twoPointRansac Applies two point ransac algorithm
   *    to mark the inliers in the input set.
   * @param pts1: first set of points.
   * @param pts2: second set of points.
   * @param R_p_c: a rotation matrix takes a vector in the previous
   *    camera frame to the current camera frame.
   * @param intrinsics: intrinsics of the camera.
   * @param distortion_model: distortion model of the camera.
   * @param distortion_coeffs: distortion coefficients.
   * @param inlier_error: acceptable error to be considered as an inlier.
   * @param success_probability: the required probability of success.
   * @return inlier_flag: 1 for inliers and 0 for outliers.
   */
    void twoPointRansac(
        const std::vector<cv::Point2f> &pts1,
        const std::vector<cv::Point2f> &pts2,
        const cv::Matx33f &R_p_c,
        const cv::Vec4d &intrinsics,
        const cv::Vec4d &distortion_coeffs,
        const double &inlier_error,
        const double &success_probability,
        std::vector<int> &inlier_markers);

    void rescalePoints(
        std::vector<cv::Point2f> &pts1,
        std::vector<cv::Point2f> &pts2,
        float &scaling_factor);

    /*
   * @brief removeUnmarkedElements Remove the unmarked elements
   *    within a vector.
   * @param raw_vec: vector with outliers.
   * @param markers: 0 will represent a outlier, 1 will be an inlier.
   * @return refined_vec: a vector without outliers.
   *
   * Note that the order of the inliers in the raw_vec is perserved
   * in the refined_vec.
   */
    template <typename T>
    void removeUnmarkedElements(
        const std::vector<T> &raw_vec,
        const std::vector<unsigned char> &markers,
        std::vector<T> &refined_vec)
    {
        if (raw_vec.size() != markers.size())
        {
            printf("The input size of raw_vec(%lu) and markers(%lu) does not match...",
                   raw_vec.size(), markers.size());
        }
        for (int i = 0; i < markers.size(); ++i)
        {
            if (markers[i] == 0)
                continue;
            refined_vec.push_back(raw_vec[i]);
        }
        return;
    }

    void updateFeatureLifetime();
    void featureLifetimeStatistics();

    void undistortPoints(
        const std::vector<cv::Point2f> &pts_in,
        const cv::Vec4d &intrinsics,
        const cv::Vec4d &distortion_coeffs,
        std::vector<cv::Point2f> &pts_out,
        const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1, 1, 0, 0));

    std::vector<cv::Point2f> distortPoints(
        const std::vector<cv::Point2f> &pts_in,
        const cv::Vec4d &intrinsics,
        const cv::Vec4d &distortion_coeffs);
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

} // end namespace msckf_vio

#endif
