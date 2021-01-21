cmake_minimum_required(VERSION 2.8.12)
project(msckf_vio)

add_compile_options(-std=c++11)

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_BUILD_TYPE Release)

ADD_DEFINITIONS(-DUSE_EUROC)
ADD_DEFINITIONS(-DUSE_DATASET)
ADD_DEFINITIONS(-DUSE_ROS_IMSHOW)
ADD_DEFINITIONS(-DUSING_SPARSE_QR)
ADD_DEFINITIONS(-DUSE_OPENCV3)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  nav_msgs
  sensor_msgs
  geometry_msgs
  eigen_conversions
  tf_conversions
  random_numbers
  message_generation
  nodelet
  image_transport
  cv_bridge
  message_filters
  pcl_conversions
  pcl_ros
  std_srvs
)

## System dependencies are found with CMake's conventions
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.0 REQUIRED)
find_package(SuiteSparse REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)


##################
## ROS messages ##
##################
# add_message_files(
#   FILES

#   FeatureMeasurement.msg
#   CameraMeasurement.msg
#   TrackingInfo.msg
# )

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES msckf_vio image_processor
  CATKIN_DEPENDS
    roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
    eigen_conversions tf_conversions random_numbers message_runtime
    image_transport cv_bridge message_filters pcl_conversions
    pcl_ros std_srvs
  DEPENDS EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########
include_directories(
  include
  include/msckf_vio
  include/Initializer
  include/imu_gps_localizer
  third_party/GeographicLib/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)


# imu gps localization
# 1. Add GeographicLib.
add_subdirectory(third_party/GeographicLib
  GeographicLib.out
)

# Msckf Vio
LIST(APPEND SOURCEFILES
  src/interface_msckf_vio.cpp
  src/vio_manager.cpp
  src/global_param.cpp
  src/image_processor.cpp
  src/msckf_vio.cpp

  # 
  src/imu_gps_localizer/imu_gps_localizer.cpp
  src/imu_gps_localizer/initializer.cpp
  src/imu_gps_localizer/imu_processor.cpp
  src/imu_gps_localizer/gps_processor.cpp
  src/image_processor.cpp
  src/global_param.cpp

  # 
  src/Initializer/static_initializer.cpp 
  src/Initializer/dynamic_initializer.cpp
  src/Initializer/feature_manager.cpp
  src/Initializer/initial_alignment.cpp
  src/Initializer/initial_sfm.cpp
  src/Initializer/solve_5pts.cpp
  src/Initializer/initializer.cpp 
  
  
)

message("OpenCV_LIBS : ${OpenCV_LIBS}")


# Set link libraries
LIST(APPEND LINK_LIBS 
  libGeographiccc 
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${catkin_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
)

add_executable(ros_euroc test/ros_euroc.cpp ${SOURCEFILES}  )
target_link_libraries(ros_euroc ${LINK_LIBS} -lpthread )
add_dependencies(ros_euroc ${catkin_EXPORTED_TARGETS})
add_dependencies(ros_euroc ${PROJECT_NAME}_gencpp)
