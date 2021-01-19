cmake_minimum_required(VERSION 2.8)
# include(vVersion.cmake)
project(msckf_vio)

set( CMAKE_CXX_FLAGS "-std=c++11" )

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo , Debug
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# ADD_DEFINITIONS(-DUSE_DEBUG_OUTPUT)
ADD_DEFINITIONS(-DUSE_EUROC)
ADD_DEFINITIONS(-DUSE_DATASET)
ADD_DEFINITIONS(-DUSE_OPENCV3)


# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
# SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas -g -O3 -pthread") #-march=native

# IF(DEFINED ENV{ARM_ARCHITECTURE})
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
# ELSE()
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
# ENDIF()

# IF(CMAKE_COMPILER_IS_GNUCC)
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g")
# ELSE()
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
# ENDIF()

# SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# set(OpenCV_DIR "/home/pv/soft/opencv-2.4.9/build/installed/share/OpenCV")
FIND_PACKAGE(OpenCV 3.0 REQUIRED )
FIND_PACKAGE(Eigen REQUIRED)
# 动态初始化时使用的优化库，后端算法层面去除掉
FIND_PACKAGE(Ceres REQUIRED)


message("Eigen is found: ${Eigen_INCLUDE_DIRS}")
message("OpenCV is found: ${OpenCV_INCLUDE_DIRS}")

include_directories( 
    ${PROJECT_SOURCE_DIR}/../include/Initializer
    ${PROJECT_SOURCE_DIR}/../include/msckf_vio
    ${PROJECT_SOURCE_DIR}/../include/thread_run
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../third_party/GeographicLib/include
    ${PROJECT_SOURCE_DIR}/../include/imu_gps_localizer
    ${Eigen_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# imu gps localization
# 1. Add GeographicLib.
add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party/GeographicLib
    GeographicLib.out
)

# 2. imu_gps_localizer
add_library(imu_gps_localizer_lib
    ${PROJECT_SOURCE_DIR}/../src/imu_gps_localizer/imu_gps_localizer.cpp
    ${PROJECT_SOURCE_DIR}/../src/imu_gps_localizer/initializer.cpp
    ${PROJECT_SOURCE_DIR}/../src/imu_gps_localizer/imu_processor.cpp
    ${PROJECT_SOURCE_DIR}/../src/imu_gps_localizer/gps_processor.cpp
)

target_link_libraries(imu_gps_localizer_lib
	${EIGEN3_LIBS}
    libGeographiccc
)

# 1. Image processor
add_library(image_processor
    ${PROJECT_SOURCE_DIR}/../src/image_processor.cpp
    ${PROJECT_SOURCE_DIR}/../src/global_param.cpp
)
target_link_libraries(image_processor
    ${OpenCV_LIBRARIES}
)

# 2. Initializer
# -- Static Initializer
add_library(staticInitializer
    ${PROJECT_SOURCE_DIR}/../src/Initializer/static_initializer.cpp 
)
target_link_libraries(staticInitializer
)
# -- Dynamic Initializer
add_library(dynamicInitializer
    ${PROJECT_SOURCE_DIR}/../src/Initializer/dynamic_initializer.cpp
    ${PROJECT_SOURCE_DIR}/../src/Initializer/feature_manager.cpp
    ${PROJECT_SOURCE_DIR}/../src/Initializer/initial_alignment.cpp
    ${PROJECT_SOURCE_DIR}/../src/Initializer/initial_sfm.cpp
    ${PROJECT_SOURCE_DIR}/../src/Initializer/solve_5pts.cpp
)
target_link_libraries(dynamicInitializer
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
# -- Initializer
add_library(initializer
    ${PROJECT_SOURCE_DIR}/../src/Initializer/initializer.cpp
)
target_link_libraries(initializer
    staticInitializer
    dynamicInitializer
)

# 3. Estimator
add_library(estimator
    ${PROJECT_SOURCE_DIR}/../src/msckf_vio.cpp 
)
target_link_libraries(estimator
    initializer
    # ${SUITESPARSE_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# 4. VIO Manager
add_library(vio_manager
    ${PROJECT_SOURCE_DIR}/../src/interface_msckf_vio.cpp
    ${PROJECT_SOURCE_DIR}/../src/vio_manager.cpp
)
target_link_libraries(vio_manager
    estimator
    image_processor
    imu_gps_localizer_lib
)

# 5. test euroc
add_executable(test_euroc 
    ${PROJECT_SOURCE_DIR}/../test/test_euroc.cpp 
)
target_link_libraries(test_euroc
    vio_manager
    -lpthread
)

