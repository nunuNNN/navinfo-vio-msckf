SET (PROJECT_NAME imuSimWithPointLine)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo , Debug

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules/")

ADD_DEFINITIONS(-DUSE_DATASET)
ADD_DEFINITIONS(-DUSING_SPARSE_QR)

find_package(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)

message("Eigen is found: ${Eigen_INCLUDE_DIRS}")
message("OpenCV is found: ${OpenCV_INCLUDE_DIRS}")
message("Sophus is found: ${Sophus_INCLUDE_DIRS}")

include_directories(
${PROJECT_SOURCE_DIR}/../include/msckf_vio
${PROJECT_SOURCE_DIR}/../include 
${EIGEN3_INCLUDE_DIR}
${SOPHUS_INCLUDE_DIR} # for both sophus and geographiclib
${OPENCV_INCLUDE_DIR}
${SUITESPARSE_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}
)

LIST(APPEND LINK_LIBS
${OpenCV_LIBS}
${Sophus_LIBRARIES}
${SUITESPARSE_LIBRARIES}
)
LIST(APPEND SOURCEFILES
    ${PROJECT_SOURCE_DIR}/../src/msckf_vio.cpp 
    ${PROJECT_SOURCE_DIR}/../src/global_param.cpp 
    ${PROJECT_SOURCE_DIR}/../src/sim/param.cpp
    ${PROJECT_SOURCE_DIR}/../src/sim/utilities.cpp
    ${PROJECT_SOURCE_DIR}/../src/sim/imu.cpp
    ${PROJECT_SOURCE_DIR}/../src/sim/param.h
    ${PROJECT_SOURCE_DIR}/../src/sim/utilities.h
    ${PROJECT_SOURCE_DIR}/../src/sim/imu.h
    )


ADD_EXECUTABLE(data_gen 
                ${PROJECT_SOURCE_DIR}/../src/sim/gener_alldata.cpp 
                ${SOURCEFILES} )
TARGET_LINK_LIBRARIES (data_gen ${LINK_LIBS} -lpthread)

