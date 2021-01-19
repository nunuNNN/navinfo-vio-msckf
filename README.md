运行方式：
$ mkdir catkin_vio_msckf && cd catkin_vio_msckf
$ mkdir src && cd src
$ git clone ssh://git@172.16.1.123:2222/diffusion/PVVIOMSCKF/pv-vision-vio-msckf.git
$ 修改/test/ros_ap03.cpp中的数据路径
$ cd ../..
$ catkin_make -j4 && source ./devel/setup.bash && roslaunch msckf_vio ap03.launch

注意：

1. 跑EuRoC需要CMakeLists中打开ADD_DEFINITIONS(-DUSE_EUROC)
