cmake_minimum_required(VERSION 2.8.3)

project(onelink_support)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

install(DIRECTORY config launch meshes urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
