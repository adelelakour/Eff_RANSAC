//
// Created by adelelakour on 08.11.23.
//

#ifndef MY_HEADER_H
#define MY_HEADER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <utility>
#include <vector>
#include <list>
#include<tuple>
#include <string>

#include <filesystem>
#include <pcl/common/distances.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/common/transforms.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/registration/correspondence_estimation.h>


typedef std::vector<Eigen::Matrix4f> RigidTransformation;


typedef pcl::PointXYZLNormal OnePoint;
typedef std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal> One_PAIR;
typedef std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>> Vec_of_Pairs;
typedef std::unordered_map<std::string, Vec_of_Pairs> InnerMap;
typedef std::unordered_map<std::string, InnerMap> OuterMap;



Eigen::Vector3f fromStringToVectorF(std::string const &vectorAsString);
std::vector<int> stringToVector(const std::string& str);
std::string fromVectorToString(Eigen::Vector3d const &vector);
Eigen::Vector3d fromStringToVectorD(std::string const &vectorAsString);
std::string fromVectorToString(Eigen::Vector3f const &vector);



double Angle_Two_Vectors(Eigen::Vector3f Vector_A, Eigen::Vector3f Vector_B);
double Euclidean_Distance_two_Vectors(Eigen::Vector3f A, Eigen::Vector3f B);
Eigen::Vector3f U_sub_V(Eigen::Vector3f A, Eigen::Vector3f B);
Eigen::Vector3f V_sub_U(Eigen::Vector3f A, Eigen::Vector3f B);


Eigen::Vector3f Compute_the_descriptor (pcl::PointXYZLNormal U, pcl::PointXYZLNormal V);
OuterMap Compute_HashTable(float radius, double pointSphereRelativeTolerance, std::string path_to_models);
bool serializeHashMap(const OuterMap& data, const std::string& filename);
#endif // MY_HEADER_H


