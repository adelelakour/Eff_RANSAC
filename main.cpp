#include "AndreiUtils/utilsBinarySerialization.hpp"
#include "Model_PreProsessing.h"
#include "Database.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <ctime>
#include <cstdlib>


#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/octree/octree_search.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/io.h>
#include <pcl/common/random.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/point_types.h>


#include "example.hpp"
#include <librealsense2/rs.hpp>
#include <pcl/filters/random_sample.h>


// Parameters to be tuned
// Model (offline) parameters
const int M = 45000;           // I don't know how to select this number
const float K = 0.1;           // suggested by prof.Burschka
const float C = 0.25;          // suggested by prof.Burschka
float OffLineRadius {0.001};  // I need to tune this later (0.0025, 0.005)

float tolerance {OffLineRadius*10/100}; //10% of radius

// Online
const float Ps = 0.9;                    // probability of success
const float resolution = 0.0037;         // I chose this number to keep 50% of the original scene


using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
pcl_ptr points_to_pcl(const rs2::points& points);

void register_glfw_callbacks(window& app, glfw_state& app_state);
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);






int main() {


    OuterMap generatedMap = DB::create_hashMap(OffLineRadius,tolerance, "../YCB_ply/Selected_two");
    DB::to_serialize_hashMap(generatedMap, "YCB_hashed.bin");
    std::string filename = "YCB_hashed.bin";
    OuterMap deserialized_map = DB::to_deserialize_hashMap(filename);




    pcl::PointCloud<pcl::PointXYZ>::Ptr S_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPLYFile<pcl::PointXYZ> ("Apple_in_Scene.ply", *S_cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    int S_size = S_cloud->size();




        // from S to S*
       // float resolution = 0.004f;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(S_cloud);
    octree.addPointsFromInputCloud();


    pcl::PointCloud<pcl::PointXYZ>::Ptr S_star_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::AlignedPointTVector voxel_centers;
    octree.getOccupiedVoxelCenters(voxel_centers);

    S_star_cloud->points.resize(voxel_centers.size());
    std::copy(voxel_centers.begin(), voxel_centers.end(), S_star_cloud->points.begin());

    std::cout << "num of points before downsampling " << S_cloud->size() << std::endl;
    std::cout << "num of points after downsampling " << S_star_cloud->size() << std::endl;

    int n = S_star_cloud->size();
    int N = (-n * log(1 - Ps)) / (M * K * C);    // number of iterations needed


    // .................. Normal Estimation .......................

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (S_star_cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr S_star_normals (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch (0.6);
    ne.compute (*S_star_normals);

    std::cout << "S* before Normal estimate : " << S_star_cloud->size()<<std::endl;
    std::cout << "S* after  Normal estimate : " << S_star_normals->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr S_start_full (new pcl::PointCloud<pcl::PointXYZLNormal>);
    pcl::concatenateFields (*S_star_cloud, *S_star_normals, *S_start_full);


    pcl::io::savePLYFile("output_cloud.ply", *S_start_full);


    /*  for (auto N : *S_start_full)
      {
          std::cout << N.x << " " << N.y << " " << N.z << " "
                  << N.normal_x << " " << N.normal_y << " " << N.normal_z << " " << std::endl;
      }*/



    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr Point_A_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr Point_B_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    pcl::PointXYZLNormal PointA;
    pcl::PointXYZLNormal PointB;
    std::vector<int> point_indices;
    std::vector<float> point_distances;


    for (int i = 0; i < N ; ++i) {


        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<int> sampledIndices;
        std::uniform_int_distribution<size_t> dist(0, S_start_full->size() - 1);
        int randomIndex = dist(gen);
        sampledIndices.push_back(randomIndex);
        PointA = S_start_full->points[randomIndex];
        Point_A_cloud->push_back(PointA);


        pcl::KdTreeFLANN<pcl::PointXYZLNormal> kdtree_of_Model_Cloud;
        kdtree_of_Model_Cloud.setInputCloud(S_start_full);
        kdtree_of_Model_Cloud.radiusSearch(PointA, OffLineRadius, point_indices, point_distances);
        for (auto Ind : point_indices) {
            Point_B_cloud->points.push_back(S_start_full->points[Ind]);
        }

        PointB = Point_B_cloud->back();

        if (Point_B_cloud->size() > 1)
        {
            if (isnan(PointA.normal_x) || isnan(PointA.normal_y) || isnan(PointA.normal_z)
            || isnan(PointB.normal_x) || isnan(PointB.normal_x) || isnan(PointA.normal_x))
            {
                continue;
            }


            std::cout << "Point A: " << PointA.x << " " << PointA.y << " " << PointA.z << " " <<
            PointA.normal_x << " " << PointA.normal_y << " " << PointA.normal_z << " " << std::endl;

            std::cout << "Point B: " << PointB.x << " " << PointB.y << " " << PointB.z << " " <<
                      PointB.normal_x << " " << PointB.normal_y << " " << PointB.normal_z << " " << std::endl;


            Eigen::Vector3f Pair_Descriptor = Compute_the_descriptor(PointA, PointB);
            std::cout << Pair_Descriptor[0] << " " << Pair_Descriptor[1] << " " << Pair_Descriptor[2] << std::endl;

        }




        Point_A_cloud->clear();
        Point_B_cloud->clear();
        point_indices.clear();
        point_distances.clear();


    }

    return 0;
}







pcl_ptr points_to_pcl(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}