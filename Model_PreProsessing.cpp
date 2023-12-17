#include "Model_PreProsessing.h"

#include <iostream>

#include <Eigen/Core>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>
#include <list>
#include <tuple>
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
#include <pcl/segmentation/segment_differences.h>
#include <pcl/registration/correspondence_estimation.h>
#include <AndreiUtils/utilsString.h>

using namespace AndreiUtils;
using namespace Eigen;
using namespace std;
namespace fs = std::filesystem;

Eigen::Vector3f fromStringToVectorF(std::string const &vectorAsString) {
    auto res = AndreiUtils::splitString(vectorAsString, ";");
    assert(res.size() == 3);
    return {std::stof(res[0]), std::stof(res[1]), std::stof(res[2])};
}

std::vector<int> stringToVector(const std::string& str) {
    std::vector<int> vec;
    std::stringstream ss(str);
    int temp;
    while (ss >> temp) {
        vec.push_back(temp);
    }
    return vec;
}



std::string fromVectorToString(Eigen::Vector3d const &vector) {
    return std::to_string(vector.x()) + ";" + std::to_string(vector.y()) + ";" + std::to_string(vector.z());
}


Eigen::Vector3d fromStringToVectorD(std::string const &vectorAsString) {
    auto res = AndreiUtils::splitString(vectorAsString, ";");
    assert(res.size() == 3);
    return {std::stod(res[0]), std::stod(res[1]), std::stod(res[2])};
}


std::string fromVectorToString(Eigen::Vector3f const &vector) {
    return std::to_string(vector.x()) + ";" + std::to_string(vector.y()) + ";" + std::to_string(vector.z());
}

Eigen::Vector3f Hash_key = {0, 0, 0};
Eigen::Vector3f An;
Eigen::Vector3f Bn;
Eigen::Vector3f A_B;
Eigen::Vector3f B_A;
double dotProduct;
double magnitudeA;
double magnitudeB;
double angleRadians;
double angleDegrees;


double Angle_Two_Vectors(Eigen::Vector3f Vector_A, Eigen::Vector3f Vector_B) {

    dotProduct = Vector_A.dot(Vector_B);
    magnitudeA = Vector_A.norm();
    magnitudeB = Vector_B.norm();

    // Compute the angle in radians
    angleRadians = std::acos(dotProduct / (magnitudeA * magnitudeB));

    // Convert the angle to degrees
    angleDegrees = angleRadians * 180.0 / M_PI;

    return angleDegrees;
}

inline double Euclidean_Distance_two_Vectors(Eigen::Vector3f A, Eigen::Vector3f B) {
    double Distance = std::sqrt(
            std::pow(B.x() - A.x(), 2) +
            std::pow(B.y() - A.y(), 2) +
            std::pow(B.z() - A.z(), 2)
    );
    return Distance;
}

inline Eigen::Vector3f U_sub_V(Eigen::Vector3f A, Eigen::Vector3f B) {
    Eigen::Vector3f U_sub_V;
    A_B.x() = A.x() - B.x();
    A_B.y() = A.y() - B.y();
    A_B.z() = A.z() - B.z();
    return U_sub_V;
}

inline Eigen::Vector3f V_sub_U(Eigen::Vector3f A, Eigen::Vector3f B) {
    Eigen::Vector3f V_sub_U;
    B_A.x() = B.x() - A.x();
    B_A.y() = B.y() - A.y();
    B_A.z() = B.z() - A.z();
    return V_sub_U;
}



Eigen::Vector3f Compute_the_descriptor (pcl::PointXYZLNormal U, pcl::PointXYZLNormal V)
{
    Eigen::Vector3f Descriptor;
    Eigen::Vector3f P_U = {U.x, U.y, U.z};
    Eigen::Vector3f P_V = {V.x, V.y, V.z};

    Eigen::Vector3f N_U = {U.normal_x, U.normal_y, U.normal_z};
    Eigen::Vector3f N_V = {V.normal_x, V.normal_y, V.normal_z};

    Descriptor[0] = Angle_Two_Vectors(N_U, N_V);
    Descriptor[1] = Angle_Two_Vectors (N_U, V_sub_U(P_V, P_U));
    Descriptor[2] = Angle_Two_Vectors (N_V, U_sub_V(P_U, P_V));

    return Descriptor;

}




OuterMap Compute_HashTable(float radius, double pointSphereRelativeTolerance, std::string path_to_models) {
    OuterMap myHashTable;       // Cell is a tuple of (pair) and (model Name)
    myHashTable.clear();

    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr Model_Cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);


    pcl::PointXYZLNormal U_XYZNorm;
    pcl::PointXYZLNormal V_XYZNorm;
    Eigen::Vector3f U_n = Eigen::Vector3f::Zero();
    Eigen::Vector3f V_n = Eigen::Vector3f::Zero();
    Eigen::Vector3f U_p = Eigen::Vector3f::Zero();
    Eigen::Vector3f V_p = Eigen::Vector3f::Zero();

    One_PAIR Detected_pairs;
    double pointDistance;
    InnerMap OneCell;

    std::vector<int> point_indices;
    std::vector<float> point_distances;


    std::string directory_path = path_to_models;
    std::string extension = ".ply";
    std::string Model_Name;
    std::string file_path;


    for (const auto &entry: fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            file_path = entry.path().string();
            Model_Name = entry.path().filename().stem();
        }

        pcl::io::loadPLYFile(file_path, *Model_Cloud);

        std::cout << "size before voxelgrid : " << Model_Cloud->size() << std::endl;

        pcl::VoxelGrid<pcl::PointXYZLNormal> sor;
        sor.setInputCloud(Model_Cloud);
        sor.setLeafSize(0.001f, 0.001f, 0.001f);
        sor.filter(*Model_Cloud);
        std::cout << "size after voxelgrid  : " << Model_Cloud->size() << std::endl;

        std::cout << "A new Model is loaded" << std::endl;

        pcl::KdTreeFLANN<pcl::PointXYZLNormal> kdtree_of_Model_Cloud;
        kdtree_of_Model_Cloud.setInputCloud(Model_Cloud);

        for (size_t i = 0; i < Model_Cloud->size(); ++i) {
            U_XYZNorm = Model_Cloud->at(i);

            U_p = Eigen::Vector3f(U_XYZNorm.x, U_XYZNorm.y, U_XYZNorm.z);
            U_n = Eigen::Vector3f(U_XYZNorm.normal_x, U_XYZNorm.normal_y, U_XYZNorm.normal_z);

            kdtree_of_Model_Cloud.radiusSearch(U_XYZNorm, radius, point_indices, point_distances);
            // std::cout << "number of points within this sphere is : " << point_indices.size() << std::endl;

            for (int point_index : point_indices) {
                V_XYZNorm = Model_Cloud->at(point_index);

                pointDistance = euclideanDistance(U_XYZNorm, V_XYZNorm);
                // Tolerance for considering a point on the perimeter
                if (pointDistance < radius * (1 + pointSphereRelativeTolerance) &&
                    pointDistance > radius * (1 - pointSphereRelativeTolerance)) {
                    // std::cout << "I found a Pair in " << Model_Name << std::endl;
                    V_p = Eigen::Vector3f(V_XYZNorm.x,
                                          V_XYZNorm.y,
                                          V_XYZNorm.z);

                    V_n = Eigen::Vector3f(V_XYZNorm.normal_x,
                                          V_XYZNorm.normal_y,
                                          V_XYZNorm.normal_z);

                    Hash_key[0] = Angle_Two_Vectors(U_n, V_n);

                    auto Pu_Pv = U_sub_V(U_p, V_p);
                    auto Pv_Pu = V_sub_U(U_p, V_p);

                    Hash_key[1] = Angle_Two_Vectors(U_n, Pv_Pu);
                    Hash_key[2] = Angle_Two_Vectors(V_n, Pu_Pv);

                    Detected_pairs.first = U_XYZNorm;
                    Detected_pairs.second = V_XYZNorm;

                    string hashKeyString = fromVectorToString(Hash_key);
                    auto findRequest = myHashTable.find(hashKeyString);
                    if (findRequest != myHashTable.end()) {    // it exists
                        findRequest->second[Model_Name].push_back(Detected_pairs);
                        // std::cout << "A new entry is added to existing key ++++++++++++++++++++++++++++:" << std::endl;
                    } else {
                        InnerMap newData;
                        newData[Model_Name].push_back(Detected_pairs);
                        myHashTable[hashKeyString] = newData;
                        // std::cout << "A new key is added : " << Hash_key.transpose() << std::endl;
                    }

                    //break;
                }
            }
        }
    }

    std::cout << "Size of myHashTable is: " << myHashTable.size() << std::endl;


/*    for (auto const &entity : myHashTable) {
        auto Cell = entity.second;
        if (std::get<0>(Cell).size() > 2) {
            std::cout << "I found big cell" << std::endl;
        }
    }*/
    std::cout << "NNNNNNN" << std::endl;


    return myHashTable;
}



