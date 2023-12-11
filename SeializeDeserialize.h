#ifndef ANDREIUTILS_HPP
#define ANDREIUTILS_HPP

#include "AndreiUtils/utilsBinarySerialization.hpp"
#include "Model_PreProsessing.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <pcl/point_types.h>

namespace AndreiUtils {
    void serialize(std::ofstream &OUT, pcl::PointXYZLNormal PCL_point);
    void deserialize(std::ifstream &IN, pcl::PointXYZLNormal &PClPoint);

    void serialize(std::ofstream &out, const std::string &data);
    void deserialize(std::ifstream &in, std::string &data);

    void serialize(std::ofstream &out, const std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal> &pair);
    void deserialize(std::ifstream &in, std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal> &pair);

    void serialize(std::ofstream &out, const std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>> &vec);
    void deserialize(std::ifstream &in, std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>> &vec);

    void serialize(std::ofstream &out, const std::unordered_map<std::string, std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>>> &map);
    void deserialize(std::ifstream &in, std::unordered_map<std::string, std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>>> &map);

    void serialize(std::ofstream &out, const OuterMap &map);
    void deserialize(std::ifstream &in, OuterMap &map);
}

// Function declarations for serialization and deserialization of OuterMap
void Serialize_hashMAP (OuterMap hashMap, std::string FileName);
OuterMap Deserialize_hashMAP (std::string FileName);

#endif // ANDREIUTILS_HPP
