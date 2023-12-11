#include "SeializeDeserialize.h"




namespace AndreiUtils {

    void serialize(std::ofstream &OUT, pcl::PointXYZLNormal PCL_point) {
        serialize(OUT, PCL_point.x);
        serialize(OUT, PCL_point.y);
        serialize(OUT, PCL_point.z);
        serialize(OUT, PCL_point.label);
        serialize(OUT, PCL_point.normal_x);
        serialize(OUT, PCL_point.normal_y);
        serialize(OUT, PCL_point.normal_z);
        serialize(OUT, PCL_point.curvature);

    }


    void deserialize(std::ifstream &IN, pcl::PointXYZLNormal &PClPoint) {
        deserialize(IN, PClPoint.x);
        deserialize(IN, PClPoint.y);
        deserialize(IN, PClPoint.z);
        deserialize(IN, PClPoint.label);
        deserialize(IN, PClPoint.normal_x);
        deserialize(IN, PClPoint.normal_y);
        deserialize(IN, PClPoint.normal_z);
        deserialize(IN, PClPoint.curvature);

    }



    void serialize(std::ofstream &out, const std::string &data) {

        size_t length = data.size();
        serialize(out, length);


        out.write(data.data(), length);
    }



    void deserialize(std::ifstream &in, std::string &data) {

        size_t length;
        deserialize(in, length);


        data.resize(length);
        in.read(data.data(), length);
        if (in.fail()) {
            throw std::runtime_error("Deserializing string data failed!");
        }
    }



    void serialize(std::ofstream &out, const std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal> &pair) {
        serialize(out, pair.first);
        serialize(out, pair.second);
    }


    void deserialize(std::ifstream &in, std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal> &pair) {
        deserialize(in, pair.first);
        deserialize(in, pair.second);
    }


    void serialize(std::ofstream &out, const std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>> &vec) {
        size_t size = vec.size();
        serialize(out, size);
        for (const auto &pair : vec) {
            serialize(out, pair);
        }
    }


    void deserialize(std::ifstream &in, std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>> &vec) {
        size_t size;
        deserialize(in, size);
        vec.clear(); // Clear the vector before populating it
        vec.resize(size); // Resize the vector to the expected size

        for (size_t i = 0; i < size; ++i) {
            deserialize(in, vec[i]);  // Deserialize pair of integers directly into vector slots
        }
    }



    void serialize(std::ofstream &out, const std::unordered_map<std::string, std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>>> &map) {
        size_t size = map.size();
        serialize(out, size);
        for (const auto &pair : map) {
            serialize(out, pair.first);  // Serialize string key
            serialize(out, pair.second); // Serialize vector of pairs of integers
        }
    }



    void deserialize(std::ifstream &in, std::unordered_map<std::string, std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>>> &map) {
        size_t mapSize;
        deserialize(in, mapSize);

        for (size_t i = 0; i < mapSize; ++i) {
            std::string key;
            deserialize(in, key); // Deserialize string key

            size_t vecSize;
            deserialize(in, vecSize); // Deserialize string key


            std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>> value;
            for (size_t j = 0; j < vecSize; ++j) {
                std::pair<int, int> pair;
                deserialize(in, pair); // Deserialize pair of integers
                value.push_back(pair);
            }

            map[key] = value;
            //map.emplace(std::move(key), std::move(value));
        }
    }

    void serialize(std::ofstream &out, const OuterMap &map) {
        size_t outerSize = map.size();
        serialize(out, outerSize);

        for (const auto &outerPair: map) {
            serialize(out, outerPair.first); // Serialize outer string key

            const auto &innerMap = outerPair.second;
            size_t innerSize = innerMap.size();
            serialize(out, innerSize);

            for (const auto &innerPair: innerMap) {
                serialize(out, innerPair.first); // Serialize inner string key
                serialize(out, innerPair.second); // Serialize vector of pairs
            }
        }
    }

    void deserialize(std::ifstream &in, OuterMap & map) {
        size_t outerSize;
        deserialize(in, outerSize);

        for (size_t i = 0; i < outerSize; ++i) {
            std::string outerKey;
            deserialize(in, outerKey); // Deserialize outer string key

            size_t innerSize;
            deserialize(in, innerSize);

            std::unordered_map<std::string, std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>>> innerMap;
            for (size_t j = 0; j < innerSize; ++j) {
                std::string innerKey;
                deserialize(in, innerKey); // Deserialize inner string key

                std::vector<std::pair<pcl::PointXYZLNormal, pcl::PointXYZLNormal>> value;
                deserialize(in, value); // Deserialize vector of pairs

                innerMap[innerKey] = value;
            }

            map[outerKey] = innerMap;
        }
    }


}






void Serialize_hashMAP (OuterMap hashMap, std::string FileName){

    std::ofstream outFile(FileName, std::ios::binary);
    AndreiUtils::serialize(outFile, hashMap);
    outFile.close();
    std::cout << " Your hashMap has been serialize into " << FileName << std::endl;
}


OuterMap Deserialize_hashMAP (std::string FileName) {

    OuterMap myOutMap_RET;
    std::ifstream inFile(FileName, std::ios::binary);
    AndreiUtils::deserialize(inFile, myOutMap_RET);
    inFile.close();

    return myOutMap_RET;
}

