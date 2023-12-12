
#ifndef DATABASE_H
#define DATABASE_H

#include <string>
#include "Model_PreProsessing.h"
#include "AndreiUtils/utilsBinarySerialization.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>


class DB {
public:
    static OuterMap create_hashMap(float radius, double pointSphereRelativeTolerance, std::string path_to_models); // my initial value 0.01, 0.001
    static void to_serialize_hashMap(OuterMap hashMap, std::string FileName);
    static OuterMap to_deserialize_hashMap(std::string FileName);
};

#endif // DATABASE_H
