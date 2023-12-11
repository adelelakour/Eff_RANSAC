#include "Database.h"
#include "Model_PreProsessing.h"
#include "AndreiUtils/utilsBinarySerialization.hpp"
#include "SeializeDeserialize.h"


void Serialize_hashMAP (OuterMap hashMap_to_serialize, std::string FileName );
OuterMap Deserialize_hashMAP (std::string FileName);


OuterMap DB::create_hashMap(float radius, double pointSphereRelativeTolerance, std::string path_to_models) {
    return Compute_HashTable(radius, pointSphereRelativeTolerance, path_to_models);
}


void DB::to_serialize_hashMap(OuterMap hashMapp, std::string FileName) {
    Serialize_hashMAP(hashMapp, FileName);
}


OuterMap DB::to_deserialize_hashMap(std::string FileName) {
    OuterMap d_map = Deserialize_hashMAP(FileName);
    return d_map;
}


