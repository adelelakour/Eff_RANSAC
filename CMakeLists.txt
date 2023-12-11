cmake_minimum_required(VERSION 3.16)
project(Effi_RANSAC_MA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.13 REQUIRED)
find_package(AndreiUtils REQUIRED core)
find_package(realsense2 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)


include_directories(${PCL_INCLUDE_DIRS} ${CMAKE_BINARY_DIR} /include/json/include/nlohmann)
message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")

include_directories(/usr/local/include/AndreiUtils)

link_directories(${PCL_LIBRARY_DIRS} )
add_definitions(${PCL_DEFINITIONS})

add_library(ObjectModel
        Model_PreProsessing.cpp
        )

#add_executable(Effi_RANSAC_MA main.cpp)
add_executable(Cereal_version main.cpp
        SerializeDeserialize.cpp
        Database.cpp
        SeializeDeserialize.h)


target_link_libraries(Cereal_version PUBLIC ObjectModel
        ${AndreiUtils_LIBRARY}
        ${PCL_LIBRARIES}
        realsense2
        glfw
        ${OPENGL_LIBRARIES}
)


