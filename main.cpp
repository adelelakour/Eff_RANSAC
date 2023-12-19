
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/auto_io.h>
#include <pcl/common/time.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/octree/octree_search.h>
#include <pcl/io/ply_io.h>
#include <thread>
#include <pcl/io/auto_io.h>
#include <pcl/common/time.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/octree/octree_search.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h> // For visualization
#include <pcl/visualization/cloud_viewer.h>   // Alternatively, for cloud viewing
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/visualization/cloud_viewer.h>
#include "Model_PreProsessing.h"
#include "Database.h"

#include <thread>
#include <pcl/io/auto_io.h>
#include <pcl/common/time.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/filter.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCubeSource.h>
#include <vtkCleanPolyData.h>
#include <chrono>

#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <chrono>

#include <thread>

#include <pcl/io/auto_io.h>
#include <pcl/common/time.h>

#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/common/centroid.h>

#include <pcl/filters/filter.h>

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCubeSource.h>
#include <vtkCleanPolyData.h>
#include <cmath>

using namespace std::chrono_literals;



class OctreeViewer
{
public:
    OctreeViewer (std::string &filename, double resolution) :
            viz ("Octree visualizator"),
            cloud (new pcl::PointCloud<pcl::PointXYZLNormal>()),
            displayCloud (new pcl::PointCloud<pcl::PointXYZLNormal>()),
            cloudVoxel (new pcl::PointCloud<pcl::PointXYZLNormal>()),
            octree (resolution)
    {

        //try to load the cloud
        if (!loadCloud(filename))
            return;

        //register keyboard callbacks
        viz.registerKeyboardCallback(&OctreeViewer::keyboardEventOccurred, *this, nullptr);

        //key legends
        viz.addText ("Keys:", 0, 170, 0.0, 1.0, 0.0, "keys_t");
        viz.addText ("a -> Increment displayed depth", 10, 155, 0.0, 1.0, 0.0, "key_a_t");
        viz.addText ("z -> Decrement displayed depth", 10, 140, 0.0, 1.0, 0.0, "key_z_t");
        viz.addText ("v -> Toggle octree cubes representation", 10, 125, 0.0, 1.0, 0.0, "key_v_t");
        viz.addText ("b -> Toggle centroid points representation", 10, 110, 0.0, 1.0, 0.0, "key_b_t");
        viz.addText ("n -> Toggle original point cloud representation", 10, 95, 0.0, 1.0, 0.0, "key_n_t");

        //set current level to half the maximum one
        displayedDepth = static_cast<int> (std::floor (octree.getTreeDepth() / 2.0));
        if (displayedDepth == 0)
            displayedDepth = 1;

        // assign point cloud to octree
        octree.setInputCloud (cloud);

        // add points from cloud to octree
        octree.addPointsFromInputCloud ();

        //show octree at default depth
        extractPointsAtLevel(displayedDepth);

        //reset camera
        viz.resetCameraViewpoint("cloud");

        //run main loop
        run();

    }

private:
    //========================================================
    // PRIVATE ATTRIBUTES
    //========================================================
    //visualizer
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr xyz;

    pcl::visualization::PCLVisualizer viz;
    //original cloud
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud;
    //displayed_cloud
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr displayCloud;
    // cloud which contains the voxel center
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloudVoxel;
    //octree
    pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZLNormal> octree;
    //level
    int displayedDepth;
    //bool to decide what should be display
    bool wireframe{true};
    bool show_cubes_{true}, show_centroids_{false}, show_original_points_{false};
    float point_size_{1.0};
    //========================================================

    /* \brief Callback to interact with the keyboard
     *
     */
    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void *)
    {

        if (event.getKeySym () == "a" && event.keyDown ())
        {
            IncrementLevel ();
        }
        else if (event.getKeySym () == "z" && event.keyDown ())
        {
            DecrementLevel ();
        }
        else if (event.getKeySym () == "v" && event.keyDown ())
        {
            show_cubes_ = !show_cubes_;
            update ();
        }
        else if (event.getKeySym () == "b" && event.keyDown ())
        {
            show_centroids_ = !show_centroids_;
            update ();
        }
        else if (event.getKeySym () == "n" && event.keyDown ())
        {
            show_original_points_ = !show_original_points_;
            update ();
        }
        else if (event.getKeySym () == "w" && event.keyDown ())
        {
            if (!wireframe)
                wireframe = true;
            update ();
        }
        else if (event.getKeySym () == "s" && event.keyDown ())
        {
            if (wireframe)
                wireframe = false;
            update ();
        }
        else if ((event.getKeyCode () == '-') && event.keyDown ())
        {
            point_size_ = std::max(1.0f, point_size_ * (1 / 2.0f));
            update ();
        }
        else if ((event.getKeyCode () == '+') && event.keyDown ())
        {
            point_size_ *= 2.0f;
            update ();
        }
    }

    /* \brief Graphic loop for the viewer
     *
     */
    void run()
    {
        while (!viz.wasStopped())
        {
            //main loop of the visualizer
            viz.spinOnce(100);
            std::this_thread::sleep_for(100ms);
        }
    }

    /* \brief Helper function that read a pointcloud file (returns false if pbl)
     *  Also initialize the octree
     *
     */
    bool loadCloud(std::string &filename)
    {
        std::cout << "Loading file " << filename.c_str() << std::endl;
        //read cloud
        if (pcl::io::load (filename, *cloud))
        {
            return false;
        }

        //remove NaN Points
        pcl::Indices nanIndexes;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, nanIndexes);
        std::cout << "Loaded " << cloud->size() << " points" << std::endl;

        //create octree structure
        octree.setInputCloud(cloud);
        //update bounding box automatically
        octree.defineBoundingBox();
        //add points in the tree
        octree.addPointsFromInputCloud();
        return true;
    }

    /* \brief Helper function that draw info for the user on the viewer
     *
     */
    void showLegend ()
    {
        char dataDisplay[256];
        sprintf (dataDisplay, "Displaying octree cubes: %s", (show_cubes_) ? ("True") : ("False"));
        viz.removeShape ("disp_octree_cubes");
        viz.addText (dataDisplay, 0, 75, 1.0, 0.0, 0.0, "disp_octree_cubes");

        sprintf (dataDisplay, "Displaying centroids voxel: %s", (show_centroids_) ? ("True") : ("False"));
        viz.removeShape ("disp_centroids_voxel");
        viz.addText (dataDisplay, 0, 60, 1.0, 0.0, 0.0, "disp_centroids_voxel");

        sprintf (dataDisplay, "Displaying original point cloud: %s", (show_original_points_) ? ("True") : ("False"));
        viz.removeShape ("disp_original_points");
        viz.addText (dataDisplay, 0, 45, 1.0, 0.0, 0.0, "disp_original_points");

        char level[256];
        sprintf (level, "Displayed depth is %d on %zu", displayedDepth, static_cast<std::size_t>(octree.getTreeDepth()));
        viz.removeShape ("level_t1");
        viz.addText (level, 0, 30, 1.0, 0.0, 0.0, "level_t1");

        viz.removeShape ("level_t2");
        sprintf(level,
                "Voxel size: %.4fm [%zu voxels]",
                std::sqrt(octree.getVoxelSquaredSideLen(displayedDepth)),
                static_cast<std::size_t>(cloudVoxel->size()));
        viz.addText (level, 0, 15, 1.0, 0.0, 0.0, "level_t2");
    }

    /* \brief Visual update. Create visualizations and add them to the viewer
     *
     */
    void update()
    {
        //remove existing shapes from visualizer
        clearView ();

        showLegend ();

        if (show_cubes_)
        {
            //show octree as cubes
            showCubes (std::sqrt (octree.getVoxelSquaredSideLen (displayedDepth)));
        }

        if (show_centroids_)
        {
            //show centroid points
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZLNormal> color_handler (cloudVoxel, "x");
            viz.addPointCloud (cloudVoxel, color_handler, "cloud_centroid");
            viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size_, "cloud_centroid");
        }

        if (show_original_points_)
        {
            //show origin point cloud
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZLNormal> color_handler (cloud, "z");
            viz.addPointCloud (cloud, color_handler, "cloud");
            viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size_, "cloud");
        }
    }

    /* \brief remove dynamic objects from the viewer
     *
     */
    void clearView()
    {
        //remove cubes if any
        vtkRenderer *renderer = viz.getRenderWindow ()->GetRenderers ()->GetFirstRenderer ();
        while (renderer->GetActors ()->GetNumberOfItems () > 0)
            renderer->RemoveActor (renderer->GetActors ()->GetLastActor ());
        //remove point clouds if any
        viz.removePointCloud ("cloud");
        viz.removePointCloud ("cloud_centroid");
    }


    /* \brief display octree cubes via vtk-functions
     *
     */
    void showCubes(double voxelSideLen)
    {
        vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New ();

        // Create every cubes to be displayed
        double s = voxelSideLen / 2.0;
        for (const auto &point : cloudVoxel->points)
        {
            double x = point.x;
            double y = point.y;
            double z = point.z;

            vtkSmartPointer<vtkCubeSource> wk_cubeSource = vtkSmartPointer<vtkCubeSource>::New ();

            wk_cubeSource->SetBounds (x - s, x + s, y - s, y + s, z - s, z + s);
            wk_cubeSource->Update ();

            appendFilter->AddInputData (wk_cubeSource->GetOutput ());
        }

        // Remove any duplicate points
        vtkSmartPointer<vtkCleanPolyData> cleanFilter = vtkSmartPointer<vtkCleanPolyData>::New ();

        cleanFilter->SetInputConnection (appendFilter->GetOutputPort ());
        cleanFilter->Update ();

        //Create a mapper and actor
        vtkSmartPointer<vtkPolyDataMapper> multiMapper = vtkSmartPointer<vtkPolyDataMapper>::New ();

        multiMapper->SetInputConnection (cleanFilter->GetOutputPort ());

        vtkSmartPointer<vtkActor> multiActor = vtkSmartPointer<vtkActor>::New ();

        multiActor->SetMapper (multiMapper);

        multiActor->GetProperty ()->SetColor (1.0, 1.0, 1.0);
        multiActor->GetProperty ()->SetAmbient (1.0);
        multiActor->GetProperty ()->SetLineWidth (1);
        multiActor->GetProperty ()->EdgeVisibilityOn ();
        multiActor->GetProperty ()->SetOpacity (1.0);

        if (wireframe)
        {
            multiActor->GetProperty ()->SetRepresentationToWireframe ();
        }
        else
        {
            multiActor->GetProperty ()->SetRepresentationToSurface ();
        }

        // Add the actor to the scene
        viz.getRenderWindow ()->GetRenderers ()->GetFirstRenderer ()->AddActor (multiActor);

        // Render and interact
        viz.getRenderWindow ()->Render ();
    }

    /* \brief Extracts all the points of depth = level from the octree
     *
     */
    void extractPointsAtLevel(int depth)
    {
        displayCloud->points.clear();
        cloudVoxel->points.clear();

        pcl::PointXYZLNormal pt_voxel_center;
        pcl::PointXYZLNormal pt_centroid;
        std::cout << "===== Extracting data at depth " << depth << "... " << std::flush;
        double start = pcl::getTime ();

        for (pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZLNormal>::FixedDepthIterator tree_it = octree.fixed_depth_begin (depth);
             tree_it != octree.fixed_depth_end ();
             ++tree_it)
        {
            // Compute the point at the center of the voxel which represents the current OctreeNode
            Eigen::Vector3f voxel_min, voxel_max;
            octree.getVoxelBounds (tree_it, voxel_min, voxel_max);

            pt_voxel_center.x = (voxel_min.x () + voxel_max.x ()) / 2.0f;
            pt_voxel_center.y = (voxel_min.y () + voxel_max.y ()) / 2.0f;
            pt_voxel_center.z = (voxel_min.z () + voxel_max.z ()) / 2.0f;
            cloudVoxel->points.push_back (pt_voxel_center);

            // If the asked depth is the depth of the octree, retrieve the centroid at this LeafNode
            if (octree.getTreeDepth () == static_cast<unsigned int>(depth))
            {
                auto* container = dynamic_cast<pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZLNormal>::LeafNode*> (tree_it.getCurrentOctreeNode ());

                container->getContainer ().getCentroid (pt_centroid);
            }
                // Else, compute the centroid of the LeafNode under the current BranchNode
            else
            {
                // Retrieve every centroid under the current BranchNode
                pcl::octree::OctreeKey dummy_key;
                pcl::PointCloud<pcl::PointXYZLNormal>::VectorType voxelCentroids;
                octree.getVoxelCentroidsRecursive (dynamic_cast<pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZLNormal>::BranchNode*> (*tree_it), dummy_key, voxelCentroids);

                // Iterate over the leafs to compute the centroid of all of them
                pcl::CentroidPoint<pcl::PointXYZLNormal> centroid;
                for (const auto &voxelCentroid : voxelCentroids)
                {
                    centroid.add (voxelCentroid);
                }
                centroid.get (pt_centroid);
            }

            displayCloud->points.push_back (pt_centroid);
        }

        double end = pcl::getTime ();
        printf("%zu pts, %.4gs. %.4gs./pt. =====\n",
               static_cast<std::size_t>(displayCloud->size()),
               end - start,
               (end - start) / static_cast<double>(displayCloud->size()));

        update();
    }

    /* \brief Helper function to increase the octree display level by one
     *
     */
    bool IncrementLevel()
    {
        if (displayedDepth < static_cast<int> (octree.getTreeDepth ()))
        {
            displayedDepth++;
            extractPointsAtLevel(displayedDepth);
            return true;
        }
        return false;
    }

    /* \brief Helper function to decrease the octree display level by one
     *
     */
    bool DecrementLevel()
    {
        if (displayedDepth > 0)
        {
            displayedDepth--;
            extractPointsAtLevel(displayedDepth);
            return true;
        }
        return false;
    }

};


pcl::PointXYZ computeCentroidOfVoxel(const pcl::PointCloud<pcl::PointXYZ>& vecOfVectors) {
    pcl::PointXYZ average;

    for (const auto& vec : vecOfVectors) {
        average.x += vec.x;
        average.y += vec.y;
        average.z += vec.z;
    }

    const float V_size = vecOfVectors.size();
    average.x = average.x / V_size;
    average.y = average.y / V_size;
    average.z = average.z / V_size;


    return average;
}











const int M = 30000;           // I don't know how to select this number
const float K = 0.1;           // default
const float C = 0.25;          // default
float OffLineRadius {0.005};  // I need to tune this later (0.0025, 0.005)

float tolerance {OffLineRadius*30/100}; //20% of radius

// Online
const float Ps = 0.9;                    // probability of success
const float Octree_resolution = 0.0016;         // I chose this number to keep 50% of the original scene


int main() {

    OuterMap generatedMap = DB::create_hashMap(OffLineRadius,tolerance, "../YCB_ply/TWO");
    DB::to_serialize_hashMap(generatedMap, "Adel.bin");
    std::string filename = "Adel.bin";
    OuterMap deserialized_map = DB::to_deserialize_hashMap(filename);

    for (auto key : deserialized_map)
    {
        cout << key.first << endl;
    }
    cout << "****************************************************************************" << endl;


    pcl::PointCloud<pcl::PointXYZ>::Ptr S_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr S_cloud_Box_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr S_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    S_cloud->clear();
    S_cloud_Box_filtered->clear();
    S_cloud_filtered->clear();
    downsampled_cloud->clear();
    temp_cloud->clear();


    if (pcl::io::loadPLYFile<pcl::PointXYZ>("Apple_and_lemon.ply", *S_cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }


    pcl::CropBox<pcl::PointXYZ> cropFilter;
    Eigen::Vector4f minPoint(-2.0, -2.0, -2.0, 2.0); // Define your minimum point
    Eigen::Vector4f maxPoint(2.0, 2.0, 2.0, 2.0);    // Define your maximum point
    cropFilter.setInputCloud(S_cloud); // Input cloud
    cropFilter.setMin(minPoint);
    cropFilter.setMax(maxPoint);
    cropFilter.filter(*S_cloud_Box_filtered); // Apply the filter

    cout << "Box cloud size " << S_cloud_Box_filtered->size() << endl;


    // ***** OCtree Downsampling


    std::vector<int> Indices_of_points_in_one_voxel;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (Octree_resolution);
    octree.setInputCloud (S_cloud_Box_filtered);
    octree.addPointsFromInputCloud ();


    int c {0};
    auto octree_it = octree.begin();
    while (octree_it != octree.end())
    {
        temp_cloud->clear();
        if (octree_it.isLeafNode())
        {
            c++;
            Indices_of_points_in_one_voxel = (octree_it.getLeafContainer().getPointIndicesVector());
            for (auto i : Indices_of_points_in_one_voxel)
            {
                temp_cloud->push_back(S_cloud_Box_filtered->points[i]);
            }
            downsampled_cloud->push_back(computeCentroidOfVoxel(*temp_cloud));

        }
        ++octree_it;

    }

    pcl::io::savePLYFile("Downsampled.ply", *downsampled_cloud);

    cout << "number of Voxels in the tree " << c << endl;

    cout << "number of points in the downsampled Cloud " << downsampled_cloud->size() << endl;

/*    std::string fileNAme = "Downsampled.ply";
    OctreeViewer (fileNAme, resolution);*/


    // ********* Lets compute N of iterations N
    int n = downsampled_cloud->size();
    int N = (-n * log(1 - Ps)) / (M * K * C);    // number of iterations needed
    cout << "Number of iterations N : " << N <<endl;

    // ********* Sample one point randomly, and from this point sample points that are d away from this point

    std::vector<int> Index_of_neighboring_points;
    std::vector<float> pointRadiusSquaredDistance;
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_of_neighboring_points (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointXYZ PointA_XYZ;
    pcl::PointXYZ PointB_XYZ;
    pcl::PointXYZLNormal PointA_with_normal;
    pcl::PointXYZLNormal PointB_with_normal;

    pcl::PointCloud<pcl::PointXYZ>::Ptr Points_within_the_shpere (new pcl::PointCloud<pcl::PointXYZ>);

    int X {0};

    // I use this kdtree to find the nearest 100 neighbors to PointA, PointB to estimate the normal



    std::vector<int> k_indices;
    std::vector<float> k_squared_distances;

    pcl::PointCloud<pcl::PointXYZ>::Ptr neighbors_of_PointA_XYZ (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr Normals_neighbors_of_PointA (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr neighbors_of_PointA_with_Normal (new pcl::PointCloud<pcl::PointXYZLNormal>);


    pcl::PointCloud<pcl::PointXYZ>::Ptr neighbors_of_PointB (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr Normals_neighbors_of_PointB (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr neighbors_of_PointB_with_Normal (new pcl::PointCloud<pcl::PointXYZLNormal>);


    neighbors_of_PointA_XYZ->clear();
    neighbors_of_PointB->clear();

    pcl::PointXYZLNormal PointA_XYZNorm = 0;
    pcl::PointXYZLNormal PointB_XYZNorm = 0;

    std::string DESCRIPTOR_in_str;

    for (int i = 0; i < N ; ++i) {

        int index_of_pointA = rand() % downsampled_cloud->size();
        PointA_XYZ = downsampled_cloud->points[index_of_pointA];

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_PointA;
        kdtree_PointA.setInputCloud(downsampled_cloud);

        std::vector<int> pointIdxKNNSearch;
        std::vector<float> pointKNNSquaredDistance;

        if (kdtree_PointA.radiusSearch(PointA_XYZ, OffLineRadius, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
            for (auto Idx: pointIdxKNNSearch) {
                neighbors_of_PointA_XYZ->push_back(downsampled_cloud->points[Idx]);
            }
        }

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(neighbors_of_PointA_XYZ);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(OffLineRadius);
        ne.compute(*Normals_neighbors_of_PointA);
        pcl::concatenateFields(*neighbors_of_PointA_XYZ, *Normals_neighbors_of_PointA,
                               *neighbors_of_PointA_with_Normal);


        for (auto P_A : *neighbors_of_PointA_with_Normal) {

            if (P_A.x == PointA_XYZ.x && P_A.y == PointA_XYZ.y && P_A.z == PointA_XYZ.z) {
                PointA_XYZNorm = P_A;
            }
        }

        // So far, we have all points around (pointA) with normal
        pcl::KdTreeFLANN<pcl::PointXYZLNormal> kdtree_Point2;
        kdtree_Point2.setInputCloud(neighbors_of_PointA_with_Normal);

        std::vector<int> pointIdxKNNSearch_2;
        std::vector<float> pointKNNSquaredDistance_2;

        if (kdtree_Point2.radiusSearch(PointA_XYZNorm, OffLineRadius, pointIdxKNNSearch_2,
                                       pointKNNSquaredDistance_2) > 0) {
            for (int In = 0; In < pointKNNSquaredDistance_2.size(); ++In) {
                if (sqrt(pointKNNSquaredDistance_2[In]) > (OffLineRadius - tolerance) &&
                    sqrt(pointKNNSquaredDistance_2[In]) < (OffLineRadius + tolerance)) {
                    PointB_XYZNorm = (*neighbors_of_PointA_with_Normal)[In];
                  /*  cout << "PointB is : " << PointB_XYZNorm << endl;
                    cout << "PointA is : " << PointA_XYZNorm << endl;*/
                    break;
                }
            }


            auto DESCRIPTOR = Compute_the_descriptor(PointA_XYZNorm, PointB_XYZNorm);
            DESCRIPTOR_in_str = fromVectorToString(DESCRIPTOR);
            cout << DESCRIPTOR_in_str << endl;

            for (auto Key :deserialized_map )
            {
                if (Key.first == DESCRIPTOR_in_str)
                {
                   // cout << "Found it +++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                }
            }

        }


        neighbors_of_PointA_with_Normal->clear();
        PointA_XYZNorm = {0,0,0,0,0,0,0,0};
        PointB_XYZNorm = {0,0,0,0,0,0,0,0};


    }


        Eigen::Vector3f OneDescriptor = {0.294422,1.585185,1.645098};
        std::string SEARCH_DESCRIPTOR = fromVectorToString(OneDescriptor);

        for (auto Key :deserialized_map )
        {
            if (Key.first == SEARCH_DESCRIPTOR)
            {
                cout << "Found it +++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
            }
        }




    return 0;
}