# MIM ROS package
Uses pointcloud_to_grid ROS package to convert `sensor_msgs/PointCloud2` LIDAR data to `nav_msgs/OccupancyGrid` 2D map data based on intensity and / or height.

Uses multiple levels of the intensity map to detect the solidity of an object. 