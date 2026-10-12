#!/usr/bin/env python3
"""Play a bag and run front-end (fast_lio) + back-end (sc_pgo_ros2) together.

Usage:
    ros2 launch sc_pgo_ros2 run_all.launch.py bag_file:=/path/to/bag

Optional args:
    loop:   true|false  (default true)  play the bag in a loop
    rviz:   true|false  (default false) also launch RViz (front & back)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    declare_bag = DeclareLaunchArgument(
        "bag_file", description="Path to the ros2 bag (file or directory) to play"
    )
    declare_loop = DeclareLaunchArgument(
        "loop", default_value="false", description="Play the bag in a loop"
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="false", description="Launch RViz for front/back-end"
    )

    bag_file = LaunchConfiguration("bag_file")
    loop = LaunchConfiguration("loop")
    rviz = LaunchConfiguration("rviz")

    # ------------------------------------------------------------------
    # 1. Play the bag (loop version XOR once-only version)
    # ------------------------------------------------------------------
    bag_play_loop = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag_file, "--loop", "--clock"],
        output="screen",
        condition=IfCondition(loop),
    )
    bag_play_once = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag_file, "--clock"],
        output="screen",
        condition=UnlessCondition(loop),
    )

    # ------------------------------------------------------------------
    # 2. Front-end: FAST-LIO
    # ------------------------------------------------------------------
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fast_lio"),
                "launch",
                "mapping.launch.py",
            )
        ),
        launch_arguments={"rviz": rviz}.items(),
    )

    # ------------------------------------------------------------------
    # 3. Back-end: SC-PGO (loop closure + graph optimization)
    # ------------------------------------------------------------------
    sc_pgo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sc_pgo_ros2"),
                "launch",
                "sc_pgo.launch.py",
            )
        ),
        launch_arguments={"rvizscpgo": rviz}.items(),
    )

    return LaunchDescription(
        [
            declare_bag,
            declare_loop,
            declare_rviz,
            bag_play_loop,
            bag_play_once,
            fast_lio_launch,
            sc_pgo_launch,
        ]
    )
