#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Sweep the head around and create a 3d map, then visualize it."""

import time
import click
import numpy as np
# import threading
import rclpy
import cv2

from stretch.agent import RobotAgent, RobotClient
# from stretch.core.task import Task
from stretch.agent.task.emote.follow_person import FindPerson
from stretch.perception import create_semantic_sensor


# ---------------- Main function ----------------
@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option(
    "--run_semantic_segmentation", is_flag=True, help="Run semantic segmentation on EE rgb images"
)
@click.option("--show-open3d", is_flag=True, help="Show the open3d visualization")
@click.option("--device_id", type=int, default=0, help="Device ID for camera")
@click.option("--verbose", is_flag=True, help="Print debug information from perception")

def main(
    robot_ip: str = "",
    local: bool = False,
    run_semantic_segmentation: bool = False,
    show_open3d: bool = False,
    device_id: int = 0,
    verbose: bool = False,on_floor:bool = False,
):

    # Create robot
    robot = RobotClient(robot_ip=robot_ip, use_remote_computer=(not local))
    if run_semantic_segmentation:
        semantic_sensor = create_semantic_sensor(
            parameters=robot.parameters,
            device_id=device_id,
            verbose=verbose,
            confidence_threshold=0.5,
            # enable_rerun_server=(not show_open3d),
        )
    else:
        semantic_sensor = None
    agent = RobotAgent(robot, robot.parameters, semantic_sensor, use_instance_memory=True, create_semantic_sensor=True, enable_realtime_updates=False)
    # agent = RobotAgent(robot, robot.parameters, semantic_sensor, enable_realtime_updates=False)

    # observation = robot.get_observation()
    # print("Printing Camera Pose",observation.camera_pose)
    robot.move_to_nav_posture()
    robot.move_base_to([0.0, 0.0, 0.0], blocking=True, timeout=30.0)
    
    # Show intermediate maps
    # agent.start(visualize_map_at_start=True, verbose=True)
       
    on_floor = FindPerson(agent)
    front = FindPerson(agent, -1.0 * np.pi / 6.0)
    
    task1 = on_floor.get_task(add_rotate=False)
    
    try:
        success = task1.run()
    except Exception as e:
        print("Task1 failed:", e)
        success = False 
    
    if not success:
        task2 = front.get_task(add_rotate=False)
        try:
            check = task2.run()
        except Exception as e:
            print("Task2 failed:", e)
            check = False
        
        if not check:
            time.sleep(3.0)
            agent.robot_say("I couldn't find person at my current position")
            time.sleep(3.0)
            agent.robot_say("Moving to another location")


    if show_open3d:
        agent.show_map()

    print("Done.")
    # print("Active threads:")
    # for t in threading.enumerate():
    #     print(t.name, t.daemon)
        
    agent.stop_realtime_updates()
    robot.stop()
    rclpy.shutdown()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()
