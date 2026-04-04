#!/usr/bin/env python
# Copyright (c) The University of Tulsa.
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

import math
# import matplotlib.pyplot as plt
# from brokenaxes import brokenaxes

from stretch.agent import RobotAgent, RobotClient
# from stretch.core.task import Task
from stretch.agent.task.emote.follow_person import FindPerson
from stretch.perception import create_semantic_sensor

def pose_xy(p):
        """
        Extract (x, y) from Pose or Tensor
        """
        # Tensor / numpy
        if hasattr(p, "__len__"):
            return float(p[0]), float(p[1])

        # Pose-like object
        return float(p.x), float(p.y)


def pose_distance(p1, p2):
    x1, y1 = pose_xy(p1)
    x2, y2 = pose_xy(p2)
    return math.hypot(x1 - x2, y1 - y2)

def score_frontier(pose, current_pose, visited_poses,
                alpha, beta, gamma=0.5):
    """
    Higher score = better frontier
    """
    d_robot = pose_distance(pose, current_pose) / 2.0
    d_visited = np.mean([pose_distance(pose, v) for v in visited_poses]) / 2.0
    
    d_obstacle = 1.0  # clearance

    score = (
        alpha * d_robot +
        beta * d_visited -
        gamma * d_obstacle
    )
    return score

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
    verbose: bool = False,
):

    # Create robot
    robot = RobotClient(robot_ip=robot_ip, use_remote_computer=(not local))
    if run_semantic_segmentation:
        semantic_sensor = create_semantic_sensor(
            parameters=robot.parameters,
            device_id=device_id,
            verbose=verbose,
            confidence_threshold=0.5,
        )
    else:
        semantic_sensor = None
    agent = RobotAgent(robot, robot.parameters, semantic_sensor, use_instance_memory=True, create_semantic_sensor=True, enable_realtime_updates=False)
    
    robot.move_to_nav_posture()
    robot.move_base_to([0.0, 0.0, 0.0], blocking=True, timeout=30.0)
       
    on_floor = FindPerson(agent)
    front = FindPerson(agent, -1.0 * np.pi / 6.0)
    
    MAX_ATTEMPTS = 10
    MAX_CANDIDATES = 10
    VISITED_DIST_THRESH = 1.45 #meters

    # ---------------- TIMER START ----------------
    start_time = time.time()

    visited_poses = []   # stores previously visited base poses
    attempt = 0
    success = False
    all_frontiers_explored = False
    space = agent.space
    
    while not success and attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"\n=== Search attempt {attempt}/{MAX_ATTEMPTS} ===")

        current = robot.get_base_pose()
        current_xy = pose_xy(current)
        
        # ---------- Frontal search ----------
        try:
            success = front.get_task(add_rotate=False).run()
        except Exception as e:
            print("Task2 failed:", e)
            success = False

        if success:
            break

        # ---------- Floor search ----------
        try:
            success = on_floor.get_task(add_rotate=False).run()
        except Exception as e:
            print("Task1 failed:", e)
            success = False

        if success:
            break

        # ---------- Recovery message ----------
        agent.robot_say("I couldn't find anyone here. Moving to another location.")
        time.sleep(5.0)
        
        # Mark current pose as visited
        visited_poses.append(current_xy)

        # ---------- Sample frontier candidates ----------
        frontier_candidates = []

        for pose in space.sample_closest_frontier(current):
            if pose is None:
                continue
            try:
                if not space.is_valid(pose):
                    continue
            except Exception:
                continue

            if any(pose_distance(pose, v) < VISITED_DIST_THRESH for v in visited_poses):
                continue

            frontier_candidates.append(pose)
            
            if len(frontier_candidates) >= MAX_CANDIDATES:
                print("I am here")
                break

        # ---------- Handle no frontiers left ----------
        if not frontier_candidates:
            agent.robot_say("I couldn't find any reachable frontiers.")
            all_frontiers_explored = True
            time.sleep(3.0)
            break

        # ---------- Score frontiers ----------
        scored_frontiers = []
        for pose in frontier_candidates:
            score = score_frontier(
                pose,
                current_pose=current,
                visited_poses=visited_poses,
                alpha=1.0,   # exploration weight
                beta=1.5     # revisit avoidance weight --1.5
            )
            scored_frontiers.append((score, pose))

        # Sort and select best
        scored_frontiers.sort(key=lambda x: x[0], reverse=True)
        best_score, candidate = scored_frontiers[0]

        print(f"Moving to frontier with score {best_score:.2f}")
        
        # ---------- Move robot ----------
        try:
            robot.move_base_to(candidate, blocking=True, timeout=30.0)
        except Exception as e:
            print("Navigation failed:", e)
            # If navigation fails, mark candidate as visited to avoid retry
            visited_poses.append(pose_xy(candidate))
            continue
            
    # ---------------- TIMER END ----------------
    end_time = time.time()
    total_time = end_time - start_time

    print("\n=== Task Summary ===")
    print(f"Success: {success}")
    print(f"Attempts: {attempt}")
    print(f"Total time taken: {total_time:.2f} seconds")

    # Final outcome
    if not success and not all_frontiers_explored:
        # print("No person found")
        agent.robot_say("I couldn't find anyone nearby.")
        
    if show_open3d:
        agent.show_map()

    print("Done.")
           
    agent.stop_realtime_updates()
    robot.stop()   


if __name__ == "__main__":
    main()
