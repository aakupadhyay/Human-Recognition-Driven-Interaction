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
import mediapipe as mp
import cv2

from stretch.agent import RobotAgent, RobotClient
from stretch.core.task import Task
from stretch.agent.task.emote.follow_person import FindPerson
from stretch.perception import create_semantic_sensor

# print("Starting Face Detection -- Camera rotation and finding location of the person")

pending_frames = {}
pending_obs = {}
face_detected = False

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the live stream mode:
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        frame = pending_frames.pop(timestamp_ms, None)
        observation = pending_obs.pop(timestamp_ms, None)
        global face_detected

        print("Callback timestamp:", timestamp_ms)
        print("face detector result:", result)

        if frame is None or observation is None:
            return

        if not result.detections:
            return

        for det in result.detections:
            bboxC = det.bounding_box

            h, w, _ = frame.shape
            x1 = max(0, bboxC.origin_x)
            y1 = max(0, bboxC.origin_y)
            x2 = min(w, bboxC.origin_x + bboxC.width)
            y2 = min(h, bboxC.origin_y + bboxC.height)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            depth_image = observation.depth
            z = depth_image[cy, cx]

            intr = observation.camera_K
            fx, fy = intr[0, 0], intr[1, 1]
            px, py = intr[0, 2], intr[1, 2]

            X = (cx - px) * z / fx
            Y = (cy - py) * z / fy
            Z = z

            print(f"Face position (camera frame): X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_detected = True

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='/home/aakriti/demo/stretch_ai/src/stretch/app/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

detector = FaceDetector.create_from_options(options)

# ---------------- Detect Face and interact -----------------


def scan_head_positions(
    robot,
    agent,
    head_positions,
    sleep_time,
):
    """
    Move robot head through predefined pan/tilt positions, capture observations,
    and run asynchronous MediaPipe detection.

    Args:
        agent: Agent with update() method
        head_positions (list of tuples): [(pan, tilt), ...] in radians
        sleep_time (float): Delay between head movements (seconds)
    """
    for pan, tilt in head_positions:
        if face_detected:
            print("Detected Face, Starting Interaction")
            return
        
        # Move head
        robot.head_to(head_pan=pan, head_tilt=tilt, blocking=True)

        # Update agent state
        agent.update()

        # Get observation
        observation = robot.get_observation()
        frame = observation.rgb.copy()

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # Timestamp (ms)
        timestamp_ms = int(time.time() * 1000)

        # Store pending data
        pending_frames[timestamp_ms] = frame
        pending_obs[timestamp_ms] = observation

        # Run async detection
        detector.detect_async(mp_image, timestamp_ms)
        print("Frame timestamp:", timestamp_ms)

        time.sleep(sleep_time)

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
            enable_rerun_server=(not show_open3d),
        )
    else:
        semantic_sensor = None
    agent = RobotAgent(robot, robot.parameters, semantic_sensor, enable_realtime_updates=False)

    # observation = robot.get_observation()
    # print("Printing Camera Pose",observation.camera_pose)
    robot.move_to_nav_posture()

    if robot.parameters["agent"]["sweep_head_on_update"]:
        agent.update()
    else:
        
        on_floor = FindPerson(agent)
        front = FindPerson(agent, 0)
        
        task = on_floor.get_task(add_rotate=False)
        task.run()
        
        if on_floor.is_person_detected():
            print("Person found on the floor")
            agent.robot_say("I found person on the floor")
            time.sleep(1.0)
            agent.robot_say("Are you okay")
            time.sleep(0.5)
            agent.robot_say("Do you need help")
        else:
            # print("No person found")
            task = front.get_task(add_rotate=False)
            task.run()
            
            head_positions = [
                (0, 0), (0, -np.pi/4), (-np.pi/4, -np.pi/4), 
                (-np.pi/2, -np.pi/4), (-3*np.pi/4, -np.pi/4), (-np.pi, -np.pi/4),
                (0,0), (np.pi/4,0), (0,0), (-np.pi/4,0), (-np.pi/2,0), (-3*np.pi/4,0), (-np.pi,0), (0,0)
            ]
            scan_head_positions(robot=robot, agent=agent, head_positions=head_positions, sleep_time=1.5)
            agent.robot_say("Hello I am Stretch Robot")
            time.sleep(0.5)
            agent.robot_say("How may I help you today")
        
        # move head, update map, detect person, navigate to person, detect face, interact facing them
        

        # for pan, tilt in head_positions:
        #     robot.head_to(head_pan=pan, head_tilt=tilt, blocking=True)
        #     agent.update()
            
        #     observation = robot.get_observation() 

        #     frame = observation.rgb.copy()

        #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #     timestamp_ms = int(time.time() * 1000)

        #     pending_frames[timestamp_ms] = frame
        #     pending_obs[timestamp_ms] = observation

        #     detector.detect_async(mp_image, timestamp_ms)
        #     print("Frame timestamp:", timestamp_ms)
                    
        #     time.sleep(1.5)


    if show_open3d:
        agent.show_map()

    print("Done.")
    robot.stop()


if __name__ == "__main__":
    main()
