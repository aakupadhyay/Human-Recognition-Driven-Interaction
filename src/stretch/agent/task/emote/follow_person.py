# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np
import mediapipe as mp

from stretch.agent.operations import (
    # ExtendArm,
    NavigateToObjectOperation,
    FaceDetectionOperation,
    SpeakOperation,
    UpdateOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Task


class FindPerson:
    """Find a person, navigate to them, and extend the arm toward them"""

    def __init__(self, agent: RobotAgent, tilt: float = -1.0 * np.pi / 3.0, target_object: str = "person") -> None:
        # super().__init__(agent)
        self.agent = agent

        self.target_object = target_object
        self.tilt = tilt
        self.detect_person = False

        # Sync these things
        self.robot = self.agent.robot
        self.voxel_map = self.agent.get_voxel_map()
        self.navigation_space = self.agent.space
        self.semantic_sensor = self.agent.semantic_sensor
        self.parameters = self.agent.parameters
        self.instance_memory = self.agent.get_voxel_map().instances
        assert (
            self.instance_memory is not None
        ), "Make sure instance memory was created! This is configured in parameters file."

        self.current_receptacle = None
        self.agent.reset_object_plans()
    
    def get_task(self, add_rotate: bool = False, mode: str = "one_shot") -> Task:
        """Create a task plan with loopbacks and recovery from failure. The robot will explore the environment, find objects, and pick them up

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.
            mode (str, optional): Type of task to create. Can be "one_shot" or "all". Defaults to "one_shot".

        Returns:
            Task: Executable task plan for the robot to pick up objects in the environment.
        """

        return self.get_one_shot_task(add_rotate=add_rotate)

    def get_one_shot_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan"""

        task = Task()

        update = UpdateOperation("update_scene", self.agent, retry_on_failure=False)
        update.configure(
            move_head=True,
            target_object=self.target_object,
            show_map_so_far=False,  # Uses Open3D display (blocking)
            clear_voxel_map=False,  # True,
            show_instances_detected=False,  # Uses OpenCV image display (blocking)
            match_method="name",  # "feature",
            arm_height=0.6,
            tilt=self.tilt,
        )

        found_a_person = SpeakOperation(
            name="found_a_person", agent=self.agent, parent=update, on_cannot_start=update
        )
        found_a_person.configure(
            message="I found a person! I am going to navigate to them.", sleep_time=2.0
        )
        
        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = NavigateToObjectOperation(
            name="go_to_object",
            agent=self.agent,
            parent=found_a_person,
            on_cannot_start=update,
            to_receptacle=False,
            for_manipulation =False,
        )

        look_for_face = SpeakOperation(
            name="look_for_face",
            agent=self.agent,
            parent=go_to_object,
            on_cannot_start=update,
        )
        look_for_face.configure(
            message="I navigated to the person. I am going to look for their face to ask a question.",
            sleep_time=5.0,
        )
        
        face_op = FaceDetectionOperation(
            name="detect_face",
            agent=self.agent,
            parent=look_for_face,
            on_cannot_start=update,
        )
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="/home/aakriti/demo/stretch_ai/src/stretch/app/blaze_face_short_range.tflite"
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=face_op.result_callback,  # ← bind callback here
        )

        detector = FaceDetector.create_from_options(options)
        
        face_op.configure(
            detector=detector,
            pan_start=np.pi / 3,
            pan_step=-np.pi / 6,
            num_steps=10, #11
            tilt=0.0,
            sleep_time=1.5,
            timeout=100.0,
        )
        
        on_floor_person = SpeakOperation(
            name="on_floor",
            agent=self.agent,
            parent=look_for_face,
            on_cannot_start=update,
        )
        on_floor_person.configure(
            message="I found a person on the floor. Are you okay",
            sleep_time=2.0,
        )
        
        interact_person = SpeakOperation(
            name="talking",
            agent=self.agent,
            parent=face_op,
            on_cannot_start=update,
        )
        interact_person.configure(
            message="Hello I am Stretch robot. How may I help you today",
            sleep_time=2.0,
        )
        
        task.add_operation(update)
        task.add_operation(found_a_person)
        task.add_operation(go_to_object)
        task.add_operation(look_for_face)
        
        if self.tilt == -1.0 * np.pi / 6.0:
            chosen_op = face_op
            task.add_operation(face_op)
            task.add_operation(interact_person)
        else:
            chosen_op = on_floor_person
            task.add_operation(on_floor_person)

        look_for_face.on_success = chosen_op
        # Terminate on a successful place
        return task


if __name__ == "__main__":
    from stretch.agent.robot_agent import RobotAgent
    from stretch.agent.zmq_client import HomeRobotZmqClient

    robot = HomeRobotZmqClient()
    # Create a robot agent with instance memory
    agent = RobotAgent(robot, create_semantic_sensor=True)

    task = FindPerson(agent).get_task(add_rotate=False)
    task.run()
