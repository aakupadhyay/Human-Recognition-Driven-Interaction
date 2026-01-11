import time
import cv2
import numpy as np
import mediapipe as mp
from stretch.agent.base import ManagedOperation

FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

class FaceDetectionOperation(ManagedOperation):
    """
    Scan for a face by sweeping the head horizontally.
    Stops immediately when a face is detected.
    """

    _successful = False

    def configure(
        self,
        detector,
        pan_start: float = np.pi / 3,
        pan_step: float = -np.pi / 6,
        num_steps: int = 11,
        tilt: float = 0.0,
        sleep_time: float = 1.0,
        timeout: float = 10.0,
    ):
        self.detector = detector

        # Scan parameters
        self.pan_start = pan_start
        self.pan_step = pan_step
        self.num_steps = num_steps
        self.tilt = tilt

        self.sleep_time = sleep_time
        self.timeout = timeout

        # Async state
        self.face_detected = False
        self.pending_frames = {}
        self.pending_obs = {}

    def can_start(self) -> bool:
        return self.detector is not None

    def run(self) -> None:
        self.intro("Starting face detection scan.")

        robot = self.agent.robot
        start_time = time.time()

        for i in range(self.num_steps):
            if time.time() - start_time > self.timeout:
                self.warn("Face detection timed out.")
                return

            if self.face_detected:
                self._successful = True
                self.cheer("Face detected. Stopping scan.")
                self.detector.close()
                return

            pan = self.pan_start + i * self.pan_step

            robot.head_to(
                head_pan=pan,
                head_tilt=self.tilt,
                blocking=True,
            )

            self.agent.update()
            obs = robot.get_observation()
            frame = obs.rgb.copy()

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            timestamp_ms = int(time.time() * 1000)
            self.pending_frames[timestamp_ms] = frame
            self.pending_obs[timestamp_ms] = obs

            self.detector.detect_async(mp_image, timestamp_ms)
            time.sleep(self.sleep_time)

        # Final wait for async callback
        self._wait_for_callback(start_time)
        # self.detector.close()
        cv2.destroyAllWindows()

    def _wait_for_callback(self, start_time):
        while time.time() - start_time < self.timeout:
            if self.face_detected:
                self._successful = True
                return
            time.sleep(0.05)

        self.warn("Face detection finished without detections.")

    def was_successful(self) -> bool:
        return self._successful

    # -------- MediaPipe callback --------
    def result_callback(self, result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        frame = self.pending_frames.pop(timestamp_ms, None)
        observation = self.pending_obs.pop(timestamp_ms, None)

        if frame is None or observation is None:
            return

        if not result.detections:
            return

        det = result.detections[0]
        bbox = det.bounding_box

        cx = int(bbox.origin_x + bbox.width / 2)
        cy = int(bbox.origin_y + bbox.height / 2)

        z = observation.depth[cy, cx]
        self.info(f"Face detected at depth {z:.2f}m")

        self.face_detected = True
