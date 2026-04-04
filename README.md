# Human Recognition-Driven Interaction

This project enables the Hello Robot Stretch to recognize and navigate to humans, specifically designed to support older adults in assistive environments.

---

## 🚀 Quickstart

1.  **Prepare the Robot:** Ensure you have the core `stretch_ai` repository installed on your robot.
    * **Source:** [hello-robot/stretch_ai](https://github.com/hello-robot/stretch_ai)
2.  **Clone this Project:** Pull this repository onto your **local workstation** and follow the same installation instruction as the `stretch_ai` repo.

---

## 🛠 Software Requirements

### 1. Dependencies
On your local machine, install the Google MediaPipe library:
```bash
pip install mediapipe
```

### 2. Virtual Environment
On your GPU workstation, ensure you are using a virtual environment before running the application:
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv
source venv/bin/activate  # Or your specific env command
deactivate # when done working within the env
```
---
## 🏃 How to Run

To run the application successfully, you must follow this specific execution order:

### Step 1: On the Robot
Open two separate terminal windows on the robot's computer:

1. **Home the Robot:**
   ```bash
   python3 stretch_robot_home.py
   ```
2. **Start the ROS2 Bridge:**
   
   Navigate into the `stretch_ai` folder on the robot and execute:
   ```bash
   ./scripts/run_stretch_ai_ros2_bridge_server.sh
   ```
### Step 2: On Your Local Machine
Once the robot server is active, set the robot IP address on your GPU machine and launch the navigation module:
```bash
./scripts/set_robot_ip.sh #.#.#.# 
python3 -m stretch.app.navigate-to-person
```
---
## 📄 Citation

This work is scheduled to appear at the **2026 ARSO** conference. Please cite our paper if you use this repository in your research:

```bibtex
@inproceedings{upadhyay2026assistive,
  title={Human Recognition-Driven Interaction with Hello Robot Stretch for Supporting Older Adults},
  author={Upadhyay, Aakriti and Gamble, Rose},
  booktitle={Proceedings of the 2026 IEEE International Conference on Advanced Robotics and its Social Impacts (ARSO)},
  year={2026},
  organization={IEEE},
  note={To appear}
}
```
