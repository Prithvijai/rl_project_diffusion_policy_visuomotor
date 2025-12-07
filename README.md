# rl_project_diffusion_policy_visuomotor

## Demo Video for us Runining the diffusion policy on PushT task

[![Watch the demo](https://img.youtube.com/vi/CH2lkMY8a9s/0.jpg)](https://youtu.be/CH2lkMY8a9s)

[YouTube Demo](https://youtu.be/CH2lkMY8a9s)

---

## Project Report: Visuomotor Policy Learning via Action Diffusion

### Overview
This project demonstrates **visuomotor robotic manipulation using Diffusion Policy**, a novel reinforcement learning-based framework that enables robots to learn control policies directly from visual inputs (camera feeds). We applied this approach to the **Push-T block manipulation task**, where a robotic arm must push a T-shaped block precisely into a T-shaped target region.

Our application focuses on:
*   Teaching a robot to perform goal-directed manipulation from raw sensory data.
*   Using diffusion-based action generation instead of conventional one-shot policy learning.
*   Testing and validating performance in both Gym-PushT (2D) and NVIDIA Isaac Sim (3D) environments.


## Installation & Setup

### 1. Download Project Files
**Important**: The full simulation environment including the `ur5_simulation` assets is available in the project zip file.
*   [Download Project Zip (Google Drive)](https://drive.google.com/file/d/126dnCWyt8QnDNPsZq9z9n-ZXegeQzqoE/view?usp=sharing)
*   **Action**: Download and unzip the file. Ensure the `ur5_simulation` directory is present in your workspace.

### 2. System Requirements
*   **OS**: Ubuntu 22.04
*   **GPU**: NVIDIA GPU with CUDA support (Recommended)

### 3. Install NVIDIA Isaac Sim
Download and install NVIDIA Isaac Sim from the [NVIDIA Omniverse](https://developer.nvidia.com/isaac-sim) website. Follow the official installation guide for your system.

### 4. Install ROS 2 & MoveIt 2
This project uses ROS 2 for robot control.
*   **ROS 2**: Install ROS 2 (Humble Hawksbill recommended for Ubuntu 22.04). [Installation Guide](https://docs.ros.org/en/humble/Installation.html)
*   **MoveIt 2**: Install MoveIt 2 for motion planning.
    ```bash
    sudo apt install ros-humble-moveit
    ```

### 5. Install LeRobot
Clone and install the Hugging Face `lerobot` library, which provides the Diffusion Policy implementation.
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 6. Environment Setup
Create a dedicated conda environment for the project to manage dependencies.
```bash
conda create -n env_isaaclab python=3.8
conda activate env_isaaclab
# Install additional dependencies
pip install torch torchvision torchaudio
pip install gym
```

---

## Usage

### Training
To train the diffusion policy on the PushT task:
```bash
python 3_train_policy_mod.py
```
*   This script initializes the `DiffusionPolicy` with the `DiffusionConfig`.
*   It loads the dataset from `lerobot/my_pusht`.
*   Training logs will be saved to `outputs/train/my_pusht_diffusion/`.

### Evaluation
To evaluate the trained policy in the ROS environment:
```bash
python 2_evaluate_pretrained_policy_ROS.py
```
*   Ensure ROS 2 and Isaac Sim are running.
*   This script loads the checkpoint and executes the policy on the robot.

---

## Results
The model was trained for 5000 steps. The training loss converged significantly, demonstrating the policy's ability to learn the action distribution.

*   **Initial Loss**: ~1.19
*   **Final Loss (approx)**: ~0.10
*   **Convergence**: The loss stabilized around 0.1 after approximately 500 steps, indicating effective learning of the denoising process.

*(See `train_logs.txt` for detailed step-by-step loss metrics)*

---

## Source of Implementation
This project utilizes the **LeRobot** library by Hugging Face for the Diffusion Policy implementation.
*   **Repository**: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)