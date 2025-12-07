# rl_project_diffusion_policy_visuomotor

## Demo Video: Diffusion Policy on PushT Task

[![Watch the demo](https://img.youtube.com/vi/CH2lkMY8a9s/0.jpg)](https://youtu.be/CH2lkMY8a9s)

[YouTube Demo](https://youtu.be/CH2lkMY8a9s)

---

## Project Report: Visuomotor Policy Learning via Action Diffusion

### 1. Abstract
This project implements **Diffusion Policy**, a novel reinforcement learning framework that generates robot action sequences by iteratively denoising random noise. We applied this method to the **Push-T block manipulation task**, training a UR5 robotic arm to push a T-shaped block into a target zone using only state/visual feedback. The system was validated in both 2D (Gym-PushT) and 3D (NVIDIA Isaac Sim) environments.

### 2. Methodology

#### Diffusion Policy
Unlike traditional policies that map observations directly to actions, Diffusion Policy models the conditional distribution of action trajectories $p(A|O)$. It generates actions by reversing a diffusion process:
1.  **Forward Process**: Gradually adds Gaussian noise to expert action sequences.
    $$q(A_k^k | A_k^{k-1}) = \mathcal{N}(\sqrt{\alpha_k} A_k^{k-1}, (1 - \alpha_k) I)$$
2.  **Reverse Process**: A neural network $\epsilon_\theta$ predicts the noise at each step to recover the original action.
    $$A_k^{k-1} = \alpha_k (A_k^k - \gamma_k \epsilon_\theta(O_k, A_k^k, k)) + \sqrt{1 - \alpha_k^2} \mathcal{N}(0, I)$$

#### Robot Control Integration
*   **Inference**: The policy outputs a sequence of end-effector poses (Cartesian space).
*   **MoveIt 2**: We integrated MoveIt 2 to solve Inverse Kinematics (IK), converting Cartesian poses into joint angles for the UR5 arm while avoiding collisions.

### 3. Experiments & Results

#### Training
The model was trained on the PushT dataset (206 episodes) for 5000 steps.
*   **Loss Convergence**: The Mean Squared Error (MSE) loss decreased from an initial ~1.19 to ~0.10, indicating successful learning of the action distribution.
*   *(See `train_logs.txt` for the complete training history)*.

#### Evaluation
*   **Simulation**: The policy successfully generalized to the 3D Isaac Sim environment, demonstrating smooth trajectory tracking and successful block pushing.
*   **Metrics**: Success was defined by the overlap (IoU) between the block and the target.

### 4. Discussion & Future Work
*   **Sim-to-Real Gap**: While successful in simulation, real-world deployment faces challenges due to physical discrepancies (friction, sensor noise).
*   **Inference Latency**: The iterative denoising process introduces latency, which requires optimization for high-frequency control.
*   **Future Work**: We plan to collect real-world data to fine-tune the model and implement domain randomization to improve robustness.

---

## Installation & Setup

### 1. Download Project Files
**CRITICAL**: You must download the full project repository, which includes the custom `ur5_simulation` assets and configuration files.
*   [**Download Project Zip (Google Drive)**](https://drive.google.com/file/d/126dnCWyt8QnDNPsZq9z9n-ZXegeQzqoE/view?usp=sharing)
*   **Action**: Download, unzip, and place the contents in your workspace.
    *   *Note: The Python scripts in this repository (`3_train_policy_mod.py`, etc.) are copies of the files found in the zip archive under `lerobot/examples`.*

### 2. Trained Model Checkpoints
Our trained model checkpoints are hosted on Hugging Face. You can download them to skip training and proceed directly to evaluation.
*   [**Hugging Face Repo: diffusion_policy_custom_isaacsim**](https://huggingface.co/Saitama0510/diffusion_policy_custom_isaacsim/tree/main)

### 3. System Requirements
*   **OS**: Ubuntu 22.04 (Recommended) or 20.04
*   **GPU**: NVIDIA GPU with CUDA support

### 4. Install Dependencies
*   **NVIDIA Isaac Sim**: Download and install from the [Isaac Sim Archive](https://developer.nvidia.com/isaac-sim-archive).
*   **ROS 2 Humble**: Follow the [official installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).
*   **MoveIt 2**: Install MoveIt 2 for motion planning.
    ```bash
    sudo apt install ros-humble-moveit
    ```
*   **LeRobot**: Clone and install the Hugging Face library.
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    pip install -e ".[pusht]"
    ```

### 5. UR5 Simulation Setup
You need to build the `ur5_simulation` package included in the downloaded zip.
1.  Navigate to your ROS 2 workspace (e.g., `colcon_ws`).
2.  Ensure the `ur5_simulation` folder is in `src/`.
3.  Build the package:
    ```bash
    colcon build --packages-select ur5_simulation
    source install/setup.bash
    ```

### 6. Environment Setup
Create a conda environment to manage dependencies:
```bash
conda create -n env_isaaclab python=3.8
conda activate env_isaaclab
pip install torch torchvision torchaudio gym
```

---

## Simulation Setup

To run the simulation with our custom assets:
1.  Launch **NVIDIA Isaac Sim**.
2.  Go to **File > Open**.
3.  Navigate to the unzipped project folder and select: `urs_simulation/PushT_custom.usd`.
4.  This will load the UR5 robot, the table, the T-block, and the target zone.

---

## Workflow: Train, Evaluate, Metrics

### Step 1: Training
Train the diffusion policy using the provided script. This script loads the dataset and optimizes the policy network.
```bash
python 3_train_policy_mod.py
```
*   **Note**: This training process learns the behavior demonstrated in the **Demo Video** above.
*   **Output**: Checkpoints will be saved in `outputs/train/my_pusht_diffusion/`.
*   **Logs**: Monitor `train_logs.txt` to see the loss decrease over time.

### Step 2: Evaluation (Visual)
Run the trained policy in the ROS 2 + Isaac Sim environment to visually verify performance.

**Prerequisites**:
*   Isaac Sim must be running with `PushT_custom.usd` loaded.
*   The MoveIt 2 controller must be active.

**Commands**:
```bash
# Terminal 1: Launch MoveIt Controller
ros2 launch ur5_moveit_config arm_diffusion_control.launch.py

# Terminal 2: Run Inference
python 2_evaluate_pretrained_policy_ROS.py
```
*   **Action**: Watch the robot in Isaac Sim. It should pick up the block and push it to the target.

### Step 3: Collect Metrics
To quantitatively evaluate the model (Success Rate), run the evaluation loop:
```bash
python 2_evaluate_pretrained_policy_ROS_eval.py
```
*   This script runs multiple episodes and prints the success rate based on the final block position.

---

## Training Logs
The file `train_logs.txt` contains the raw training data.
*   **Format**: `step: <step_number> loss: <mse_loss>`
*   **Usage**: You can parse this file to plot the training curve and verify convergence.

---

## References & Source
*   **Diffusion Policy**: [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)
*   **LeRobot Library**: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
*   **MoveIt 2**: [https://moveit.picknik.ai/](https://moveit.picknik.ai/)