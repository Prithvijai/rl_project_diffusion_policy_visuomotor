from pathlib import Path
import os
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image
import threading
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge
import copy
from math import sin, cos, pi
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# --- GLOBAL VARIABLES ---
bridge = CvBridge()
tool_pose_xy = [0.0, 0.0] 
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 360
vid_W = 640
device = "cuda"

# This where we can our LOAD POLICY ---
pretrained_policy_path = Path("/media/saitama/Games1/Documents_ubuntu/lerobot/outputs/train/my_pusht_diffusion/20251119160215")
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.reset()
print("Policy Loaded.")

class Get_End_Effector_Pose(Node):
    def __init__(self):
        super().__init__('get_modelstate')
        self.sub_tf = self.create_subscription(TFMessage, '/tf', self.listener_callback, 10)
        self.sub_isaac = self.create_subscription(TFMessage, '/isaac_tf', self.listener_callback, 10)
        self.euler_angles = np.array([0.0, 0.0, 0.0], float)

    def listener_callback(self, data):
        global tool_pose_xy, tbar_pose_xyw
        for transform in data.transforms:
            if "tool" in transform.child_frame_id or "end_effector" in transform.child_frame_id: 
                tool_pose_xy[0] = transform.transform.translation.y
                tool_pose_xy[1] = transform.transform.translation.x
            
            if "tbar" in transform.child_frame_id:
                t = transform.transform.translation
                r = transform.transform.rotation
                tbar_pose_xyw[0] = t.y   # World Y
                tbar_pose_xyw[1] = t.x   # World X
                self.euler_angles[:] = R.from_quat([r.x, r.y, r.z, r.w]).as_euler('xyz', degrees=False)
                tbar_pose_xyw[2] = self.euler_angles[2]

class Action_Publisher(Node):
    def __init__(self):
        super().__init__('Joy_Publisher')
        Hz = 10 
        self.pub_joy = self.create_publisher(Joy, '/joy', 10)
        self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
        self.joy_commands = Joy()
        self.joy_commands.axes = [0.0] * 8
        self.joy_commands.buttons = [1] + [0]*12 
        self.timer = self.create_timer(1/Hz, self.timer_callback)

        # Image Setup
        img_path = "/media/saitama/Games1/Documents_ubuntu/ur5_simulation/images/stand_top_plane.png"
        self.initial_image = cv2.imread(img_path)
        if self.initial_image is None:
            self.initial_image = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Constants
        self.scale = 1.639344 
        self.C_W = 182 
        self.OBL1 = int(150/self.scale)
        self.OBL2 = int(120/self.scale)
        self.OBW = int(30/self.scale)
        self.radius = int(10/self.scale)

        # --- TARGET POSE (TO BE UPDATED BY YOU) ---
        # Default guess. You must update this after running the calibration step!
        self.TARGET_POSE = [0.54, 0.0, 0.0] 
        self.max_coverage_episode = 0.0
        
        self.target_mask = np.zeros((vid_H, vid_W), dtype=np.uint8)
        # Init target mask (Will refresh in loop)
        self.target_area = 1 

    def get_tbar_polygons(self, world_y, world_x, th_rad):
        # 1. Horizontal (Screen X) <- Controlled by World Y
        # Center (320) - Scaled Y
        x_pix = int(320 - (world_y * 1000 / self.scale))

        # 2. Vertical (Screen Y) <- Controlled by World X
        # Bottom Offset (510) - Scaled X
        y_pix = int(510 - (world_x * 1000 / self.scale))
        
        th1 = -th_rad - pi/2
        dx1 = -self.OBW/2*cos(th1 - pi/2); dy1 = -self.OBW/2*sin(th1 - pi/2)
        pts1 = np.array([
            [int(cos(th1)*self.OBL1/2 - sin(th1)*self.OBW/2 + dx1 + self.C_W + x_pix), int(sin(th1)*self.OBL1/2 + cos(th1)*self.OBW/2 + dy1 + y_pix)],
            [int(cos(th1)*self.OBL1/2 - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + x_pix), int(sin(th1)*self.OBL1/2 + cos(th1)*(-self.OBW/2)+ dy1 + y_pix)],
            [int(cos(th1)*(-self.OBL1/2) - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + x_pix), int(sin(th1)*(-self.OBL1/2) + cos(th1)*(-self.OBW/2)+ dy1 + y_pix)],
            [int(cos(th1)*(-self.OBL1/2) - sin(th1)*self.OBW/2 + dx1 + self.C_W + x_pix), int(sin(th1)*(-self.OBL1/2) + cos(th1)*self.OBW/2 + dy1 + y_pix)]
        ], np.int32)

        th2 = -th_rad - pi
        dx2 = self.OBL2/2*cos(th2); dy2 = self.OBL2/2*sin(th2)
        pts2 = np.array([
            [int(cos(th2)*self.OBL2/2 - sin(th2)*self.OBW/2 + dx2 + self.C_W + x_pix), int(sin(th2)*self.OBL2/2 + cos(th2)*self.OBW/2 + dy2 + y_pix)],
            [int(cos(th2)*self.OBL2/2 - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + x_pix), int(sin(th2)*self.OBL2/2 + cos(th2)*(-self.OBW/2)+ dy2 + y_pix)],
            [int(cos(th2)*(-self.OBL2/2) - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + x_pix), int(sin(th2)*(-self.OBL2/2) + cos(th2)*(-self.OBW/2)+ dy2 + y_pix)],
            [int(cos(th2)*(-self.OBL2/2) - sin(th2)*self.OBW/2 + dx2 + self.C_W + x_pix), int(sin(th2)*(-self.OBL2/2) + cos(th2)*self.OBW/2 + dy2 + y_pix)]
        ], np.int32)
        return pts1, pts2

    def timer_callback(self):
        global tool_pose_xy, tbar_pose_xyw
        base_image = copy.copy(self.initial_image)
        
        # --- 1. Draw TARGET (Green) ---
        # Pass Y, X, Theta
        t_pts1, t_pts2 = self.get_tbar_polygons(self.TARGET_POSE[1], self.TARGET_POSE[0], self.TARGET_POSE[2])
        cv2.polylines(base_image, [t_pts1, t_pts2], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Update Target Mask (Do this here to ensure it matches visual)
        self.target_mask = np.zeros((vid_H, vid_W), dtype=np.uint8)
        cv2.fillPoly(self.target_mask, [t_pts1, t_pts2], 255)
        self.target_area = cv2.countNonZero(self.target_mask)

        # --- 2. Draw ROBOT (Blue/Red) ---
        # FIX: Pass tbar[0] (Y) then tbar[1] (X) to match Target logic
        c_pts1, c_pts2 = self.get_tbar_polygons(tbar_pose_xyw[0], tbar_pose_xyw[1], tbar_pose_xyw[2])
        cv2.fillPoly(base_image, [c_pts1, c_pts2], (0, 0, 180)) 

        # --- 3. Draw Tool ---
        x = int(320 - (tool_pose_xy[0]*1000 / self.scale))
        y = int(510 - (tool_pose_xy[1]*1000 / self.scale))
        cv2.circle(base_image, center=(x, y), radius=5, color=(0, 200, 0), thickness=cv2.FILLED)

        img_msg = bridge.cv2_to_imgmsg(base_image, encoding="bgr8")
        self.pub_img.publish(img_msg)

        # --- 4. EVALUATION ---
        current_mask = np.zeros((vid_H, vid_W), dtype=np.uint8)
        cv2.fillPoly(current_mask, [c_pts1, c_pts2], 255)
        intersection = cv2.bitwise_and(self.target_mask, current_mask)
        coverage = cv2.countNonZero(intersection) / self.target_area if self.target_area > 0 else 0.0
        if coverage > self.max_coverage_episode: self.max_coverage_episode = coverage

        # PRINT DEBUG INFO FOR CALIBRATION
        print(f"ROBOT AT: [{tbar_pose_xyw[1]:.2f}, {tbar_pose_xyw[0]:.2f}, {tbar_pose_xyw[2]:.2f}]")
        print(f"Coverage: {coverage:.2f} | Best: {self.max_coverage_episode:.2f}", end="\r")
        
        # --- 5. INFERENCE ---
        state = torch.from_numpy(np.array(tool_pose_xy)).to(torch.float32).to(device, non_blocking=True).unsqueeze(0)
        image = torch.from_numpy(base_image).to(torch.float32).permute(2, 0, 1).to(device, non_blocking=True) / 255
        image = image.unsqueeze(0)
        observation = {"observation.state": state, "observation.image": image}
        with torch.inference_mode():
            action = policy.select_action(observation)
        numpy_action = action.squeeze(0).to("cpu").numpy()
        self.joy_commands.header.stamp = self.get_clock().now().to_msg()
        self.joy_commands.axes[0] = float(numpy_action[0])
        self.joy_commands.axes[1] = float(numpy_action[1])
        self.pub_joy.publish(self.joy_commands)

if __name__ == '__main__':
    rclpy.init(args=None)
    get_end_effector_pose = Get_End_Effector_Pose()
    joy_publisher = Action_Publisher()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_end_effector_pose)
    executor.add_node(joy_publisher)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = joy_publisher.create_rate(20)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
    executor_thread.join()
