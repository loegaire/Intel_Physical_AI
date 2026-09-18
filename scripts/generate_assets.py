#!/usr/bin/env python3
"""Generate SO-101 robot meshes and XML from primitive shapes.

This script creates the SO-101 robot model using basic geometric primitives
(cylinders, boxes, capsules) instead of relying on pre-generated STL files.
This avoids large binary assets in the repository.

The generated model is functionally equivalent to the Menagerie SO-101
but uses simple MuJoCo geom primitives for collision and visual.

Usage:
    python scripts/generate_assets.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASSETS_DIR = ROOT / "assets" / "so101"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# SO-101 dimensions (approximate, based on Menagerie model)
# All units in meters
LINK_SIZES = {
    "base": {"radius": 0.045, "height": 0.05},
    "shoulder": {"radius": 0.025, "height": 0.06},
    "upper_arm": {"radius": 0.02, "height": 0.14},
    "lower_arm": {"radius": 0.02, "height": 0.14},
    "wrist": {"radius": 0.02, "height": 0.05},
    "gripper_base": {"radius": 0.025, "height": 0.03},
    "finger": {"width": 0.01, "depth": 0.01, "length": 0.06},
}

# Colors (RGBA)
COLORS = {
    "yellow": [1.0, 0.82, 0.12, 1.0],
    "dark_gray": [0.1, 0.1, 0.1, 1.0],
    "gray": [0.5, 0.5, 0.5, 1.0],
    "light_gray": [0.7, 0.7, 0.7, 1.0],
}


def write_so101_xml():
    """Generate SO-101 XML using MuJoCo primitives."""
    xml = '''<mujoco model="so101">
  <option integrator="implicitfast" timestep="0.005" cone="elliptic" iterations="10" ls_iterations="20" impratio="10"/>

  <compiler angle="radian" autolimits="true"/>

  <visual>
    <scale contactwidth="0.1" contactheight="0.05" forcewidth="0.02" framelength="0.75" framewidth="0.025" camera="0.2"/>
  </visual>

  <default>
    <mesh maxhullvert="128"/>

    <default class="so101">
      <joint damping="1" frictionloss="0.1" armature="0.005"/>
      <position kp="50"/>
      <default class="visual">
        <geom type="cylinder" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3" condim="3"/>
      </default>
      <default class="collision_gripper">
        <geom group="3" condim="6" friction="1 5e-3 5e-4" solref="0.01 1" priority="1" rgba="1.0 0 0 1.0" mass="0"/>
      </default>
    </default>

    <default class="sts3215">
      <geom contype="0" conaffinity="0"/>
      <joint damping="0.60" frictionloss="0.052" armature="0.028"/>
      <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>
    </default>
  </default>

  <asset>
    <material name="yellow" rgba="1 0.82 0.12 1"/>
    <material name="dark_gray" rgba="0.1 0.1 0.1 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="light_gray" rgba="0.7 0.7 0.7 1"/>
  </asset>

  <worldbody>
    <!-- Link base -->
    <body name="base" pos="0 0 0" quat="1 0 0 0" childclass="so101">
      <inertial pos="0 0 0.025" mass="0.147" diaginertia="0.0001 0.0001 0.0001"/>
      
      <!-- Base cylinder -->
      <geom class="visual" pos="0 0 0.025" size="0.045 0.025" material="yellow"/>
      <geom class="collision" pos="0 0 0.025" size="0.045 0.025"/>
      
      <!-- Motor housing -->
      <geom class="visual" pos="0.025 0 0.05" size="0.02 0.015 0.02" material="dark_gray" type="box"/>
      
      <!-- Frame baseframe -->
      <site group="3" name="baseframe" pos="0 0 0" quat="1 0 0 0"/>
      
      <!-- Link shoulder -->
      <body name="shoulder" pos="0.0388 0 0.0624" quat="0 0 -1 0">
        <joint axis="0 0 1" name="shoulder_pan" type="hinge" range="-1.91986 1.91986" class="sts3215"/>
        <inertial pos="0 0 0.03" mass="0.1" diaginertia="0.00008 0.00008 0.00002"/>
        
        <!-- Shoulder motor -->
        <geom class="visual" pos="-0.03 0 -0.04" size="0.023 0.02" material="dark_gray" type="cylinder"/>
        <geom class="collision" pos="-0.03 0 -0.04" size="0.023 0.02" type="cylinder"/>
        
        <!-- Motor holder -->
        <geom class="visual" pos="-0.025 0 0.015" size="0.038 0.025 0.02" material="yellow" type="box"/>
        <geom class="collision" pos="-0.025 0 0.015" size="0.038 0.025 0.02" type="box"/>
        
        <!-- Rotation pitch -->
        <geom class="visual" pos="0.012 0 0.046" size="0.015 0.015 0.02" material="yellow" type="box"/>
        
        <!-- Link upper_arm -->
        <body name="upper_arm" pos="-0.0304 -0.0183 -0.0542" quat="1 -1 -1 -1">
          <joint axis="0 0 1" name="shoulder_lift" type="hinge" range="-1.74533 1.74533" class="sts3215"/>
          <inertial pos="0 0 0.07" mass="0.103" diaginertia="0.00004 0.00015 0.00014"/>
          
          <!-- Upper arm motor -->
          <geom class="visual" pos="-0.112 0 0.018" size="0.01 0.015 0.03" material="dark_gray" type="box"/>
          <geom class="collision" pos="-0.06 0 0.02" size="0.01 0.07 0.03" type="box"/>
          
          <!-- Upper arm link -->
          <geom class="visual" pos="-0.065 0.012 0.018" size="0.01 0.07" material="yellow" type="cylinder"/>
          <geom class="collision" pos="-0.12 -0.014 0.018" size="0.01 0.02 0.015" type="box"/>
          
          <!-- Link lower_arm -->
          <body name="lower_arm" pos="-0.1126 -0.028 0" quat="1 0 0 1">
            <joint axis="0 0 1" name="elbow_flex" type="hinge" range="-1.69 1.69" class="sts3215"/>
            <inertial pos="0 0 0.07" mass="0.104" diaginertia="0.00003 0.00016 0.00014"/>
            
            <!-- Lower arm motor -->
            <geom class="visual" pos="-0.065 -0.032 0.018" size="0.07 0.01 0.03" material="yellow" type="box"/>
            <geom class="collision" pos="-0.05 0 0.018" size="0.07 0.01 0.03" type="box"/>
            
            <!-- Wrist motor holder -->
            <geom class="visual" pos="-0.065 -0.032 0.018" size="0.023 0.013 0.018" material="yellow" type="box"/>
            <geom class="collision" pos="-0.125 0.005 0.018" size="0.023 0.013 0.018" type="box"/>
            
            <!-- Wrist motor -->
            <geom class="visual" pos="-0.122 0.005 0.018" size="0.02 0.015" material="dark_gray" type="cylinder"/>
            <geom class="collision" pos="0 -0.042 0.025" size="0.029 0.015 0.018" type="box"/>
            
            <!-- Wrist roll/pitch -->
            <geom class="visual" pos="0 -0.028 0.018" size="0.027 0.01 0.03" material="yellow" type="box"/>
            <geom class="collision" pos="0 -0.02 0.019" size="0.027 0.01 0.03" type="box"/>
            
            <!-- Link gripper -->
            <body name="gripper" pos="0 -0.061 0.018" quat="0.017 -0.017 0.707 0.707">
              <joint axis="0 0 1" name="wrist_flex" type="hinge" range="-1.65806 1.65806" class="sts3215"/>
              <inertial pos="0 0 0.02" mass="0.079" diaginertia="0.00004 0.00003 0.00002"/>
              
              <!-- Wrist motor -->
              <geom class="visual" pos="0 -0.042 0.03" size="0.02 0.015" material="dark_gray" type="cylinder"/>
              <geom class="collision" pos="0 -0.042 0.025" size="0.029 0.015 0.018" type="box"/>
              
              <!-- Fixed jaw -->
              <geom class="visual" pos="0 0 0" size="0.01 0.01 0.018" material="yellow" type="box" quat="0 1 0 0"/>
              <geom class="collision" pos="0 0 0" size="0.01 0.01 0.018" type="box" quat="0 1 0 0"/>
              
              <!-- Fixed jaw collision boxes -->
              <geom name="fixed_jaw_box1" class="collision_gripper" type="box" size="0.0325 0.015 0.015" pos="-0.0025 0 -0.022"/>
              <geom name="fixed_jaw_box2" class="collision_gripper" type="box" size="0.01 0.015 0.005" pos="-0.024 0 -0.04"/>
              <geom name="fixed_jaw_sph_tip1" class="collision_gripper" type="sphere" size="0.00075" pos="-0.0081 0 -0.101"/>
              <geom name="fixed_jaw_sph_tip2" class="collision_gripper" type="sphere" size="0.00075" pos="-0.0081 0.0035 -0.0975"/>
              <geom name="fixed_jaw_sph_tip3" class="collision_gripper" type="sphere" size="0.00075" pos="-0.0081 -0.0035 -0.0975"/>
              <geom name="fixed_jaw_box3" class="collision_gripper" type="capsule" size="0.0011 0.002" pos="-0.009 0 -0.103" euler="1.57 0 0"/>
              <geom name="fixed_jaw_box4" class="collision_gripper" type="box" size="0.001 0.004 0.004" pos="-0.009 0 -0.0982"/>
              <geom name="fixed_jaw_box5" class="collision_gripper" type="box" size="0.001 0.005 0.006" pos="-0.0108 0 -0.0905"/>
              <geom name="fixed_jaw_box6" class="collision_gripper" type="box" size="0.001 0.009 0.008" pos="-0.0125 0 -0.0727"/>
              <geom name="fixed_jaw_box7" class="collision_gripper" type="box" size="0.001 0.01 0.008" pos="-0.0143 0 -0.053"/>
              
              <!-- Frame gripperframe -->
              <site group="3" name="gripperframe" pos="0.012 0 -0.098" quat="1 0 1 0"/>
              
              <!-- Camera mount -->
              <body name="camera_mount">
                <camera name="wrist_cam" mode="fixed" pos="0 0.055 -0.045" euler="-0.57 0 0" resolution="1920 1080" sensorsize="0.00576 0.00324" focal="0.0036 0.0036"/>
                <geom class="visual" pos="0 0 0" size="0.015 0.015 0.003" material="yellow" type="box" quat="0 1 0 0"/>
                <!-- Camera collision boxes (disabled like original) -->
                <geom name="camera_box1" contype="0" conaffinity="0" type="box" size="0.015 0.015 0.003" pos="-0.0025 0.03 -0.03"/>
                <geom name="camera_box2" contype="0" conaffinity="0" type="box" size="0.021 0.021 0.003" pos="-0.001 0.06 -0.04" euler="-0.55 0 0"/>
              </body>
              
              <!-- Moving jaw -->
              <body name="moving_jaw" pos="0.02 0.019 -0.023" quat="1 1 0 0">
                <joint axis="0 0 1" name="wrist_roll" type="hinge" range="-2.74385 2.74385" class="sts3215"/>
                <inertial pos="0 0 0.02" mass="0.087" diaginertia="0.00003 0.00004 0.00003"/>
                
                <geom class="visual" pos="0 0 0.019" size="0.015 0.015 0.02" material="yellow" type="box" quat="1 0 0 0"/>
                
                <!-- Moving jaw collision -->
                <geom name="moving_jaw_box1" class="collision_gripper" type="box" size="0.01 0.01 0.015" pos="0 -0.013 0.019"/>
                <geom name="moving_jaw_sph_tip1" class="collision_gripper" type="sphere" size="0.00075" pos="-0.012 -0.078 0.019"/>
                <geom name="moving_jaw_sph_tip2" class="collision_gripper" type="sphere" size="0.00075" pos="-0.012 -0.0745 0.0225"/>
                <geom name="moving_jaw_sph_tip3" class="collision_gripper" type="sphere" size="0.00075" pos="-0.012 -0.0745 0.0155"/>
                <geom name="moving_jaw_box2" class="collision_gripper" type="box" size="0.001 0.004 0.004" pos="-0.011 -0.076 0.01875"/>
                <geom name="moving_jaw_box3" class="collision_gripper" type="box" size="0.001 0.005 0.006" pos="-0.009 -0.067 0.01875"/>
                
                <!-- Gripper finger joint -->
                <body name="finger" pos="0.01 0.02 -0.02">
                  <joint axis="0 0 1" name="gripper" type="hinge" range="-0.1745 1.74533" class="sts3215"/>
                  <inertial pos="0 0 0.01" mass="0.012" diaginertia="0.000006 0.000002 0.000005"/>
                  
                  <geom class="visual" pos="0 0 0.019" size="0.01 0.01 0.015" material="yellow" type="box"/>
                  <geom class="collision_gripper" type="box" size="0.005 0.008 0.02" pos="0 0 0.02"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position class="sts3215" name="shoulder_pan" joint="shoulder_pan" ctrlrange="-1.91986 1.91986"/>
    <position class="sts3215" name="shoulder_lift" joint="shoulder_lift" ctrlrange="-1.74533 1.74533"/>
    <position class="sts3215" name="elbow_flex" joint="elbow_flex" ctrlrange="-1.69 1.69"/>
    <position class="sts3215" name="wrist_flex" joint="wrist_flex" ctrlrange="-1.65806 1.65806"/>
    <position class="sts3215" name="wrist_roll" joint="wrist_roll" ctrlrange="-2.74385 2.84121"/>
    <position class="sts3215" name="gripper" joint="gripper" ctrlrange="-0.17453 1.74533"/>
  </actuator>
</mujoco>'''

    output_path = ASSETS_DIR / "so101_generated.xml"
    output_path.write_text(xml)
    print(f"Generated {output_path}")
    return output_path


def main():
    print("Generating SO-101 assets...")
    write_so101_xml()
    print("Done! Use assets/so101/so101_generated.xml instead of so101.xml")
    print("The original STL-based model is in assets/so101/so101.xml (gitignored)")


if __name__ == "__main__":
    main()