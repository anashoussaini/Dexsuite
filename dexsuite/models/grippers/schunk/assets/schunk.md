# Schunk SVH Hand - Genesis Joint Configuration

This document outlines the specific joint order loaded by the Genesis physics engine for the Schunk SVH hand URDF. The order below is the ground truth for controlling the hand in this simulation environment.

## Genesis Joint Order and DOF Indices

The engine loads 20 Degrees of Freedom (DOFs). The following table shows the DOF index assigned to each joint and specifies which ones are directly controllable versus which are mimic joints.

| DOF Index | Joint Name | Type |
| :---: | :--- | :---: |
| 0 | `right_hand_j5` | Mimic |
| 1 | `right_hand_Thumb_Opposition` | **Controlled** |
| 2 | `right_hand_index_spread` | Mimic |
| 3 | `right_hand_Middle_Finger_Proximal` | **Controlled** |
| 4 | `right_hand_ring_spread` | Mimic |
| 5 | `right_hand_Finger_Spread` | **Controlled** |
| 6 | `right_hand_Thumb_Flexion` | **Controlled** |
| 7 | `right_hand_Index_Finger_Proximal` | **Controlled** |
| 8 | `right_hand_Middle_Finger_Distal` | **Controlled** |
| 9 | `right_hand_Ring_Finger` | **Controlled** |
| 10 | `right_hand_Pinky` | **Controlled** |
| 11 | `right_hand_j3` | Mimic |
| 12 | `right_hand_Index_Finger_Distal` | **Controlled** |
| 13 | `right_hand_j15` | Mimic |
| 14 | `right_hand_j12` | Mimic |
| 15 | `right_hand_j13` | Mimic |
| 16 | `right_hand_j4` | Mimic |
| 17 | `right_hand_j14` | Mimic |
| 18 | `right_hand_j16` | Mimic |
| 19 | `right_hand_j17` | Mimic |

-----

## Final Controllable Indices

To command the hand, use the following list of indices. These 9 indices correspond to the **"Controlled"** joints in the table above.

```python
controllable_dof_indices = [1, 3, 5, 6, 7, 8, 9, 10, 12]
```
