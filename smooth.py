import copy
import numpy as np
from mmhuman3d.utils.demo_utils import smooth_process
#load data

data=np.load('/mimer/NOBACKUP/groups/snic2022-22-770/Tackle1_mono/seq1/0_seq1/0_seq1_cliff_hr48_infill_window64.npz', allow_pickle=True)
pose = data['pose'] # (N,72), "pose" in npz file
trans = data['global_t'] # (N,3), "global_t" in npz file

smooth_type = 'smoothnet_windowsize64'

# start from 0, the interval is 2
p0 = pose[::2]
t0 = trans[::2]
frame_num = p0.shape[0]
print(frame_num)
new_pose_0 = smooth_process(p0.reshape(frame_num,24,3),
            smooth_type='smoothnet_windowsize64',
            cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,72)
new_trans_0 = smooth_process(t0[:, np.newaxis],
             smooth_type='smoothnet_windowsize64',
             cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,3)

# start from 1, the interval is 2
p1 = pose[1::2]
t1 = trans[1::2]
frame_num = p1.shape[0]
new_pose_1 = smooth_process(p1.reshape(frame_num,24,3),
            smooth_type='smoothnet_windowsize64',
            cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,72)
new_trans_1 = smooth_process(t1[:, np.newaxis],
             smooth_type='smoothnet_windowsize64',
             cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,3)
new_pose = copy.copy(pose)
new_trans = copy.copy(trans)
new_pose[::2] = new_pose_0
new_pose[1::2] = new_pose_1
new_trans[::2] = new_trans_0
new_trans[1::2] = new_trans_1


# Save the new npz file with the updated pose and global_t, keeping other data unchanged
np.savez('/mimer/NOBACKUP/groups/snic2022-22-770/Tackle1_mono/seq1/0_seq1/smpl_window64.npz',
         imgname=data['imgname'],
         pose=new_pose,
         shape=data['shape'],
         global_t=new_trans,
         pred_joints=data['pred_joints'],
         focal_l=data['focal_l'],
         detection_all=data['detection_all'])