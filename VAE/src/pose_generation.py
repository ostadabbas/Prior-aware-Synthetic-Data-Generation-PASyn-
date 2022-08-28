import torch
import numpy as np
from os import path as osp
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

support_dir = './result/training'
expr_dir = osp.join(support_dir, '26')
# Loading VPoser Body Pose Prior
vp, ps = load_model(expr_dir, model_code=VPoser,
                    remove_words_in_model_weights='vp_model.',
                    disable_grad=True)
vp = vp.to('cuda')

# Sample a 32 dimensional vector from a Normal distribution
num_pose = 101
i = 0
total = 0
arry = np.zeros([num_pose, 36], dtype=np.float32)

while i < num_pose:
    num = np.random.normal(0., 2.0, size=(1, 16)).astype(np.float32)

    poZ_body_sample = torch.from_numpy(num).to('cuda')

    pose_body = vp.decode(poZ_body_sample)['pose_body'].contiguous().view(-1, 36)
    pose = pose_body.squeeze().tolist()

    if pose[0] < 100 and pose[0] > 40 and pose[3] < 0 and pose[3] > -125 and pose[6] < 100 and pose[6] > -25 and \
            pose[9] < 100 and pose[9] > 40 and pose[12] < 0 and pose[12] > -125 and pose[15] < 100 and pose[
        15] > -25 and \
            pose[18] < -60 and pose[18] > -120 and pose[21] < 80 and pose[21] > 0 and pose[24] < 0 and pose[
        24] > -125 and \
            pose[27] < -60 and pose[27] > -120 and pose[30] < 80 and pose[30] > 0 and pose[33] < 0 and pose[33] > -125:
        arry[i] = pose
        i += 1
    total += 1

dropout = i / total
print(dropout)
np.savez('./result/newpose100.npz', poses=arry)

