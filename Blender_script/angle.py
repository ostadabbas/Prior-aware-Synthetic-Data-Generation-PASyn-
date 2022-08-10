import bpy
import numpy as np
from numpy.linalg import inv
import math
import os

scene = bpy.context.scene
armature = bpy.data.objects['Arm_Deer']


def bone_matrix_3x3(bone):
    result = np.eye(3)
    for i in range(0, 2):
        result[i][i] = bone.matrix_basis[i][i]
    return result


def decomposeRotMatrix(rot):
    thetaX = math.atan2(rot[2][1], rot[2][2])
    thetaY = math.atan2(-1 * rot[2][0], math.sqrt(math.pow(rot[2][1], 2) + math.pow(rot[2][2], 2)))
    thetaZ = math.atan2(rot[1][0], rot[0][0])
    return math.degrees(thetaX), math.degrees(thetaY), math.degrees(thetaZ)


def vectorAlignmentRotMatrix(v1, v2):
    a, b = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0, -1 * v[2], v[1]], [v[2], 0, -1 * v[0]], [-1 * v[1], v[0], 0]])
    result = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    return result


relevant_joints = ['hip_1_f.L and hip_2_f.L',
                   'hip_2_f.L and thigh_f.L',
                   'thigh_f.L and shin_f.L',
                   'hip_1_f.R and hip_2_f.R',
                   'hip_2_f.R and thigh_f.R',
                   'thigh_f.R and shin_f.R',
                   'Spine_05 and hip_b.L',
                   'hip_b.L and thigh_b.L',
                   'thigh_b.L and shin_b.L',
                   'Spine_05 and hip_b.R',
                   'hip_b.R and thigh_b.R',
                   'thigh_b.R and shin_b.R']

path = 'D:\\Angles1.npz'
poses = []
if os.path.isfile(path):
    current = np.load('D:\\Angles1.npz')
    poses = current['poses'].tolist()

cycles = 50

with open('D:\\Angles1.txt', 'w') as file:
    frameLength = scene.frame_end - scene.frame_start
    currentFrame = 0
    while currentFrame < cycles:
        scene.frame_set(currentFrame % frameLength)
        file.write("#####################\nFrame" + str(currentFrame % frameLength) + "\n")

        frame_triples = [0, 0, 0]
        for pbone in armature.pose.bones:
            if pbone.parent is not None:

                v1 = pbone.parent.vector
                v2 = pbone.vector
                if pbone.name == 'hip_b.L':
                    v1 = [0, 1, 0]
                if pbone.name == 'hip_b.R':
                    v1 = [0, 1, 0]
                rot = vectorAlignmentRotMatrix(v1, v2)
                x, y, z = decomposeRotMatrix(rot)
                stat = "between " + str(pbone.parent.name) + " and " + str(pbone.name) + "\n"
                xT = "x: " + str(x) + "\n"
                yT = "y: " + str(y) + "\n"
                zT = "z: " + str(z) + "\n"
                line = stat + xT + yT + zT
                print(str(pbone.parent.name) + " and " + str(pbone.name))
                if (str(pbone.parent.name) + " and " + str(pbone.name)) in relevant_joints:
                    frame_triples.append(x)
                    frame_triples.append(y)
                    frame_triples.append(z)
                    file.write("%s\n" % line)
        poses.append(frame_triples)
        currentFrame += 1

print(len(poses))
np.savez('D:\\Angles1.npz', poses=poses)