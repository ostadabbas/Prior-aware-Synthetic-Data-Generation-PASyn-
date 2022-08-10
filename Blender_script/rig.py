import bpy
import bpy_extras
import pdb
import os
import numpy as np
from numpy.linalg import inv
import math
import mathutils
import random

# partsList for limb part iteration
partsList = ['neck2', 'torso', 'shoulder.R', 'upperleg.R', 'frontleg.R', 'forearm.R', 'shoulder.L', 'upperleg.L',
             'frontleg.L', 'forearm.L', 'pelvis.R', 'thigh.R', 'shin.R', 'backfoot.R', 'pelvis.L', 'thigh.L', 'shin.L',
             'backfoot.L', 'tail', 'neck', 'head', 'lefteye', 'righteye', 'frontleg2.R', 'frontleg2.L', 'hip.R',
             'hip.L']
partsList_parent = [1, None, 0, 2, 3, 4, 0, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 0, 19, None, None, None, None,
                    None, None]  # corresponding parent bones
# how many synthetic images will be generated
numFrames = 100

# from VAE
data = np.load("D:\\newpose.npz")

priors = data['poses'].tolist()

# which bones need random, which ones need prior
isRandom = [True, True, True, False, False, False, True, False, False, False, True, False,
            False, False, True, False, False, False, True, True, True, True, True, True, True, True, True]

average_pose = [0, 0, 0, 75, -60, 25, 0, 75, -60, 20, 0, -95, 40, -50, 0, -95, 40, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0]

bone_prior_indices = dict()

bone_prior_indices['upperleg.L'] = [0, 1, 2]
bone_prior_indices['frontleg.L'] = [3, 4, 5]
bone_prior_indices['forearm.L'] = [6, 7, 8]
bone_prior_indices['upperleg.R'] = [9, 10, 11]
bone_prior_indices['frontleg.R'] = [12, 13, 14]
bone_prior_indices['forearm.R'] = [15, 16, 17]
bone_prior_indices['thigh.L'] = [18, 19, 20]
bone_prior_indices['shin.L'] = [21, 22, 23]
bone_prior_indices['backfoot.L'] = [24, 25, 26]
bone_prior_indices['thigh.R'] = [27, 28, 29]
bone_prior_indices['shin.R'] = [30, 31, 32]
bone_prior_indices['backfoot.R'] = [33, 34, 35]


def rotate2DX(theta):
    theta = math.radians(theta)
    return np.array([[1, 0, 0], [0, math.cos(theta), -1 * math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])


# y rotation matrix given input in degrees
def rotate2DY(theta):
    theta = math.radians(theta)
    return np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-1 * math.sin(theta), 0, math.cos(theta)]])


# z rotation matrix given input in degrees
def rotate2DZ(theta):
    theta = math.radians(theta)
    return np.array([[math.cos(theta), -1 * math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])


# xyz rotation matrix given x,y,z rotation in degrees
def rotate3D(xTheta, yTheta, zTheta):
    xRot = rotate2DX(xTheta)
    yRot = rotate2DY(yTheta)
    zRot = rotate2DZ(zTheta)
    return xRot @ yRot @ zRot


# convert a rotation matrix o x, y, x angles in degrees
def decomposeRotMatrix(rot):
    thetaX = math.atan2(rot[2][1], rot[2][2])
    thetaY = math.atan2(-1 * rot[2][0], math.sqrt(math.pow(rot[2][1], 2) + math.pow(rot[2][2], 2)))
    thetaZ = math.atan2(rot[1][0], rot[0][0])
    return math.degrees(thetaX), math.degrees(thetaY), math.degrees(thetaZ)


# rotation matrix to align vector v1 to vector v2
def vectorAlignmentRotMatrix(v1, v2):
    a, b = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0, -1 * v[2], v[1]], [v[2], 0, -1 * v[0]], [-1 * v[1], v[0], 0]])
    result = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    return result


# not used
def getOffsetRotMatrix(bone, parent, newX, newY, newZ):
    v2 = bone.vector
    v1 = parent.vector
    rot = vectorAlignmentRotMatrix(v1, v2)
    oldX, oldY, oldZ = decomposeRotMatrix(rot)
    diffX, diffY, diffZ = newX - oldX, newY - oldY, newZ - oldZ
    return diffX, diffY, diffZ


# Gaussian noise for x rotation for given bone
def getXNoise(i):
    ranges = [[-20, 10], [0, 0], [0, 0], [-40, 30], [-40, 30], [-30, 50], [-10, 10],
              [-40, 30], [-40, 30], [-30, 50], [0, 0], [-40, 40], [-50, 50], [-50, 50], [0, 0], [-40, 40], [-50, 50],
              [-50, 50], [-30, 30], [-30, 20], [-30, 30], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    # ranges = [[0, 0], [0,0], [0,0], [0, 0], [0,0], [0,0], [0,0],
    # [0,0], [0,0], [0,0], [0,0], [0, 0], [0,0], [0,0], [0,0],[0, 0], [0,0], [0,0], [0,0], [0,0],[0,0],[0,0],[0,0]]

    lowerB = ranges[i][0]
    upperB = ranges[i][1]
    return random.randint(lowerB, upperB)


# Gaussian noise for y rotation for given bone
def getYNoise(i):
    ranges = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
              [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-20, 20],
              [-20, -20], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    lowerB = ranges[i][0]
    upperB = ranges[i][1]
    return random.randint(lowerB, upperB)


# Gaussian noise for z rotation for given bone
def getZNoise(i, x, y):
    ranges = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
              [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-30, 30],
              [-10, 10], [-40, 40], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    lowerB = ranges[i][0]
    upperB = ranges[i][1]
    return random.randint(lowerB, upperB)


# based on the bone index and frame number, retrieves the corresponding x y z angles from the VAE in degrees
def getPrior(frameNumber, boneIndex):
    whichFrame = priors[frameNumber]
    bone_name = partsList[boneIndex]
    triple_indices = bone_prior_indices[bone_name]
    ix, iy, iz = triple_indices[0], triple_indices[1], triple_indices[2]
    return whichFrame[ix], whichFrame[iy], whichFrame[iz]


# converts a bone vector to world coordinates
def getWorldCoordinates(poseBone, armature, which):
    world_location = ''
    if which == "tail":
        world_location = armature.matrix_world @ poseBone.tail
    if which == "head":
        world_location = armature.matrix_world @ poseBone.head
    loc = np.array(world_location)
    return mathutils.Vector([loc[0], loc[1], loc[2]])


# the h5 order for joint coordinates in one frame
def setupH5Order():
    h5order = dict()
    h5order['head'] = None  # 1
    h5order['neck2'] = None  # 2
    # h5order['torso'] = None #3
    h5order['frontleg2.R'] = None  # 4
    h5order['frontleg.R'] = None  # 5
    h5order['forearm.R'] = None  # 6
    h5order['frontleg2.L'] = None  # 7
    h5order['frontleg.L'] = None  # 8
    h5order['forearm.L'] = None  # 9
    h5order['pelvis'] = None  # 10
    h5order['hip.R'] = None  # 11
    h5order['shin.R'] = None  # 12
    h5order['backfoot.R'] = None  # 13
    h5order['hip.L'] = None  # 14
    h5order['shin.L'] = None  # 15
    h5order['backfoot.L'] = None  # 16
    # h5order['tail'] = None #17
    h5order['lefteye'] = None  # 18
    h5order['righteye'] = None  # 19
    return h5order


if __name__ == '__main__':
    # set up camera
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale))
    cam = bpy.data.objects["Camera"]
    # what bones we want to rotate (this depends on your specific armature)
    bone_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    armature = bpy.data.objects["metarig"]
    RR = np.zeros((3, 3, 27))
    # all bones start with no rotation
    for i in range(len(partsList)):
        RR[:, :, i] = np.eye(3)
    coordinates_2D_across_frames = []

    # do this for every frame
    for idFrame in range(0, numFrames + 1):

        h5order = setupH5Order()

        scene.frame_set(idFrame)

        # record the pixel coordinates as tuples for each bone in this frame
        coordinates2D_frame_i = []

        for i in bone_indices:
            # the bone in question
            boneName = partsList[i]
            poseBone = armature.pose.bones[boneName]
            # not the torso, which does not rotate
            if partsList_parent[i] is not None:
                poseRot = np.eye(3)
                # random rotation
                if isRandom[i]:
                    # bone is rotated by Gaussian noise with parameters
                    noiseX = int(getXNoise(i))
                    noiseY = getYNoise(i)
                    noiseZ = getZNoise(i, noiseX, noiseY)
                    rotRand3D = rotate3D(noiseX, noiseY, noiseZ)

                    RR[:, :, i] = rotRand3D

                # uses prior angles from VAE
                else:
                    prior_x, prior_y, prior_z = getPrior(idFrame, i)

                    diffX = prior_x - average_pose[i]
                    tmp[i] = diffX
                    if i % 4 == 3:
                        rotPrior = rotate3D(diffX, 0, 0)
                    elif i % 4 == 0:
                        rotPrior = rotate3D(diffX + tmp[i - 1], 0, 0)
                    else:
                        rotPrior = rotate3D(diffX + tmp[i - 1] + tmp[i - 2], 0, 0)
                    RR[:, :, i] = rotPrior

                poseRot = inv(RR[:, :, partsList_parent[i]]) @ RR[:, :, i]
                Rot = np.eye(4)
                Rot[:3, :3] = poseRot
                poseBone.matrix_basis = Rot.transpose()

            # no rotation for this bone, since it has no parent; we only
            # handle the special case for the torso here
            else:
                # for the torso the head is the pelvis
                if partsList[i] == 'torso':
                    wco_head = getWorldCoordinates(poseBone, armature, "head")
                    cco_head = bpy_extras.object_utils.world_to_camera_view(scene, cam, wco_head)
                    xhead = round(cco_head.x * render_size[0])
                    yhead = render_size[1] - round(cco_head.y * render_size[1])
                    h5order['pelvis'] = (xhead, yhead)

            # set key frame for animation
            blSet = poseBone.keyframe_insert('rotation_quaternion', frame=idFrame)

            # save the joint coordinates for this frame in h5Order
            wco = getWorldCoordinates(poseBone, armature, "tail")
            cco = bpy_extras.object_utils.world_to_camera_view(scene, cam, wco)
            x = round(cco.x * render_size[0])
            y = render_size[1] - round(cco.y * render_size[1])
            if boneName in h5order:
                h5order[boneName] = (x, y)

        # coordinates are ahead by 1 frame so skip frame 0
        if idFrame > 0:
            print(h5order)
            for h5joint in h5order:
                coordinates2D_frame_i.append(h5order[h5joint])
            coordinates_2D_across_frames.append(coordinates2D_frame_i)

    # open output file for writing
    with open('D:\\zebra_2Dcoordinates.txt', 'w') as f:
        for frameCoords in coordinates_2D_across_frames:
            frame_line = ""
            for tuple in frameCoords:
                frame_line += "h" + str(tuple[0]) + " " + str(tuple[1])
            f.write("%s\n" % frame_line)












