"Añadir las condiciones restantes"

import airsim
from airsim.utils import to_eularian_angles
from airsim.types import YawMode

import cv2
import numpy as np
from math import cos, sin, tanh, pi
import time

class BlockAvoidEnvBase(object):
    def __init__(self, ip='', camlist=[0]):
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()

        self.IMGTYPELIST = ['DepthPlanner', 'Segmentation']
        self.CAMLIST = camlist

        self.imgRequest = []
        for k in self.CAMLIST:
            for imgtype in self.IMGTYPELIST:
                if imgtype == 'DepthPlanner':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.DepthPlanner, True))

                elif imgtype == 'Segmentation':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Segmentation, False, False))

        self.starttime = time.time()
        self.lasttimeind = 0
        self.observation_space = np.zeros(2)
        self.imgwidth, self.imgheight = 0, 0

    def readimgs(self):
        responses = self.client.simGetImages(self.imgRequest)
        depthlist, seglist = [], []
        idx = 0
        for k in range(len(self.CAMLIST)):
            for imgtype in self.IMGTYPELIST:
                response = responses[idx]
                if response.height == 0 or response.width == 0:
                    print('Something wrong with image return.. {}'.format(idx))
                    return None, None

                if imgtype == 'DepthPlanner':
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    depthimg = img1d.reshape(response.height, response.width)
                    depthlist.append(depthimg)

                elif imgtype == 'Segmentation':
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                    img_rgba = img1d.reshape(response.height, response.width, -1)
                    img_seg = img_rgba[:, :, 0]
                    seglist.append(img_seg)
                idx += 1

        self.imgwidth = response.width
        self.imgheight = response.height

        return depthlist, seglist

    def world2drone(self, vx, vy, yaw):
        vvx = vx * cos(yaw) - vy * sin(yaw)
        vvy = vx * sin(yaw) + vy * cos(yaw)

        return vvx, vvy

    def readKinematics(self, dronestate):
        kinematics_estimated = dronestate.kinematics_estimated
        drone_pos = kinematics_estimated.position
        drone_quat = kinematics_estimated.orientation
        world_drone_vel = kinematics_estimated.linear_velocity
        (pitch, roll, yaw) = to_eularian_angles(drone_quat)

        drone_vel_x, drone_vel_y = self.world2drone(world_drone_vel.x_val, world_drone_vel.y_val, yaw)

        return drone_pos.x_val, drone_pos.y_val, drone_pos.z_val, \
               pitch, roll, yaw, \
               drone_vel_x, drone_vel_y, world_drone_vel.z_val

    def reset(self):
        pass

    def step(self, action):
        yaw_change = 0.2 * pi if action == 0 else -0.2 * pi
        self.client.moveByVelocityAsync(0, 0, 0, 1, yaw_mode=YawMode(is_rate=True, yaw_or_rate=yaw_change))

        time.sleep(0.1)

        x, y, z, vx, vy, vz = self.read_kinematics()
        seglist = self.read_imgs()

        # Verifica si hay un bloque en cualquier parte de la imagen
        block_id = 1
        block_in_image = any(block_id in seg for seg in seglist)

        done = z < 0.1  # Condición de terminación, ajusta según sea necesario
        reward = -10.0 if block_in_image else 1.0

        return [vx, vy], reward, done

        

    def render(self):
        pass

    def close(self):
        self.client.simPause(False)
        self.client.reset()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

class BlockAvoidDiscreteOneDirEnv(BlockAvoidEnvBase):
    def __init__(self, period=0.1, camlist=[0], strictsafe=False):
        super(BlockAvoidDiscreteOneDirEnv, self).__init__(camlist=camlist)
        self.client.reset()
        time.sleep(1)
        dronestate = self.client.getMultirotorState()
        _, _, self.iniHeight, _, _, self.iniYaw, _, _, _ = self.readKinematics(dronestate)

        self.targetHeight = -1.0 + self.iniHeight
        self.heightThresh = 0.1

        self.setSegmentation()
        self.blockIdx = 1  # Change to the appropriate ID for the blocks in the segmentation image
        self.bgIdx = 0

        self.x, self.y, self.z = 0, 0, 0
        self.vx, self.vy, self.vz = 0, 0, 0
        self.yaw = 0
        self.last_x = 0
        self.depthimg, self.segimg = None, None

        self.stepCount = 0
        self.episodeLen = 1000
        self.period = period

        self.x_axis_positive = 1
        self.strictsafe = strictsafe

        self.rewardCrash = -10.0
        self.rewardSafe = 1.0
        self.rewardDist = 0.01
        self.safeDist = 1.5

        self.client.simPause(False)
        self.client.enableApiControl(True)

    def setSegmentation(self):
        self.client.simSetSegmentationObjectID('Block', self.blockIdx, True)
        self.client.simSetSegmentationObjectID('Background', self.bgIdx, True)

    def computeReward(self, dronestate, done):
        if done:
            return self.rewardCrash

        if self.strictsafe and self.block_idx_in_center():
            return self.rewardCrash

        dist = ((self.x - self.last_x) ** 2 + (self.y) ** 2) ** 0.5

        if dist < self.safeDist:
            return self.rewardSafe

        return self.rewardDist

    def getObservation(self):
        return np.array([self.vx, self.vy])

    def isDone(self):
        if self.z < self.targetHeight + self.heightThresh and self.z > self.targetHeight - self.heightThresh:
            return True

        if self.stepCount > self.episodeLen:
            return True

        return False

    def reset(self):
        self.client.reset()
        time.sleep(1)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simPause(False)

        self.stepCount = 0

        self.vx, self.vy, self.vz = 0, 0, 0
        self.yaw = self.iniYaw

        self.last_x = self.x
        self.x, self.y, self.z = 0, 0, 0

        self.client.moveToPosition(self.x, self.y, self.z, 1)

        self.depthimg, self.segimg = self.readimgs()
        return self.getObservation()

    def step(self, action):
        self.stepCount += 1
        if action == 0:
            self.yaw += 0.2 * pi
        elif action == 1:
            self.yaw -= 0.2 * pi

        self.client.moveToPosition(self.x, self.y, self.z, 1, yaw_mode=YawMode(is_rate=False, yaw_or_rate=self.yaw))

        time.sleep(self.period)

        self.last_x = self.x
        self.x, self.y, self.z, _, _, _, self.vx, self.vy, self.vz = self.readKinematics(self.client.getMultirotorState())

        self.depthimg, self.segimg = self.readimgs()

        done = self.isDone()
        reward = self.computeReward(self.client.getMultirotorState(), done)
        observation = self.getObservation()

        return observation, reward, done, None

    def block_idx_in_center(self):
        # Check if the block is in the center of the image
        center_x = self.imgwidth // 2
        center_y = self.imgheight // 2

        return self.segimg[center_y, center_x] == self.blockIdx


if __name__ == '__main__':
    env = BlockAvoidDiscreteOneDirEnv()
    obs = env.reset()

    for _ in range(100):
        action = np.random.randint(2)
        obs, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        env.render()
        if done:
            obs = env.reset()

    env.close()
