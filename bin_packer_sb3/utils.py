import math
import numpy as np
import xml.etree.ElementTree as ET
import gym
import torch

ENV_ID = "Bin-packer"

def euler_from_quaternion(x, y, z, w):
        """
        Converts a quaternion into euler angles
        rx, ry, rz (counterclockwise in degrees)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        rx = 180*roll_x/np.pi
        ry = 180*pitch_y/np.pi
        rz = 180*yaw_z/np.pi
     
        return rx, ry, rz

def quaternion_from_euler(rx, ry, rz):
    """
        Converts euler angles into a quaternion
        qx, qy, qz, qw
    """
    
    roll = np.pi*rx/180
    pitch = np.pi*ry/180
    yaw = np.pi*rz/180

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return qx, qy, qz, qw


def find_index(queued, val):
    """
    Simple returns the index of the val in the queued (list)
    """

    for i in range(len(queued)):
        if(queued[i]==val):
            return i
    
    return -1

def create_box_xml(worldbody, nObj, SKU_size, SKU_rgba, SKU_mass, init_pos):
    """
        Creates xml script for a particular group of boxes
        
        worldbody = xml subfield to which the current creation of body belong
        nObj = number of objects pertaining to the current group
        SKU_size = string value of current group's x, y, z dims
        SKU_rgba = string value of current group's r, g, b, a values
        SKU_mass = string value of current group's mass
        init_pos = string value of initial position of the current group

        In XML, objects in the same group is positioned in ascending z (height) with increasing index
        e.g., z = 1.1 (at index 0), z = 3.1 (at index 1), z = 5.1 (at index 2)
        Z is written in ascending order with the increasing index so when last index is pulled out for stacking on a pallet, this translates to picking the topmost box
    """
    
    for b in range(nObj):
        body = ET.SubElement(worldbody, 'body')
        body_joint = ET.SubElement(body, 'joint')
        body_geom = ET.SubElement(body, 'geom')

        dims = SKU_size.split(' ')
        init_p = init_pos.split(' ')
        pos = str(float(init_p[0])) +' '+ str(float(init_p[1])) +' '+ str(0.1+float(dims[2])+2*float(dims[2])*(b))
        body.set('pos', pos)
        body_joint.set('type', 'free')
        body_geom.set('type', 'box')
        body_geom.set('size', SKU_size)
        body_geom.set('rgba', SKU_rgba)
        body_geom.set('mass', SKU_mass)

    return

def make_env(xml_path, pallet_x_min, pallet_x_max, pallet_y_min, pallet_y_max, boxes):
    env = gym.make(ENV_ID, xml_path=xml_path, pallet_x_min=pallet_x_min, pallet_x_max=pallet_x_max, pallet_y_min=pallet_y_min, pallet_y_max=pallet_y_max, boxes=boxes)

    #if OBS_HISTORY_STEPS > 1:
    #    env = ptan.common.wrappers_simple.FrameStack1D(env, OBS_HISTORY_STEPS)
    return env


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)