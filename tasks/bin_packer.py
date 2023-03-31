import numpy as np
import os
import torch
from gym import spaces
import time

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

from typing import Dict

SPATIAL_DELTA = 1e-3
MOTION_DELTA = 1e-4

class BinPacker(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.max_episode_length = 100 #Arbitrarily big enough number (required for IssacGymEnv)

        #Duplicate - needed to allocate SKU info based spaces, prior to super.__init__() when self.num_envs becomes available
        #TODO: remove duplicate
        self.nEnvs = self.cfg["env"]["numEnvs"] 

        ###########################################################
        ## Boxes and SKU Info
        ###########################################################  
        self.nSKU = self.cfg["boxes"]["nSKU"]
        self.boxes = dict()
        for i in range(self.nSKU):
            self.boxes["SKU"+str(i+1)] = self.cfg["boxes"]["SKU"+str(i+1)]

        self.nBoxes = 0 #number of total boxes
        self.dimPerSKUs = {} #dimensions of each SKU 
        self.maxDims = [0.0, 0.0, 0.0] #max size of the dimensions
        self.queued = [{} for _ in range(self.nEnvs)] #boxes to be stacked
        self.nQueued = [0 for _ in range(self.nEnvs)] #number of boxes waiting to be stacked
        self.stacked = [[] for _ in range(self.nEnvs)] #boxes already stacked, recording indices
        self.xyAreas = [[] for _ in range(self.nEnvs)] #xyArea of each group // in version 1.0, this determines the stacking order (larger first)
        self.nBoxesPerSKUs = [] #number of boxes for each SKU
        self.currBoxIdx = [-1 for _ in range(self.nEnvs)] #index of currently handling boxes (used for state calculation & stacked record)
        self.InitSKUInfo()

        ###########################################################
        ## Pallet Info (3D space Info)
        ###########################################################
        self.x_pallet_min = self.cfg["pallet"]["x_min"]
        self.x_pallet_max = self.cfg["pallet"]["x_max"]
        self.y_pallet_min = self.cfg["pallet"]["y_min"]
        self.y_pallet_max = self.cfg["pallet"]["y_max"]
        self.z_pallet_min = self.cfg["pallet"]["z_min"]
        self.z_pallet_max = self.cfg["pallet"]["z_max"]

        ###########################################################
        ## RL observation space, state, action space definition
        ########################################################### 

        #Obs space range (-1, 1), action range(-1, 1)
        self.cfg["env"]["numObservations"] = 12*self.nBoxes
        self.cfg["env"]["numActions"] = 2

        self.env_boundaries = spaces.Box(low=np.array([self.x_pallet_min, self.y_pallet_min, self.z_pallet_min]), high=np.array([self.x_pallet_max, self.y_pallet_max, self.z_pallet_max]), dtype=np.float32)

        self.action_bias = [float(self.x_pallet_min + self.x_pallet_max)/2, float(self.y_pallet_min + self.y_pallet_max)/2]
        self.action_scale = [float(self.x_pallet_max - self.x_pallet_min), float(self.y_pallet_max - self.y_pallet_min)]

        ###########################################################
        ## Issac Gym set up
        ########################################################### 
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        _root_state_desc = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_states = gymtorch.wrap_tensor(_root_state_desc).view(self.num_envs, -1, 13)

        self._root_pos = self._root_states[..., 0:3]
        self._root_quat = self._root_states[..., 3:7]
        self._root_lin_vel = self._root_states[..., 7:10]
        self._root_ang_vel = self._root_states[..., 10:13]

        #For reward calculation
        self.totalStackedVol = torch.zeros(self.nEnvs, device=self.device)
        self.occupiedVol = torch.zeros(self.nEnvs, 6, device=self.device)
        self.occupiedVol[:, :3] = 1.0
        '''
        Addition
        '''
        self.stacked_z_vals = [{} for _ in range(self.nEnvs)]
        self.stacked_z_upper_pivot_coord = [{} for _ in range(self.nEnvs)]
        self.delta_gap = torch.tensor([1e-3/self.action_scale[0], 1e-3/self.action_scale[1]], device=self.device)
        self.stacked_z_max = torch.zeros(self.nEnvs, device=self.device)

        #Initial reset
        self.reset_idx(torch.arange(self.nEnvs, device=self.device))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.reset_buf[torch.arange(self.nEnvs, device=self.device)] = 0
        self.progress_buf[torch.arange(self.nEnvs, device=self.device)] = 0

        #Finally, override the the observation space with box [0, +1]
        self.obs_space = spaces.Box(low=0.0, high=1.0, shape=(12*self.nBoxes,))
        self.state_space = spaces.Box(low=0.0, high=1.0, shape=(12*self.nBoxes,))

    def InitSKUInfo(self):

        for idx, b_entry in enumerate(self.boxes):
            self.nBoxes += self.boxes[b_entry]["num"]
            self.nBoxesPerSKUs.append(self.boxes[b_entry]["num"])
            self.dimPerSKUs[b_entry] = self.boxes[b_entry]["dim"]
            self.maxDims[0] = max(self.maxDims[0], self.dimPerSKUs[b_entry][0])
            self.maxDims[1] = max(self.maxDims[1], self.dimPerSKUs[b_entry][1])
            self.maxDims[2] = max(self.maxDims[2], self.dimPerSKUs[b_entry][2])
            for n in range(self.nEnvs):
                self.queued[n][b_entry] = self.boxes[b_entry]["num"]
                self.nQueued[n] += self.boxes[b_entry]["num"]
                self.xyAreas[n].append([self.boxes[b_entry]["dim"][0]*self.boxes[b_entry]["dim"][1], idx])

        for n in range(self.nEnvs):
            self.xyAreas[n].sort(reverse=True)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        asset_file = "urdf/pallet/pallet.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        pallet_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        pallet_pose = gymapi.Transform()
        pallet_pose.p = gymapi.Vec3(2.0, 2.95, 0.0)
        pallet_pose.r = gymapi.Quat(np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        box_assets = []
        box_poses =[]
        self.initPositions = []
        for i, nBox in enumerate(self.nBoxesPerSKUs):

            color = np.clip(np.abs(np.random.normal(loc=0.5, scale=0.5, size=2)), 0.0, 1.0)
            #curr_SKU_color = gymapi.Vec3(i/self.nSKU, color[0], color[1])
            curr_SKU_size = [self.dimPerSKUs["SKU"+str(i+1)][0], self.dimPerSKUs["SKU"+str(i+1)][1], self.dimPerSKUs["SKU"+str(i+1)][2]]
            curr_box_opts = gymapi.AssetOptions()
            for b in range(nBox):
                box_asset = self.gym.create_box(self.sim, *(curr_SKU_size), curr_box_opts)
                curr_SKU_color = gymapi.Vec3(i/self.nSKU, color[0]+0.15*(b/nBox), color[1]+0.15*(b/nBox))
                box_assets.append([box_asset, curr_SKU_color])

                initPosition = [-spacing + curr_SKU_size[0]/2 + i*self.maxDims[0], -spacing + curr_SKU_size[1]/2, curr_SKU_size[2]/2 + b*curr_SKU_size[2]]
                self.initPositions.append(initPosition) 

                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(initPosition[0], initPosition[1], initPosition[2])               
                pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                box_poses.append(pose)

        self.envs = []
        self.handles = []
        for i in range(self.num_envs):
            #create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, 1+self.nBoxes, 1+self.nBoxes, True)
            pallet_actor = self.gym.create_actor(env_ptr, pallet_asset, pallet_pose, "pallet", i)
            
            self.handles.append(pallet_actor)

            for b in range(self.nBoxes):
                box = self.gym.create_actor(env_ptr, box_assets[b][0], box_poses[b], "box_"+str(b+1), i)
                self.gym.set_rigid_body_color(env_ptr, box, 0, gymapi.MESH_VISUAL, box_assets[b][1])

                self.envs.append(env_ptr)
                self.handles.append(box)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)   

        self._global_indices = torch.arange(self.num_envs * (self.nBoxes + 1), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)


    def compute_reward(self, valid_ids):
        '''
        The self.rew_buf is updated based on a stacked volume efficiency
        '''

        self.rew_buf[valid_ids] += compute_binpacking_reward(self.totalStackedVol, self.occupiedVol, valid_ids)

        return
    
    def compute_observations(self):
        '''
        Based on self.reset_buf, the self.obs_buf is updated accordingly.
            -> If self.reset_buf for a particular env is not zero, self.obs_buf is reset 
                [Final state's next trasition is masked. So update it to the init state] => Seems like IssacGym Env doesn't call reset outside of the step func
            -> Otherwise, self.obs_buf is updated with newly stacked positions of the currently handling item, and dimensions of a newly extracted item
        '''
        valid_ids = torch.arange(self.num_envs, device=self.device)
        invalid_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        mask = self.reset_buf.eq(0)
        valid_ids = torch.masked_select(valid_ids, mask)
        final_ids= []
        non_final_ids = []

        if len(invalid_ids) > 0:
            self.reset_idx(invalid_ids)

        if len(valid_ids) > 0:
            non_final_ids, final_ids = self.updateState(valid_ids)
            #self.compute_reward(valid_ids)
            
            #Turn the lists to tensors
            non_final_ids = torch.tensor(non_final_ids, device=self.device).to(dtype=torch.long)
            final_ids = torch.tensor(final_ids, device=self.device).to(dtype=torch.long)

            if(len(non_final_ids)>0):
                #deque for non_final environments
                self.extractItem(non_final_ids)
                self.rew_buf[non_final_ids] = 0.01

            if len(final_ids) > 0:
                self.compute_reward(final_ids)
                #self.rew_buf[final_ids] += 10.0*(1 - self.stacked_z_max[valid_ids])
                self.reset_idx(final_ids)
                self.reset_buf[final_ids] = 1.0

        return
    
    def updateState(self, valid_ids):

        final_ids= []
        non_final_ids = []
            
        #obs_buf update 
        currBoxIdx = torch.tensor(self.currBoxIdx, device=self.device)
        currBoxIdx = currBoxIdx[valid_ids]
        x = self._root_pos[valid_ids, 1+currBoxIdx, 0].clone()
        y = self._root_pos[valid_ids, 1+currBoxIdx, 1].clone()
        z = self._root_pos[valid_ids, 1+currBoxIdx, 2].clone()
        z_scale = (self.env_boundaries.high[2] - self.env_boundaries.low[2])

        x_min = x - self.action_scale[0]*self.obs_buf[valid_ids, -3]/2
        x_max = x + self.action_scale[0]*self.obs_buf[valid_ids, -3]/2
        y_min = y - self.action_scale[1]*self.obs_buf[valid_ids, -2]/2
        y_max = y + self.action_scale[1]*self.obs_buf[valid_ids, -2]/2
        z_min = z - z_scale*self.obs_buf[valid_ids, -1]/2
        z_max = z + z_scale*self.obs_buf[valid_ids, -1]/2
        
        self.obs_buf[valid_ids, 12*currBoxIdx + 0] = (x_min - self.env_boundaries.low[0])/self.action_scale[0]
        self.obs_buf[valid_ids, 12*currBoxIdx + 1] = (y_min - self.env_boundaries.low[1])/self.action_scale[1]
        self.obs_buf[valid_ids, 12*currBoxIdx + 2] = (z_min - self.env_boundaries.low[2])/z_scale

        self.obs_buf[valid_ids, 12*currBoxIdx + 6] = (x_max - self.env_boundaries.low[0])/self.action_scale[0]
        self.obs_buf[valid_ids, 12*currBoxIdx + 7] = (y_max - self.env_boundaries.low[1])/self.action_scale[1]
        self.obs_buf[valid_ids, 12*currBoxIdx + 8] = (z_max - self.env_boundaries.low[2])/z_scale
        self.stacked_z_max[valid_ids] = torch.max(self.stacked_z_max[valid_ids], self.obs_buf[valid_ids, 12*currBoxIdx + 8])
                      
        self.obs_buf[valid_ids, :] = torch.clamp(self.obs_buf[valid_ids, :], min=0.0, max=1.0) 
        self.rew_buf[valid_ids] = 1.0 - self.obs_buf[valid_ids, 12*currBoxIdx + 2] 

        #for reward calculation
        self.totalStackedVol[valid_ids] += (self.obs_buf[valid_ids, 12*currBoxIdx + 6] - self.obs_buf[valid_ids, 12*currBoxIdx + 0])*(self.obs_buf[valid_ids, 12*currBoxIdx + 7]-self.obs_buf[valid_ids, 12*currBoxIdx + 1])*(self.obs_buf[valid_ids, 12*currBoxIdx + 8]-self.obs_buf[valid_ids, 12*currBoxIdx + 2])
        self.occupiedVol[valid_ids, 0] = torch.min(torch.vstack([self.occupiedVol[valid_ids, 0], self.obs_buf[valid_ids, 12*currBoxIdx + 0]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 1] = torch.min(torch.vstack([self.occupiedVol[valid_ids, 1], self.obs_buf[valid_ids, 12*currBoxIdx + 1]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 2] = torch.min(torch.vstack([self.occupiedVol[valid_ids, 2], self.obs_buf[valid_ids, 12*currBoxIdx + 2]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 3] = torch.max(torch.vstack([self.occupiedVol[valid_ids, 3], self.obs_buf[valid_ids, 12*currBoxIdx + 6]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 4] = torch.max(torch.vstack([self.occupiedVol[valid_ids, 4], self.obs_buf[valid_ids, 12*currBoxIdx + 7]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 5] = torch.max(torch.vstack([self.occupiedVol[valid_ids, 5], self.obs_buf[valid_ids, 12*currBoxIdx + 8]]).t(), dim=1).values

        #stacked boxes recording
        ptr = 0
        for idx, stacked in enumerate(self.stacked):
            if(ptr==len(valid_ids)):
                break
            if(idx==valid_ids[ptr]):
                ptr += 1
                stacked.append(self.currBoxIdx[idx])

                if(self.nQueued[idx]>0):
                    non_final_ids.append(idx)
                else:
                    #final state
                    final_ids.append(idx)
                
                
                
                

        return non_final_ids, final_ids
    
    def extractItem(self, env_ids):
        '''
        Item extraction module:
        The idea is that based on the extraction policy (In version 1.0, it's simply bigger xy area first), the index of a box that is getting stacked is extracted

        The extracted box index is recorded to self.currBoxIdx,
        and the box's dimensions are recorded to self.obs_buf
        '''

        for idx in range(len(env_ids)):
            currSKU_idx = self.xyAreas[env_ids[idx]][0][1]
            b_entry = "SKU"+str(currSKU_idx+1)
            self.queued[env_ids[idx]][b_entry] -= 1
            self.nQueued[env_ids[idx]] -= 1
            if(self.queued[env_ids[idx]][b_entry]==0):
                self.xyAreas[env_ids[idx]].pop(0)

            currBoxIdx = 0
            for box_entry in self.boxes:
                if(box_entry==b_entry):
                    currBoxIdx += self.queued[env_ids[idx]][b_entry] #extracting the box from top
                    break
                else:
                    currBoxIdx += self.nBoxesPerSKUs[int(box_entry[-1])-1]

            #update the obs_buf with normalized curr box dims
            for b in range(self.nBoxes):
                self.obs_buf[env_ids[idx], 12*b + 3] = self.dimPerSKUs[b_entry][0]/(self.action_scale[0])
                self.obs_buf[env_ids[idx], 12*b + 4] = self.dimPerSKUs[b_entry][1]/(self.action_scale[1])
                self.obs_buf[env_ids[idx], 12*b + 5] = self.dimPerSKUs[b_entry][2]/(self.env_boundaries.high[2] - self.env_boundaries.low[2])

                self.obs_buf[env_ids[idx], 12*b + 9] = self.dimPerSKUs[b_entry][0]/(self.action_scale[0])
                self.obs_buf[env_ids[idx], 12*b + 10] = self.dimPerSKUs[b_entry][1]/(self.action_scale[0])
                self.obs_buf[env_ids[idx], 12*b + 11] = self.dimPerSKUs[b_entry][2]/(self.env_boundaries.high[2] - self.env_boundaries.low[2])
            #print(currBoxIdx)

            self.currBoxIdx[env_ids[idx]] = currBoxIdx

    def updateZcoord(self, x, y):
        '''
        Given the action values (x, y), find the valid z value
        The idea is to first assess if the 2D area (xy) required for the current box is already occupied or not.
        If it is, then find the maximum value of the z_max of the box(es) that occupies this area.
        The current item will be placed at max(z_max) + curr_box_z_dim

        Otherwise, z = curr_box_z_dim + self.env_boundaries.low[2]

        Returns
        ________
        Tensor [numEnv, ]: z value
        '''

        z_action_scale = (self.env_boundaries.high[2] - self.env_boundaries.low[2])
        z_max = torch.zeros(self.num_envs, device=self.device) + self.env_boundaries.low[2]
        for n in range(self.num_envs):
            curr_x_min = x[n] - self.action_scale[0]*self.obs_buf[n, -3]/2
            curr_x_max = x[n] + self.action_scale[0]*self.obs_buf[n, -3]/2
            curr_y_min = y[n] - self.action_scale[1]*self.obs_buf[n, -2]/2
            curr_y_max = y[n] + self.action_scale[1]*self.obs_buf[n, -2]/2

            for b in self.stacked[n]:
                stacked_x_min = self.obs_buf[n, 12*b + 0]*self.action_scale[0] + self.env_boundaries.low[0]
                stacked_x_max = self.obs_buf[n, 12*b + 6]*self.action_scale[0] + self.env_boundaries.low[0]
                stacked_y_min = self.obs_buf[n, 12*b + 1]*self.action_scale[1] + self.env_boundaries.low[1]
                stacked_y_max = self.obs_buf[n, 12*b + 7]*self.action_scale[1] + self.env_boundaries.low[1]

                x2 = min(curr_x_max, stacked_x_max)
                x1 = max(curr_x_min, stacked_x_min)
                y2 = min(curr_y_max, stacked_y_max)
                y1 = max(curr_y_min, stacked_y_min)

                if(x2-x1>0 and y2-y1>0):
                    stacked_z_max = self.obs_buf[n, 12*b+8]*(z_action_scale) + self.env_boundaries.low[2]
                    if(stacked_z_max > z_max[n]):
                        z_max[n] = stacked_z_max

            #final validity check
            if(z_max[n] > self.env_boundaries.high[2] - z_action_scale*self.obs_buf[n, -1]):
                #Place it at max valid z coord for current box in this case
                #This will create collision with the already stacked box(es), and isStaionary will return False
                z_max[n] = self.env_boundaries.high[2] - z_action_scale*self.obs_buf[n, -1]

        return z_max + z_action_scale*self.obs_buf[:, -1]/2
    
    def isNonStationary(self):
        '''
        This determins if the physics simulation creates any motion for currently handling item and/or stacked items.
        Based on self.prePosition that is assigned at each pre_sim call, it compares any position changes (due to collision, gravity, external forces, etc.)
        If the positional changes are above the certain threshold, that particular environment will be assigned "non stationary"
        And, based on the non-stationarity, self.reset_buf is updated for self.obs_buf and self.rew_buf calculations later on
        '''
        
        bNonStationary = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.bool)
        for n in range(self.num_envs):
            currNonStationary = False
            currIdx = self.prePositions[0][n] #single scalar
            compIdx = self.prePositions[1][n].copy()
            compIdx.append(currIdx)
            compIdx = 1 + torch.tensor(compIdx, device=self.device, dtype=torch.int64) #scalar array
            prevPosVectors = self.prePositions[2][n, compIdx] #[nBoxes, 3] vector
            currPosVectors = self._root_pos[n, compIdx] #[nBoxes, 3] vector

            diffPos = currPosVectors - prevPosVectors
            diffPos = torch.sqrt(torch.pow(diffPos[:, 0], 2) + torch.pow(diffPos[:, 1], 2) + torch.pow(diffPos[:, 2], 2)) > MOTION_DELTA
            bNonStationary[n] = torch.any(diffPos)
            
        self.reset_buf[bNonStationary] = 1.0
        self.rew_buf[bNonStationary] = -2.25


    def reset_idx(self, env_ids):
        #print("reset call", len(env_ids))

        ###########################################################
        ## Boxes and SKU Info reset
        ########################################################### 
        for idx in range(len(env_ids)):
            self.stacked[env_ids[idx]] = []
            self.nQueued[env_ids[idx]] = 0
            self.xyAreas[env_ids[idx]] = []
            self.stacked_z_vals[env_ids[idx]] = {}
            self.stacked_z_upper_pivot_coord [env_ids[idx]] = {}
            for jdx, b_entry in enumerate(self.boxes):
                self.queued[env_ids[idx]][b_entry] = self.boxes[b_entry]["num"]
                self.nQueued[env_ids[idx]] += self.boxes[b_entry]["num"]
                self.xyAreas[env_ids[idx]].append([self.boxes[b_entry]["dim"][0]*self.boxes[b_entry]["dim"][1], jdx])

            self.xyAreas[env_ids[idx]].sort(reverse=True)
            self.currBoxIdx[env_ids[idx]] = -1

        self.totalStackedVol[env_ids] = 0.0
        self.occupiedVol[env_ids] = torch.zeros(len(env_ids), 6, device=self.device)
        self.occupiedVol[env_ids, :3] = 1.0
        self.occupiedVol[env_ids, 3:] = 0.0
        self.stacked_z_max[env_ids] = 0.0
        ###########################################################
        ## RL observation space reset
        ########################################################### 
        state = torch.zeros(len(env_ids), self.observation_space.shape[0], device=self.device)
        self.obs_buf[env_ids, :] = state 

        #update currently handling box's dims in obs_buf, and update currBoxIdx
        self.extractItem(env_ids)

        currBoxIdx = torch.tensor(self.currBoxIdx, device=self.device)
        currBoxIdx = currBoxIdx[env_ids]

        ###########################################################
        ## Issac Gym reset
        ########################################################### 

        positions = torch.zeros(len(env_ids), self.nBoxes, 3, device=self.device)
        quats = torch.zeros(len(env_ids), self.nBoxes, 4, device=self.device)

        for b in range(self.nBoxes):

            positions[:, b, 0] = self.initPositions[b][0]
            positions[:, b, 1] = self.initPositions[b][1]
            positions[:, b, 2] = self.initPositions[b][2]
            quats[:, b, 3] = 1.0


        self._root_pos[env_ids, 1:, :] = positions
        self._root_quat[env_ids, 1:,  :] = quats
        self._root_lin_vel[env_ids, 1:, :] = torch.zeros(len(env_ids), self.nBoxes, 3, device=self.device)
        self._root_ang_vel[env_ids, 1:, :] = torch.zeros(len(env_ids), self.nBoxes, 3, device=self.device)

        env_ids_int32 = self._global_indices[env_ids, -self.nBoxes:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def pre_physics_step(self, actions):

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        x = actions[:, 0]
        y = actions[:, 1]
  
        #convert back to the original scale
        positions = torch.zeros(self.num_envs, 3, device=self.device)
        quats = torch.zeros(self.num_envs, 4, device=self.device)
        quats[:, -1] = 1.0

        positions[:, 0] = self.action_bias[0] + x*(self.action_scale[0] - self.obs_buf[:, -3]*self.action_scale[0])/2
        positions[:, 1] = self.action_bias[1] + y*(self.action_scale[1] - self.obs_buf[:, -2]*self.action_scale[1])/2
        positions[:, 2] = self.updateZcoord(positions[:, 0], positions[:, 1])

        #root state vectors update for simulation
        indices = 1 + torch.tensor(self.currBoxIdx, device=self.device)
        self._root_pos[:, indices, :] = positions
        self._root_quat[:, indices,  :] = quats
        self._root_lin_vel[:, indices, :] = torch.zeros(self.num_envs,  3, device=self.device)
        self._root_ang_vel[:, indices, :] = torch.zeros(self.num_envs, 3, device=self.device)

        #For motion validity check at post_physics_step
        self.prePositions = [self.currBoxIdx, self.stacked, self._root_pos.clone()]

        env_ids_int32 = self._global_indices[torch.arange(self.nEnvs, device=self.device), -self.nBoxes:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)

        return 
    
    def post_physics_step(self):
        self.progress_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.isNonStationary()

        self.compute_observations()

        # debug viz
        '''
        if self.viewer:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                for b in range(self.nBoxes):
                    pos = self.root_pos[i, b]
                    quat = self.root_quat[i, b]

                    px = (pos + quat_apply(quat, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos + quat_apply(quat, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos + quat_apply(quat, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos.cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
        '''

    
@torch.jit.script
def compute_binpacking_reward(total_stacked_vol, occupiedVol, valid_ids):
    '''
    This calculates the final reward when box(es) are stacked
    The reward will be assigned based on the overall volume efficiency of the current stacking
        
    Returns
    ________
    Tensor [numEnv (valid), ]: reward
    '''  

    curr_occupied_volume = torch.abs((occupiedVol[valid_ids,3] - occupiedVol[valid_ids,0])*(occupiedVol[valid_ids,4] - occupiedVol[valid_ids,1])*(occupiedVol[valid_ids,5] - occupiedVol[valid_ids,2]))
    reward = 10*total_stacked_vol[valid_ids]/curr_occupied_volume

    return reward
