import gym
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import time

from mujoco_base import MuJoCoBase


SPATIAL_DELTA = 1e-3
MOTION_DELTA = 3e-3
STEP_SIM_TIME = 1e-1

class BinPacker(MuJoCoBase):
    def __init__(self, xml_path, boxes, pallet_x_min, pallet_x_max, pallet_y_min, pallet_y_max, pallet_z_min = 0.1, pallet_z_max = 10.1):
        super().__init__(xml_path)

        self.boxes = boxes #dict containing number of boxes for each SKUs (e.g., {0: 2, 1: 3, ...} )
        self.nBoxes = 0 #number of total boxes
        self.nSKUs = len(boxes) #number of different SKUs (groups)
        self.queued = {} #boxes to be stacked
        self.nQueued = 0 #number of boxes waiting to be stacked
        self.stacked = [] #boxes already stacked, recording index (geom idx of MuJoCo)
        self.xyAreas = [] #xyArea of each group //in ver 1.0, this determines the stacking order (larger first)
        self.nBoxesPerSKUs = [] #number of boxes for each SKU -> used for qpos idx calc.
        self.totalReward = 0.0
        self.currBoxIdx = -1
        #self.masses = self.model.body_mass.copy()
        
        temp = 0
        for i in boxes:
            self.nBoxes += boxes[i]
            self.queued[i] = boxes[i]
            self.nQueued += boxes[i]
            self.xyAreas.append([(2*self.model.geom_size[2+temp][0])*(2*self.model.geom_size[2+temp][1]), i])
            temp += boxes[i]
            self.nBoxesPerSKUs.append(boxes[i])

        self.xyAreas.sort(reverse=True)

        msg = "Created xml file has " + str(int(len(self.data.qpos)/7)) + " boxes and the input dict contains "+str(self.nBoxes)+" boxes"
        assert(len(self.data.qpos)/7 == float(self.nBoxes)), msg
        
        #State is represented as 1D list of
        #[box_x_min, box_y_min, box_z_min, curr_box_x_dim, curr_box_y_dim, curr_box_z_dim,
        # box_x_max, box_y_max, box_z_max, curr_box_x_dim, curr_box_y_dim, curr_box_z_dim,...
        # for all stacked boxes (if not stacked, they should be all set to 0) ]
        # Conv1D with kernal size = 6, stride = 6 and convolve stacked (x,y,z) min dims with curr (x,y,z) min dims, etc.
        # box dims and curr_box dims are normalized with respect to the pallet size, which represent the relative size
        # box dims contain min/max values which are used to represent coordinate values
        # curr_box dims contain only the relative size with respect to the pallet
        self.state = np.array([0 for i in range(12*self.nBoxes)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.state.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.env_boundaries = gym.spaces.Box(low=np.array([pallet_x_min, pallet_y_min, pallet_z_min], dtype=np.float32), high = np.array([pallet_x_max, pallet_y_max, pallet_z_max], dtype=np.float32))
        
        self.action_bias = [float(pallet_x_min + pallet_x_max)/2, float(pallet_y_min + pallet_y_max)/2]
        self.action_scale = [float(pallet_x_max - pallet_x_min), float(pallet_y_max - pallet_y_min)]
        
        self.stacked_z_max = 0.0
        
    def reset(self):
        # Set back to the original state
        mj.mj_resetData(self.model, self.data)
        self.stacked = []
        self.nQueued = 0
        self.xyAreas = []
        temp = 0
        for i in self.boxes:
            self.queued[i] = self.boxes[i]
            self.nQueued += self.boxes[i]
            self.xyAreas.append([(2*self.model.geom_size[2+temp][0])*(2*self.model.geom_size[2+temp][1]), i])
            temp += self.boxes[i]
        
        self.xyAreas.sort(reverse=True)

        #state info reset
        self.state = np.array([0 for i in range(12*self.nBoxes)], dtype=np.float32)
        self.stacked_z_max = 0.0
        
        self.currBoxIdx = -1
        #picking the very first item (currBoxIdx and the state update)
        self.currBoxIdx = self.extractItem()
        
        #state is rounded to 3 decimal points //for learning stability, convergence speed
        #self.state = np.abs(np.round(self.state, 1))

        obs, _, _, _ = self.step((-1,-1))

        return obs
    
    def extractItem(self):
        currSKU = self.xyAreas[0][1]
        self.queued[currSKU] -= 1
        self.nQueued -= 1
        if(self.queued[currSKU]==0):
            self.xyAreas.pop(0)
        
        #mujoco qpos idx
        currBoxIdx = 0
        for i in self.boxes:
            if(i == currSKU):
                currBoxIdx += self.queued[i] #extracting the box from top
                break
            else:
                currBoxIdx += self.nBoxesPerSKUs[i]

        #Update the state with normalized curr box dims
        for i in range(self.nBoxes):
            self.state[12*i + 3] = 2*self.model.geom_size[2+currBoxIdx][0]/(self.action_scale[0])
            self.state[12*i + 4] = 2*self.model.geom_size[2+currBoxIdx][1]/(self.action_scale[1])
            self.state[12*i + 5] = 2*self.model.geom_size[2+currBoxIdx][2]/(self.env_boundaries.high[2] - self.env_boundaries.low[2])
            
            #duplicates
            self.state[12*i + 9] = 2*self.model.geom_size[2+currBoxIdx][0]/(self.action_scale[0])
            self.state[12*i + 10] = 2*self.model.geom_size[2+currBoxIdx][1]/(self.action_scale[1])
            self.state[12*i + 11] = 2*self.model.geom_size[2+currBoxIdx][2]/(self.env_boundaries.high[2] - self.env_boundaries.low[2])
        
        return currBoxIdx
    
    def updateZcoord(self, x, y):
        '''
        Given the action values (x, y), find the valid z value
        The idea is to first assess if the 2D area (xy) required for the current box is already occupied
        If it is, then find the maximum value of the z_max of the box(es) that occupies this area
        The current item will be placed at max(z_max) + curr_box_z_dim
        
        Otherwise, z = curr_box_z_dim

        Returns
        -------
        z value
        '''
        curr_x_min = x - self.action_scale[0]*self.state[-3]/2
        curr_x_max = x + self.action_scale[0]*self.state[-3]/2
        curr_y_min = y - self.action_scale[1]*self.state[-2]/2
        curr_y_max = y + self.action_scale[1]*self.state[-2]/2
        
        #print("update z coord curr:",curr_x_min, curr_x_max, curr_y_min, curr_y_max)
        
        z_max = self.env_boundaries.low[2]
        for b in self.stacked:
            stacked_x_min = self.state[12*b + 0]*self.action_scale[0] + self.env_boundaries.low[0]
            stacked_x_max = self.state[12*b + 6]*self.action_scale[0] + self.env_boundaries.low[0]
            stacked_y_min = self.state[12*b + 1]*self.action_scale[1] + self.env_boundaries.low[1]
            stacked_y_max = self.state[12*b + 7]*self.action_scale[1] + self.env_boundaries.low[1]
            
            #print("update z coord stacked:",stacked_x_min, stacked_x_max, stacked_y_min, stacked_y_max)
            x2 = min(curr_x_max, stacked_x_max)
            x1 = max(curr_x_min, stacked_x_min)
            y2 = min(curr_y_max, stacked_y_max)
            y1 = max(curr_y_min, stacked_y_min)
            
            if(x2-x1>SPATIAL_DELTA and y2-y1>SPATIAL_DELTA):
                stacked_z_max = self.state[12*b + 8]*(self.env_boundaries.high[2] - self.env_boundaries.low[2]) + self.env_boundaries.low[2]
                if(stacked_z_max > z_max):
                    z_max = stacked_z_max
                    
        #final validity check
        z_action_scale = (self.env_boundaries.high[2] - self.env_boundaries.low[2])
        if(z_max > self.env_boundaries.high[2] - z_action_scale*self.state[-1]):
            #Place it at max valid z coord for current box.
            #This will create collision with the already stacked box(es), and isStationary will return False with motion penalty
            z_max = self.env_boundaries.high[2] - z_action_scale*self.state[-1]
            
        return z_max + z_action_scale*self.state[-1]/2
    
    def isStationary(self, prePosition):
        '''
        This determines if a physics simulation creates any motion for currently handling item and stacked items
        the prePosition is a 1D list of [(bIdx, box center x, box center y, box center z) for current box and all stacked boxes]

        Returns
        -------
        bool, float = if it is stationary, motion penalty score
        '''
        
        bStationary = True
        motion = 0.0
        
        for i in prePosition:
            new_x = self.data.qpos[7*i[0] + 0]
            new_y = self.data.qpos[7*i[0] + 1]
            new_z = self.data.qpos[7*i[0] + 2]

            old_x = i[1]
            old_y = i[2]
            old_z = i[3]
            
            #print("Motion test: ", new_x, new_y, old_x, old_y)
            
            deltaMotion_H = np.sqrt(np.power((new_x-old_x),2) + np.power((new_y-old_y),2)+np.power((new_z-old_z),2))
            if deltaMotion_H > MOTION_DELTA:
                bStationary = False
                motion += deltaMotion_H
                #if(deltaMotion_H > maxMotion):
                #    maxMotion = deltaMotion_H
                    
        #Generally, motion penalty will be [0.0, ~-1.0]
        #When there is a collision with the already stacked box(es), the motion penalty will be large (<-10.0)
        return bStationary, -motion*10
    
    def updateState(self):
        '''
        This updates the state when the action on the currently handling item passes isStationary
        '''
        
        x = self.data.qpos[7*self.currBoxIdx+0]
        y = self.data.qpos[7*self.currBoxIdx+1]
        z = self.data.qpos[7*self.currBoxIdx+2]
        
        x_min = x - self.action_scale[0]*self.state[-3]/2
        x_max = x + self.action_scale[0]*self.state[-3]/2
        y_min = y - self.action_scale[1]*self.state[-2]/2
        y_max = y + self.action_scale[1]*self.state[-2]/2
        z_min = z - (self.env_boundaries.high[2] - self.env_boundaries.low[2])*self.state[-1]/2
        z_max = z + (self.env_boundaries.high[2] - self.env_boundaries.low[2])*self.state[-1]/2

        
        self.state[12*self.currBoxIdx + 0] = (x_min - self.env_boundaries.low[0])/self.action_scale[0]
        self.state[12*self.currBoxIdx + 1] = (y_min - self.env_boundaries.low[1])/self.action_scale[1]
        self.state[12*self.currBoxIdx + 2] = (z_min - self.env_boundaries.low[2])/(self.env_boundaries.high[2] - self.env_boundaries.low[2])
        
        self.state[12*self.currBoxIdx + 6] = (x_max - self.env_boundaries.low[0])/self.action_scale[0]
        self.state[12*self.currBoxIdx + 7] = (y_max - self.env_boundaries.low[1])/self.action_scale[1]
        self.state[12*self.currBoxIdx + 8] = (z_max - self.env_boundaries.low[2])/(self.env_boundaries.high[2] - self.env_boundaries.low[2])
        
        self.state = np.clip(self.state, a_min=0.0, a_max=1.0)
        
        self.stacked_z_max = max(self.state[12*self.currBoxIdx + 8], self.stacked_z_max)
        
        return
    
    def calcStepReward(self):
        '''
        This calculates the step reward by assessing the current box's proximity to the already stacked boxes.
        The comparison happens only on the xy-plane - thus, only compares with the boxes on the same z-level.

        (e.g., if curr_box is on z=0.1, while box1 z = 0.1, box2 z=4.1, ..., curr_box is compared only with box1)

        The distance b/w curr_box and other boxes are compared, and the reward is assigned based on the closest distance value.
        The closest distance is the same as the dimension of the other box (e.g., x_max - x_min).
        In order to assign the same score for different box dimensions, the distance value is normalized between [0, +1]

        Returns
        -------
        float = step reward
        '''
        z_curr_min = self.state[12*self.currBoxIdx + 2]
        x_curr_min = self.state[12*self.currBoxIdx + 0]
        y_curr_min = self.state[12*self.currBoxIdx + 1]

        min_dist = 1.0
        for b in self.stacked:
            z_other_min = self.state[12*b + 2]
            x_other_min = self.state[12*b + 0]
            y_other_min = self.state[12*b + 1]

            if np.round(z_curr_min,3)!=np.round(z_other_min,3): #another sanity check
                continue
            
            x_other_max = self.state[12*b + 6]
            y_other_max = self.state[12*b + 7]

            x_dim = x_other_max - x_other_min
            y_dim = y_other_max - y_other_min
            min_dim = min(x_dim, y_dim)

            dist = np.sqrt(np.power(x_curr_min-x_other_min,2) + np.power(y_curr_min-y_other_min,2))
            dist_normalized = (dist - min_dim)/10
            if(dist_normalized<min_dist):
                min_dist = dist_normalized

        return 0.1 + (1-min_dist)

    
    def calcVolEfficiency(self):
        '''
        This calculates the final reward when every box is stacked
        The reward will be assigned based on the overall volume efficiency of the current stacking
        
        Returns
        -------
        float = final reward
        '''
        
        total_vol = 0.0
        min_x = 1.0
        max_x = 0.0
        min_y = 1.0
        max_y = 0.0
        min_z = 1.0
        max_z = 0.0
        
        for b in self.stacked:
            curr_min_x = self.state[12*b + 0] 
            curr_max_x = self.state[12*b + 6] 
            curr_min_y = self.state[12*b + 1] 
            curr_max_y = self.state[12*b + 7] 
            curr_min_z = self.state[12*b + 2] 
            curr_max_z = self.state[12*b + 8]

            min_x = min(min_x, curr_min_x)
            max_x = max(max_x, curr_max_x)
            min_y = min(min_y, curr_min_y)
            max_y = max(max_y, curr_max_y)
            min_z = min(min_z, curr_min_z)
            max_z = max(max_z, curr_max_z)
            
            total_vol += (curr_max_x - curr_min_x)*(curr_max_y - curr_min_y)*(curr_max_z - curr_min_z)
        
        occupied_vol = (max_x - min_x)*(max_y - min_y)*(max_z - min_z)
        #print("Volume efficiency: ", total_vol/occupied_vol)
        return total_vol/occupied_vol
        
    
    def step(self, action):
        '''
        With the given action, the step function implements the state transition
        
        Returns
        -------
        np.array(float), float, bool, dict = next state, reward, is_done, info
        '''
        
        for i in range(2):
            msg = "action["+str(i)+"] value is not within [-1, +1] range, curr value is "+str(action[i])
            assert action[i]<=1 and action[i]>=-1, msg
        #values are within [-1, +1]
        (x, y) = action
        
        #Convert back to the original scale
        x = self.action_bias[0] + x*(self.action_scale[0] - self.state[-3]*self.action_scale[0])/2
        y = self.action_bias[1] + y*(self.action_scale[1] - self.state[-2]*self.action_scale[1])/2
        z = self.updateZcoord(x, y)    
        
        #print("curr item loc:", x, y, z)
        #mujoco qpos update
        self.data.qpos[7*self.currBoxIdx + 0] = x
        self.data.qpos[7*self.currBoxIdx + 1] = y
        self.data.qpos[7*self.currBoxIdx + 2] = z
        
        #Motion validity check
        prePosition = [[self.currBoxIdx, x,y,z]]
        for b in self.stacked:
            prePosition.append([b, self.data.qpos[7*b+0], self.data.qpos[7*b+1], self.data.qpos[7*b+2]])
        
        #Physics simulation 
        simstart = self.data.time
        while (self.data.time - simstart < 5*STEP_SIM_TIME):
            # Step simulation environment
            mj.mj_step(self.model, self.data)

        bStationary, motionPenalty = self.isStationary(prePosition)
        if(not bStationary):
            #print("Current stacked count: ",len(self.stacked))
            return self.state, -1.0, True, {}
        
        #Valid placement
        self.updateState()
        #step_reward = self.calcStepReward()
        self.stacked.append(self.currBoxIdx)
        if(self.nQueued>0):
            #After the valid placement, pick a next item
            self.currBoxIdx = self.extractItem()
            #print("_____________________________")
            return self.state, self.calcVolEfficiency(), False, {}
        
        #All items are stacked
        #print("Current stacked count: ",len(self.stacked))
        return self.state, self.calcVolEfficiency(), True, {}
    
    
    def render(self, agent = None, render_length = 5.0):
        '''
        MuJoCo simulation logic for graphics rendering
        '''
        self.window_init()
        # Set camera configuration for MuJoCo graphics rendering
        self.cam.azimuth = 45.608063
        self.cam.elevation = -30.588379
        self.cam.distance = 50.0
        self.cam.lookat = np.array([10, 10, 1.5])
        
        state = self.reset()
        done = False
        cumulative_r = 0
        step_cnt = 0
        frame_cnt = 0
        frame_cut = int(render_length/STEP_SIM_TIME)

        while not glfw.window_should_close(self.window):
            if(frame_cnt%frame_cut==0 and done):
                state = self.reset()
                cumulative_r = 0
                step_cnt = 0
                frame_cnt = 0
                done = False
                time.sleep(2)
            
            if(frame_cnt%frame_cut==0 and not done):
                #Given current STEP_SIM_TIME = 0.1, this means that the rendering will last for 5 seconds each (if the default render_length is used)
                #And after each 5 second, env.step(action) is called
                if(agent==None):
                    action = self.action_space.sample()
                else:
                    action, _states = agent.predict(state, deterministic=True)
                state, reward, done, _ = self.step(action)
            
                cumulative_r += reward
                step_cnt += 1
            else:
                simstart = self.data.time
                while (self.data.time - simstart < STEP_SIM_TIME):
                    # Step simulation environment
                    mj.mj_step(self.model, self.data)
            frame_cnt += 1
            
            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Show joint frames (this shows the mujoco "joint" of the box geometry - nice to have for better visualization of a box location)
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1

            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            mj.mjr_text(200, 'Step: '+str(step_cnt)+' cumulative reward: '+str(cumulative_r), self.context, 0.1, 0.1, 0, 1, 0)
            
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
        
        glfw.terminate()

'''
boxes = {0:5, 1:4, 2:3, 3:2}
env = BinPacker("./xml/environment.xml",  boxes=boxes, pallet_x_min=5, pallet_x_max=15, pallet_y_min=5, pallet_y_max=15)

env.render()
'''
'''
obs = env.reset()
done = False
while(not done):
        
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)

'''

