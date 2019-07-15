import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5.,
                 target_pos=None, action_repeat = 3, task="reach"):
        """Initialize a Task object.
        Params
        ======
            action_repeat: number of steps the agent can repeat the action.
            init_angle_velocities: initial radians/second for each of the three Euler angles
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            task: selects a pre-defined task such as 'take_off', 'landing', 'hover' ou 'reach'. The last
                  is the only one that uses all parameters from input. The first three calculate target
                  from the initial_pose.
        """
        self.action_repeat = action_repeat
        self.state_size = self.action_repeat * 6
        self.action_low = 500
        self.action_high = 1000
        self.action_size = 4

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 

        if task == 'hover':
            target_pos = self.sim.pose[:3]
        elif task == 'landing':
            target_pos = self.sim.pose[:3] * np.array([1, 1, 0])
        elif task == 'take_off':
            self.sim.angular_v *= 0
            self.sim.init_angle_velocities = self.sim.angular_v

            self.sim.v *= 0
            self.sim.init_velocities = self.sim.v

            self.sim.reset()
            target_pos = self.sim.pose[:3] * np.array([1, 1, 0]) + np.array([0, 0, 150])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_minkowski_distance(self, x1, x2=None, p=3):
        if not isinstance(x2, np.ndarray):
            x2 = np.zeros_like(x1)
        
        distance = 0
        for dim_x1, dim_x2 in zip(x1, x2):
            distance += abs(dim_x1 - dim_x2) ** p
        distance = distance ** (1./p)

        return distance

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #defines the minkowski degree p=2 to get the euclidian distance
        p = 2

        #calculates the minkowski distance
        lin_distance = self.get_minkowski_distance(x1=self.sim.pose[:3], x2=self.target_pos, p=p)

        #calculates the angular velocity as scalar
        ang_veloc = self.get_minkowski_distance(x1=self.sim.angular_v, p=p)

        #calculates the angular acceleration as scalar
        angular_accels = self.get_minkowski_distance(x1=self.sim.angular_accels, p=p)

        #calculates the linear velocities as scalar
        lin_veloc = self.get_minkowski_distance(x1=self.sim.v[:2], p=p)

        #calculates the linear acceleration as scalar
        lin_accel = self.get_minkowski_distance(x1=self.sim.linear_accel, p=p)

        #calculates the terms for angular and linear velocities
        ang_v = abs(1/lin_distance) * ang_veloc
        lin_v = abs(1/lin_distance) * lin_veloc
        ang_a = angular_accels 
        lin_a = lin_accel

        reward = 1 - lin_distance + 0.5 * (ang_a + lin_a)# - 0.2 * (ang_v + lin_v)
        # reward = np.tanh(1 + np.tanh(self.sim.v[-1]) - np.tanh(lin_veloc))
        # print(f"lin_distance={lin_distance} self.sim.v={self.sim.v}")
        # reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        # if self.reached_target():
        #     reward += 10
        return reward

    def reached_target(self):
        #verifies if the quadcopter is at inside the range of it own dimensions from the target position
        for dim_pos, dim_tgt, dim_siz in zip(self.sim.pose[:3], self.target_pos, self.sim.dims):
            if abs(dim_pos - dim_tgt) <= dim_siz:
                return True
        return False

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            if done: reward += 10
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state