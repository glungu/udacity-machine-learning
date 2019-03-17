import sys
import numpy as np
import pandas as pd
from agent_ddpg.ddpg_agent import AgentDDPG
from agent_ddpg.task_takeoff import TaskTakeoff

num_episodes = 1000
init_pos = np.array([0., 0., 0., 0., 0., 0.])
init_v = np.array([0., 0., 0.])
init_angle_v = np.array([0., 0., 0.])
target_pos = np.array([0., 0., 10.])

task = TaskTakeoff(init_pose=init_pos, init_velocities=init_v, 
        init_angle_velocities=init_angle_v, target_pos=target_pos)
agent = AgentDDPG(task) 

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, R_average = {:7.3f}, Position: [{:7.3f},{:7.3f},{:7.3f}]".format(
                i_episode, agent.average_reward, task.sim.pose[0], task.sim.pose[1], task.sim.pose[2]))
            break

    sys.stdout.flush()