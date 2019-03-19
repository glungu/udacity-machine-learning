import os
import sys
import csv
import numpy as np
import pandas as pd
from agent_ddpg.ddpg_agent import AgentDDPG
from agent_ddpg.task_takeoff import TaskTakeoff

num_episodes = 2000
init_pos = np.array([0., 0., 150., 0., 0., 0.])
init_v = np.array([0., 0., 1.])
init_angle_v = np.array([0., 0., 0.])
target_pos = np.array([0., 0., 200.])

task = TaskTakeoff(init_pose=init_pos, init_velocities=init_v, 
        init_angle_velocities=init_angle_v, target_pos=target_pos)
agent = AgentDDPG(task) 

# dir for writing episode data
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data') 
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode

    # write episode to file
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    csvfile = open(os.path.join(data_dir, 'episode_' + str(i_episode).zfill(4) + '.csv'), 'w')
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    
    successes = np.zeros(num_episodes)
    stop_training = False

    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state

        # write episode step
        rowdata = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)
        rowdata = [format(x,'.3f') if type(x) is np.float64 or type(x) is float else x for x in rowdata]
        writer.writerow(rowdata)

        if done:
            successes[i_episode-1] = 1 if task.success else 0
            success_rate = np.sum(successes[i_episode-10:i_episode]) / 10.0
            if success_rate > 0.7:
                stop_training = True                
            
            formatted = "\rEpisode = {:4d}, R_average = {:7.3f}, R_total = {:7.3f}, Position: [{:7.3f},{:7.3f},{:7.3f}]".format(
                i_episode, agent.average_reward, agent.total_reward, task.sim.pose[0], task.sim.pose[1], task.sim.pose[2])
            if task.success:
                print("Successes:", successes[i_episode-10:i_episode])
                formatted += ", Success! Rate: " + str(success_rate)

            print(formatted)
            break

    csvfile.close()
    sys.stdout.flush()
    if stop_training: 
        break