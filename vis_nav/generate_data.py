import os
import numpy as np
import imageio
import re
from time import time
import random
import pathlib
import sys

# Location environment variables must be set
assert 'VLV_LOCATION' in os.environ, "VLV location must be set"
assert 'GIBSON_LOCATION' in os.environ, "Gibson location must be set"
sys.path.insert(0,os.environ['VLV_LOCATION'])
import habitat
from habitat_sim.utils import common as hutil
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from vis_nav.async_data_writer import AsyncLambdaRunner, ensure_folders, numpy_writer

# Note: These two files must be imported from the VLV habitat repo
# You may need to set your python path to include the install location of that repo
from habitat_test_env import HabitatTestEnv
from gibson_info import get_house

start_perterbation_radius = np.array([0.25,0,0.25])
start_pos_base = np.array([-1.8475336 ,  0.08005357,  2.4482806 ])
start_rot = hutil.quat_from_coeffs(np.array([ 0.        ,  0.01335649,  0.        , -0.99991083]))
goal_far = np.array([-1.4666749 ,  0.08005357, -7.259746  ])
no_goal_near = np.array([ 2.1523066 ,  0.08005357, -0.5214175 ])
goal_near = np.array([ 0.82861066,  0.08005357, -2.71881   ])
goals = [goal_far,no_goal_near,goal_near]

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('write_location')
args = parser.parse_args()
assert 'VLV_LOCATION' in os.environ
assert 'GIBSON_LOCATION' in os.environ
os.chdir(os.environ["VLV_LOCATION"])

epsilon = 0.2
file_root = args.write_location
pathlib.Path(file_root).mkdir(parents=True, exist_ok=True)
trajectories_per_env = 1000
step_limit = 700
stall_time = 200

houses = [get_house('Arkansaw')]

writer = AsyncLambdaRunner(4)
writer.start()

for house in houses:
    episode_nums = []
    episodes = os.popen(f'ls {file_root}/{house.name} | grep episode').read().split()
    episode_nums = [int(re.match('episode_(\d+)',e)[1]) for e in episodes]
    remaining_episodes = sorted(list(set(range(trajectories_per_env)) - set(episode_nums))) 
    print(remaining_episodes)
    env = HabitatTestEnv(os.path.join(os.environ['GIBSON_LOCATION'],f'{house.name}.glb'), False, False, panorama=True,gpu_device_id=0,random_goal=True,turn_angle=30,num_floors = house.num_floors)
    follower = ShortestPathFollower(env.env.sim, 1, False)
    for i in remaining_episodes:
        print(f'\n\n\nStarting {house.name} - {i}')
        env.reset()
        env.reset(1)
        start_pos = start_pos_base + start_perterbation_radius*random.uniform(-1,1)
        env.set_agent_state(start_pos,start_rot)
        first_frame = env.get_observation()['rgb']
        goal_ind = np.random.multinomial(1,[0.5,0.5*0.99,0.5*0.01])
        goal_ind = np.argmax(goal_ind)
        destination = goals[goal_ind]

        action = follower.get_next_action(destination)
        # loop to ensure we don't sample something already at the end
        while action == 0 or action is None:
            first_frame = env.reset(floor)
            action = follower.get_next_action(destination)

        # move agent to destination
        steps = 0
        episode_path = f'{file_root}/{house.name}/episode_{i}'
        # os.system(f'mkdir {episode_path}')

        print(f"Gen episode {i}")
        episode_writes = []
        def queue_save_obs(array, step):
            path_prefix = f'{episode_path}/{step}'
            def lam():
                for index, im in enumerate(array):
                    path = f'{path_prefix}/{index}.jpg'
                    ensure_folders(path)
                    imageio.imwrite(path,im)
            episode_writes.append(lam)
            return path_prefix


        pre_obs = queue_save_obs(first_frame, steps)
        pre_pos,pre_angle = env.agent_state()

        print(f'Start {house.name}: {env.agent_state()[0]}')
        print(f'Dest: {destination}')
        action = follower.get_next_action(destination)
        samples = []

        start = time()
        stall_steps = 0
        stall = False
        while steps < step_limit:
            steps += 1
            # take a random action with probability epsilon
            if random.uniform(0,1) < epsilon:
                action = random.randint(1,3)
            obs,reward,done,info = env.step(action - 1)
            obs = obs['rgb']
            post_obs = queue_save_obs(obs, steps)
            post_pos,post_angle = env.agent_state()
            sample = np.array([pre_obs,*pre_pos,*hutil.quat_to_coeffs(pre_angle),post_obs,*post_pos,*hutil.quat_to_coeffs(post_angle),action])
            samples.append(sample)

            pre_obs,pre_pos,pre_angle = post_obs,post_pos,post_angle

            action = follower.get_next_action(destination)
            if stall and stall_steps < stall_time:
                stall_steps += 1
                action = 2
            done = action == None or action == 0
            if done and (destination == no_goal_near).all():
                done = False
                destination = goal_near
                stall = True
                action = 2
            if done:
                print("terminating")
                for e in episode_writes:
                    writer.put(e)
                writer.put(numpy_writer(f'{episode_path}/data', np.array(samples)))
                break
        print(steps)
        print(f"FPS: {steps*4/(time()-start)}")
        if steps == step_limit:
            raise Exception(f'ERROR: Terminal not reached')
            break
    env.close()
print("Waiting for files to write")
writer.join()
