"""
Script to collect human demos for the mimicgen environments - the human can contol the computer's IO devices.
Adapted from the robosuite repository, original script can be found here - https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py 
"""
import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from demo_random_action import choose_mimicgen_environment
from robosuite.utils.input_utils import *

def collect_human_trajcetory(env, device, arm, env_configuration):
    """
    Use device (keyboard or SpaceNav 3D Mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats
    """

    env.reset()

    env.render()

    is_first = True

    task_completion_hold_count = -1 # counter to collect 10 timesteps after reaching goal
    device.start_control()

    while True:
        # set active robot
        active_robot = env.robots[0] if env_configuration == 'bimanual' else env.robots[arm == 'left']

        # get the newest action
        action, grasped = input2action(device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration)
        if action is None:
            break
        
        env.step(action)
        env.render()

        # break if we complete the task
        if task_completion_hold_count == 0:
            break
        
        print(env.reward())

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -=1 # decrement until task_compltion_hold reaches 0.
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1
    
    # cleanup env
    env.close()

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows - heirarchical data format

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - env name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            state (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        
        demo2 (group)
    
    Args:
        directory (str): Path to the dir containing raw demonstrations
        out_dir (str): Path to store hdf5 file
        env_info (str): JSON encoded string containing environment information, including controller and robot info.
    
    """
    hdf5_path = os.path.join(out_dir, 'demo.hdf5')
    f = h5py.File(hdf5_path, 'w')

    grp = f.create_group('data')

    num_eps = 0
    env_name = None

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, 'state_*.npz')
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle = True)
            env_name = str(dic['env'])
            states.extend(dic['states'])

            for ai in dic['action_infos']:
                actions.append(ai['actions'])
            success = success or dic['successful']
            

        if len(states) == 0:
            continue
        
        if success:
            print("Demonstration is successful and has been saved")
            del states[-1]
            assert len(states) == len(actions)

            num_eps+=1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            xml_path = os.path.join(directory, ep_directory, "model.xml")

            with open(xml_path, "r") as f:
                xml_str = f.read()
            
            ep_data_grp.attrs['model_file'] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset('states', data = np.array(states))
            ep_data_grp.create_dataset('actions', data = np.array(actions))
        
        else:
            print("Demonstration is unsuccessful and has not been saved!")
        

        now = datetime.datetime.now()
        grp.attrs['date'] = '{}-{}-{}'.format(now.month, now.day, now.year)
        grp.attrs['time'] = '{}:{}:{}'.format(now.hour, now.minute, now.second)
        grp.attrs['repository_version'] = suite.__version__
        grp.attrs['env'] = env_name
        grp.attrs['env_info'] = env_info
        f.close()


if __name__ == '__main__':
    
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default=os.path.join(suite.models.assets_root, "demonstrations"))

    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview_full", help="Which camera to use for collecting demos")
    parser.add_argument("--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()
    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    
    config = {}
    config["env_name"] = choose_mimicgen_environment()

    # Choose robot
    config["robots"] = choose_robots(exclude_bimanual=True)
    # Create argument configuration
    config["controller_configs"] = controller_config

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in config['env_name']:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)

    env_info = json.dumps(config)

    tmp_directory = os.path.join("../logs/{}".format(str(time.time()).replace(".", "_")))


    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajcetory(env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
