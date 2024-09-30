import os
from collections import OrderedDict
from copy import deepcopy 
import numpy as numpy

from robosuite.utils.mjcf_utils import CustomMaterial, add_material, find_elements, string_to_array

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

import mimicgen
from mimicgen.models.robosuite.objects import BlenderObject, CoffeeMachinePodObject, LongDrawerObject, CupObject
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class StackFour(SingleArmEnv_MG):
    """
    This class corresponds to the task that involves placing 4 blocks on top of each other - using robot only for now, but later TODO - this should be extended to including the human also.

    Arguments to instantiate the class:
        robots (str or list of str): Specification for the robot arm to be placed in the environment, if multiple arms need to be placed in the environment, you must provide a list of strings to specify what are the robot arms that you want to place in the environment 

        env_configuration (str): defines how to position the multiple robot arms within the environment. For a single arm, this variable has no impact on the robot setup

        controller_configs (str or list of dict): Using this argument we can set what the specific parameters for the controllers are for each of the robots - if nothing is specified then the default controllers for the robot arms are used - i.e. the default ones that are available in robosuite.

        gripper_types (str or list of str): type of gripper, used to instantiate gripper models from gripper factory. Default is "default", which is the default grippers(s) associated with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model overrides the default gripper. Should either be single str if same gripper type is to be used for all robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initializing noise parameters - the keys of these parameters are as follows - 

            'magnitude': the scale factor of the uni-variate random noise applied to each of the robots given initial positions. Setting this value to `None` or 0.0 results in no noise being applied.
            
            If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
            
            If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            :Note: Specifying "default" will automatically use the default noise settings.
            
            Specifying None will automatically create the required dict with "magnitude" set to 0.0.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for the table - static, dynamic and rolling friction.

            use_camera_obs (bool): if True, every observation includes rendered image(s)

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_scale (None or float): Scales the normalized reward function by the amount specified.
            
            If None, environment reward remains unnormalized

            reward_shaping (bool): if True, use dense rewards.

            has_offscreen_renderer (bool): True if using off-screen rendering

            render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
            
            render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

            control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

            camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
            convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each robot's camera list).

            camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

            camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

            camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

            camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

            `None`: no segmentation sensor used `'instance'`: segmentation at the class-instance level `'class'`: segmentation at the class level `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(self, 
    robots,
    env_configuration="default",
    initialization_noise="default",
    table_full_size=(0.8,0.8,0.05),
    table_friction = (1, 5e-3, 1e-4),
    use_camera_obs = True)
