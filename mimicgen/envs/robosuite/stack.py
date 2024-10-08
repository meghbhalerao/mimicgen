# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from collections import OrderedDict
import numpy as np
import sys

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements, string_to_array

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, ObjectPositionSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.manipulation.stack import Stack

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class Stack_D0(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 2
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return { 
            k : dict(
                x=(-0.08, 0.08),
                y=(-0.08, 0.08),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB"]
        }


class Stack_D1(Stack_D0):
    """
    Much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(old_agentview_full_camera_pose[0]),
            quat=string_to_array(old_agentview_full_camera_pose[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(old_agentview_camera_pose[0]),
            quat=string_to_array(old_agentview_camera_pose[1]),
        )

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return { 
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB"]
        }


class StackThree(Stack_D0):
    """
    Stack three cubes instead of two.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 3
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))
            assert np.array_equal(np.array(bounds["cubeB"][k]), np.array(bounds["cubeC"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def reward(self, action=None):
        """
        We only return sparse rewards here.
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _check_cubeC_lifted(self):
        # cube C needs to be higher than A
        return self._check_lifted(self.cubeC_body_id, margin=0.08)

    def _check_cubeC_stacked(self):
        grasping_cubeC = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        cubeC_lifted = self._check_cubeC_lifted()
        cubeC_touching_cubeA = self.check_contact(self.cubeC, self.cubeA)
        return (not grasping_cubeC) and cubeC_lifted and cubeC_touching_cubeA

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.
        Returns:
            3-tuple:
                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # Stacking successful when A is on top of B and C is on top of A.
        # This means both A and C are lifted, not grasped by robot, and we have contact
        # between (A, B) and (A, C).

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_reach = 0.
        r_lift = 0.
        r_stack = 0.
        if self._check_cubeA_stacked() and self._check_cubeC_stacked():
            r_stack = 1.0

        return r_reach, r_lift, r_stack

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Add reference for cube C
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled
        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeC_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeC_body_id])

            @sensor(modality=modality)
            def cubeC_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeC_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache[f"{pf}eef_pos"] if \
                    "cubeC_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeA_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache["cubeA_pos"] if \
                    "cubeA_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeB_to_cubeC(obs_cache):
                return obs_cache["cubeB_pos"] - obs_cache["cubeC_pos"] if \
                    "cubeB_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            sensors = [cubeC_pos, cubeC_quat, gripper_to_cubeC, cubeA_to_cubeC, cubeB_to_cubeC]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return { 
            k : dict(
                x=(-0.10, 0.10),
                y=(-0.10, 0.10),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }

class StackThree_D0(StackThree):
    """Rename base class for convenience."""
    pass


class StackThree_D1(StackThree_D0):
    """
    Less z-rotation (for easier datagen) and much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(old_agentview_full_camera_pose[0]),
            quat=string_to_array(old_agentview_full_camera_pose[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(old_agentview_camera_pose[0]),
            quat=string_to_array(old_agentview_camera_pose[1]),
        )

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return { 
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }


class StackFour_D0(Stack, SingleArmEnv_MG):

    def __init__(self, **kwargs):
        

        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer!"


        bounds = self._get_initial_placement_bounds()


        # ensure cube symmetry
        assert len(bounds) == 4
        for k in ["x", "y", "z_rot", "reference"]:
            arrays = [np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]), np.array(bounds["cubeC"][k]), np.array(bounds["cubeD"][k])]

            assert all(np.array_equal(arrays[0], arr) for arr in arrays[1:]), "all arrays must be equal"

        placement_initializer = UniformRandomSampler(name = "ObjectSampler",
        x_range=bounds["cubeA"]["x"],
        y_range=bounds["cubeA"]["y"],
        rotation=bounds["cubeA"]["z_rot"],
        rotation_axis='z',
        ensure_object_boundary_in_range = False,
        ensure_valid_placement = True,
        reference_pos = bounds['cubeA']['reference'],
        z_offset = 0.01)

        Stack.__init__(self, placement_initializer = placement_initializer, **kwargs)

    def edit_model_xml(self, xml_str):
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)
    
    def staged_rewards(self):
        pass
    
    def _check_success(self):
        if self.reward() > 0:
            return True
        return False

    def reward(self, action = None):
        return int(self._check_cubes_stacked())
    
    def _check_valid_ordering(self, cube_ordering, cube_height, lift_margin = 0.01,  cube_dims = [0.02, 0.02, 0.02]):
        assert cube_dims[0] == cube_dims[1] == cube_dims[2]
        table_height = self.table_offset[2]

        for idx, _ in enumerate(cube_ordering):
            # check if the cubes except for the botton one is not simply lying on the table, that is they are lifted off the ground, since they could be in contact in the specified order, but they might just be lying on the table
            if idx > 0:
                body_height = self.sim.data.body_xpos[self.cube_str_to_cube_id["cube" + cube_ordering[idx]]][2]
                print(body_height)
                body_lifted = body_height > table_height  + lift_margin
                if not body_lifted:
                    return False

        # check if the gripper is grasping the cube - it should not be grasping any of the cubes in the final state - i.e the state in which reward will be = 1.      
        for idx, _ in enumerate(cube_ordering):
            if self._check_grasp(gripper = self.robots[0].gripper, object_geoms = self.cube_str_to_cube["cube" + cube_ordering[idx]]):
                return False

        for idx, _ in enumerate(cube_ordering):
            # check if consecutive cubes in the ordering are touching each other 
            if idx < len(cube_ordering) - 1:
                is_touching = self.check_contact(self.cube_str_to_cube["cube" + cube_ordering[idx]], self.cube_str_to_cube["cube" + cube_ordering[idx + 1]])
                if not is_touching:
                    return False

        return True
            
    def _check_cubes_stacked(self):
        # this is the final state that we want the cubes to be in
        
        valid_cube_orderings = [["A", "B", "C", "D"],["B", "A", "C", "D"],["B", "A", "D", "C"],["A", "B", "D", "C"]]

        for cube_ordering in valid_cube_orderings:
            is_valid = self._check_valid_ordering(cube_ordering, 0.02)

        return is_valid

    def _setup_references(self):

        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)
        self.cubeD_body_id = self.sim.model.body_name2id(self.cubeD.root_body)
        self.cube_str_to_cube_id = {"cubeA": self.cubeA_body_id, "cubeB": self.cubeB_body_id, "cubeC": self.cubeC_body_id, "cubeD": self.cubeD_body_id}
        self.cube_str_to_cube = {"cubeA": self.cubeA, "cubeB": self.cubeB, "cubeC": self.cubeC, "cubeD": self.cubeD}

    def _load_arena(self):
        # allow subclasses to easily overrride arena settings

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts in self.model
        """

        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly

        xpos = self.robots[0].robot_model.base_xpos_offset['table'](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {"type" : "cube"}

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",}
        
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,)
    
        
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0, 1, 0, 1],
            material=greenwood)
        

        self.cubeD = BoxObject(
            name="cubeD",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0, 1, 0, 1],
            material=greenwood)

        
        cubes = [self.cubeA, self.cubeB, self.cubeC, self.cubeD]


        # Create a placement initializer object 
        print(self.placement_initializer)
        sys.exit()
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(name = 'ObjectSampler', mujoco_objects=cubes,
            x_range  = [-0.08, 0.08],
            y_range = [-0.08, 0.08],
            rotation = None,
            ensure_object_boundary_in_range = False,
            ensure_valid_placement = True,
            reference_pos = self.table_offset,
            z_offset = 0.01,)
        
        # task/environment includes arena, robot, and objects of interest 

        self.placement_initializer = CustomStackPlacementInitializer(cubes=cubes, table_offset=self.table_offset)
        self.placement_initializer.reset()
        print(self.placement_initializer)
        self.model = ManipulationTask(mujoco_arena = mujoco_arena, mujoco_robots = [robot.robot_model for robot in self.robots], mujoco_objects = cubes)
    
    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placement of objects, e.g. what happens when env.reset is called, should return a dictionary with the following structure:
            object_name
                x: 2 tuple for low and high values for uniform sampling of x position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not changing
        """
        
        return { 
            k : dict(
                x=(-0.08, 0.08),
                y=(-0.08, 0.08),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC", "cubeD"]
        }

class CustomStackPlacementInitializer(ObjectPositionSampler):
    """
    Custom placement initializer that stacks red cubes on the bottom and green cubes on top.
    """
    def __init__(self, cubes, table_offset, z_offset=0.02):
        self.cubes = cubes
        self.table_offset = table_offset
        self.z_offset = z_offset

    def reset(self):
        # Stack the cubes explicitly by setting their positions
        # Assumes cubes are in order: [red_cubeA, red_cubeB, green_cubeC, green_cubeD]
        xpos = self.table_offset[0]
        ypos = self.table_offset[1]

        # Set object positions in the Mujoco XML by assigning initial pos for each cube
        self.cubes[0].body_offset = [xpos, ypos, self.z_offset]         # Red cube A
        self.cubes[1].body_offset = [xpos, ypos, self.z_offset + 0.02]   # Red cube B on top of cube A
        self.cubes[2].body_offset = [xpos, ypos, self.z_offset + 0.04]   # Green cube C on top of red cubes
        self.cubes[3].body_offset = [xpos, ypos, self.z_offset + 0.06]   # Green cube D on top of cube C

    def add_objects(self, objects):
        pass


# def main():
#     import robosuite as suite

#     env = suite.make(env_name = 'StackFour_D0', robots = 'Panda', controller_configs = load_controller_config(default_controller="OSC_POSE"), has_renderer=True, has_offscreen_renderer=False, ignore_done=True, use_camera_obs=False, control_freq=20)


#     env.reset()
#     env.viewer.set_camera(camera_id=0)
    
#     print(env._check_success)



# if __name__ == '__main__':
#     main()