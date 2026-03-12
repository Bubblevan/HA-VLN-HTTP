from typing import Any, Dict, Optional, Tuple, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.discrete_planner import DiscretePathPlanner
from habitat_extensions.utils import generate_video, navigator_video_frame

import json
import os
import threading
import queue
import time
import magnum as mn
import random
from habitat.core.logging import logger
import habitat_sim


class HAVLNCE():
    def __init__(self, config: Config, sim):
        
        self._config = config
        self._sim = sim
        # Initialize simulator resources
        self._initialize_simulator_resources()
        self.__init_manager__()

        # Initialize total signals sent and previous human object IDs
        self.total_signals_sent = 0  # Total number of signals sent
        self.previous_human_object_ids = []  # Store object IDs to remove later

        # Initialize a lock for thread safety
        self.human_model_lock = threading.Lock()

        # Initialize the signal queue for thread communication, maxsize=120
        self.signal_queue = queue.Queue(maxsize=120)

        # Start the child thread to send signals
        self._signal_thread = threading.Thread(target=self._signal_sender)
        self._signal_thread.daemon = True
        self._signal_thread.start()

    def reset(self) -> None:

        # Clear signal queue and reset counters
        self._reset_signal_queue_and_counters()

        self.signal_queue.put('REFRESH_HUMAN_MODEL', timeout=1)
        self.total_signals_sent += 1  # Increment total signals sent

        # self._handle_signals()
        self.add_new_human_model(frame_id=0)

    
    
    def _reset_signal_queue_and_counters(self):
        """Reset the signal queue and related counters before starting a new episode."""
        # Clear the signal queue
        with self.signal_queue.mutex:
            self.signal_queue.queue.clear()
        # logger.info("Signal queue cleared.")

        # Reset total signals sent counter
        self.total_signals_sent = 0
        # logger.info("Total signals sent counter reset to 0.")

        # Reset frame counter if used
        self.frame_counter = 0
        # logger.info("Frame counter reset to 0.")

        # Remove any existing human models
        self.remove_previous_human_model()
        # logger.info("Previous human models removed.")
        self.frame_id = 0


    def _initialize_simulator_resources(self):
        """Initialize resources required by the simulator."""
        # Load human motion data and other resources
        self.human_motion_json_path = self._config.SIMULATOR.HUMAN_INFO_PATH
        self.data_path = self._config.SIMULATOR.HUMAN_GLB_PATH
        self.nav_mesh_path = self._config.SIMULATOR.RECOMPUTE_NAVMESH_PATH

        with open(self.human_motion_json_path, 'r') as f:
            self.human_motion_data = json.load(f)

        self.obj_templates_mgr = self._sim.get_object_template_manager()
        logger.info("Simulator resources initialized.")

    def _signal_sender(self):
        """Child thread function to send signals to the main thread every 0.5 seconds."""
        logger.info("Signal sender thread started.")
        while True:
            try:
                # Send a signal to refresh the human model if queue is not full
                self.signal_queue.put('REFRESH_HUMAN_MODEL', timeout=1)
                self.total_signals_sent += 1  # Increment total signals sent
                # logger.info(f"Signal sent to main thread to refresh human model. Total signals sent: {self.total_signals_sent}")
            except queue.Full:
                # logger.warning("Signal queue is full. Cannot send more signals.")
                pass
            # Wait for 0.5 seconds before sending the next signal
            time.sleep(0.1)

    def _handle_signals(self):
        """Main thread method to handle signals from the child thread."""
        try:
            signals_processed = 0
            while not self.signal_queue.empty():
                signal = self.signal_queue.get_nowait()
                signals_processed += 1
                # No need to process the signal content, we just count them

            if signals_processed > 0:
                # Calculate frame ID based on total signals sent
                frame_id = (self.total_signals_sent - 1) % 120
                
                if self.total_signals_sent <= 120:
                    # logger.info(f"Processing {signals_processed} signals. Using frame {frame_id}")

                    self.refresh_human_model(frame_id)
                    self.frame_id = frame_id
                    scan = self._sim._current_scene.split('/')[-2]
                    # self._sim.pathfinder.load_nav_mesh(os.path.join(self.nav_mesh_path, scan, scan + f'_{self.frame_id:03d}.navmesh'))
                    if not os.path.exists(os.path.join(self.nav_mesh_path, scan)):
                        os.makedirs(os.path.join(self.nav_mesh_path, scan))
                    navmesh_path = os.path.join(self.nav_mesh_path, scan, scan + f'_{self.frame_id:03d}.navmesh')
                    if not os.path.exists(navmesh_path):
                        navmesh_settings = habitat_sim.nav.NavMeshSettings()
                        navmesh_settings.set_defaults()
                        navmesh_settings.agent_radius = 0.1
                        navmesh_settings.agent_height = 1.5
                        state = self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings, include_static_objects=True)
                        assert state 
                        state = self._sim.pathfinder.save_nav_mesh(navmesh_path)
                        assert state 
                    else:
                        self._sim.pathfinder.load_nav_mesh(navmesh_path)
                    
        except Exception as e:
            logger.error(f"An error occurred while handling signals: {e}")

    def refresh_human_model(self, frame_id):
        """Method to refresh the human model (remove previous and add new one)."""
        # Remove previous human model
        self.remove_previous_human_model()
        # Add new human model with the specified frame_id
        self.add_new_human_model(frame_id)

    def remove_previous_human_model(self):
        """Remove the previous human model from the simulator."""
        # logger.info("Removing previous human model.")
        with self.human_model_lock:
            if self.previous_human_object_ids:
                existing_ids = set(self._sim.get_existing_object_ids())
                for obj_id in self.previous_human_object_ids:
                    if obj_id in existing_ids:
                        self._sim.remove_object(obj_id)
                        # logger.info(f"Removed object with ID {obj_id}")
                    else:
                        logger.warning(f"Object ID {obj_id} does not exist in simulator.")
                        # raise
                self.previous_human_object_ids = []

    def add_new_human_model(self, frame_id):
        """Add a new human model to the simulator."""
        # logger.info(f"Adding new human model at frame {frame_id}.")
        scan = self._sim._current_scene.split('/')[-2]
        human_motions = self.human_motion_data.get(scan, {})
        # frame_id is passed as an argument
        human_positions = {}

        with self.human_model_lock:
            self.previous_human_object_ids = []  # Reset the list for current objects

            for viewpoint, viewpoint_data in human_motions.items():
                category = viewpoint_data["category"]
                glb_index = viewpoint_data["index"]
                # Get translation and rotation for the specified frame
                translation = viewpoint_data["translation"][frame_id]
                rotation_euler = viewpoint_data["rotation"][frame_id]

                category = category.replace(" ", "_").replace("/", "_")

                template_id = self.obj_templates_mgr.get_template_handle_by_ID(self.or_num +
                                                (self.category2idx[f'{category}_{glb_index}'])*120+frame_id)

                if translation is not None:
                    translations = mn.Vector3(translation)
                    rotation_quat = (
                        mn.Quaternion.rotation(mn.Deg(rotation_euler[0]), mn.Vector3(1.0, 0.0, 0.0)) *
                        mn.Quaternion.rotation(mn.Deg(rotation_euler[1]), mn.Vector3(0.0, 1.0, 0.0)) *
                        mn.Quaternion.rotation(mn.Deg(rotation_euler[2]), mn.Vector3(0.0, 0.0, 1.0))
                    )
                    object_id = self._sim.add_object_by_handle(template_id)

                    self._sim.set_translation(translations, object_id)
                    self._sim.set_rotation(rotation_quat, object_id)
                    rotation_quat = self._sim.get_rotation(object_id)
                    human_positions[viewpoint] = (np.array(translation), rotation_euler)

                    self._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, object_id)

                    self._sim.step_physics(1.0 / 5.0)
                    self.previous_human_object_ids.append(object_id)
            
            self._sim._human_positions = human_positions
    
    def __init_manager__(self):

        self.obj_templates_mgr = self._sim.get_object_template_manager()
        
        idx2category = {}
        self.category2idx = {}
        pre_num = self.obj_templates_mgr.get_num_templates()
        self.or_num = pre_num

        glb_lists = sorted(os.listdir(self.data_path))
        glb_lists = [f for f in glb_lists if os.path.isdir(os.path.join(self.data_path, f))]

        for i, category in enumerate(glb_lists):
            idx2category[i] = category
            self.category2idx[category] = i
            self.obj_templates_mgr.load_configs(os.path.join(self.data_path, category))
            add_num = self.obj_templates_mgr.get_num_templates() - pre_num
            assert add_num == 120
            pre_num = self.obj_templates_mgr.get_num_templates()