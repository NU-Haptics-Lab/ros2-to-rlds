from absl import flags
from absl import logging
import envlogger
from envlogger.backends import tfds_backend_writer
from envlogger.step_data import StepData

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import dm_env

import rclpy
from .utils import ConvertROS2ToNumpy, ConvertNumpyToTFDSFeatures
from enum import Enum

from absl import logging
logging.set_verbosity(logging.INFO)

# following https://github.com/google-deepmind/envlogger
# and the example https://github.com/google-deepmind/envlogger/blob/main/envlogger/examples/tfds_random_agent_catch.py

class State(Enum):
    WAITING = 1
    BEGIN   = 2
    RUNNING = 3
    END     = 4

# 
class ROS2_RLDS():
    """ ROS2-specific class. Assumes that the provided action_cb and observation_cb return dictionaries of topic_names -> ROS2 messages.
    
    all data streams must be continuous, or else the logger won't be able to get an initial message, and won't be able to compute the TFDS features
    """
    
    def __init__(self, nh, 
                 action_cb, 
                 observation_cb, 
                 has_epsiode_begun_cb, 
                 has_episode_ended_cb, 
                 rate, 
                 TRAJECTORIES_DIR,
                 reward_cb = None
                 ):
        # nh: rclpy node handle
        # action_cb: callback function that returns a dictionary of topic_names -> ROS2 messages
        # observation_cb: callback function that returns a dictionary of topic_names -> ROS2 messages
        # has_episode_ended: callback function that returns a boolean indicating whether the episode should be restarted
        # period: The time (in seconds) after which the environment updates.
        self.DEBUG = True
        self.nh = nh
        self._action_cb = action_cb
        self._observation_cb = observation_cb
        self._rate = rate
        self.HasEpisodeBegun = has_epsiode_begun_cb
        self.HasEpisodeEnded = has_episode_ended_cb
        self.TRAJECTORIES_DIR = TRAJECTORIES_DIR
        
        ## Params
        self.eps_per_file = 50
        
        ## Vars
        self._stop = False
        self.nb_steps_in_this_ep = 0
        self.nb_total_steps = 0
        self.nb_eps = 0
        
        
        ## members
        if reward_cb is None:
            self._ros2_reward_cb = None
            self._reward_example = 0.0
        else:
            self._ros2_reward_cb = reward_cb
            self._reward_example = self.GetNextReward()
        
        self._action_example = self.GetNextAction()
        self._observation_example = self.GetNextObservation()
        
        # checks
        if len(self._action_example) == 0 or len(self._observation_example) == 0:
            raise ValueError('The action and observation examples must not be empty.')
        
        action_info = ConvertNumpyToTFDSFeatures(self._action_example)
        observation_info = ConvertNumpyToTFDSFeatures(self._observation_example)
        reward_info = ConvertNumpyToTFDSFeatures(self._reward_example)
        
        if self.DEBUG:
            self.nh.get_logger().info('action_info: {}'.format(action_info))
            self.nh.get_logger().info('observation_info: {}'.format(observation_info))
            self.nh.get_logger().info('reward_info: {}'.format(reward_info))
            
        # config
        # https://github.com/tensorflow/datasets/blob/fdad1d9e8f1cb34389a336132b2f842cbc7aca57/tensorflow_datasets/rlds/rlds_base.py#L29 
        self.dataset_config = tfds.rlds.rlds_base.DatasetConfig(
                name='ros2',
                observation_info=observation_info,
                action_info=action_info,
                reward_info=reward_info,
                discount_info=np.float64, # default python type for 0.
        )
        #
    def GetNextObservation(self):
        raw_observation = self._observation_cb()
        return ConvertROS2ToNumpy(raw_observation)
        
    def GetNextAction(self):
        # call the action cb to get a dictionary of topic_names -> ROS2 messages
        raw_action = self._action_cb()
        return ConvertROS2ToNumpy(raw_action)
    
    def GetNextReward(self):
        # call the reward cb to get a dictionary of topic_names -> ROS2 messages
        if self._ros2_reward_cb is None:
            return 0.0
        else:
            ros2_reward = self._ros2_reward_cb()
            return ConvertROS2ToNumpy(ros2_reward)
        
    def RunTimer(self):
        # make the env
            # https://github.com/google-deepmind/envlogger/blob/e59ff30e78bc05e8621a2932351d27908b29294e/envlogger/backends/tfds_backend_writer.py#L54 
        self.backend = tfds_backend_writer.TFDSBackendWriter(
                data_directory=self.TRAJECTORIES_DIR,
                max_episodes_per_file=self.eps_per_file,
                ds_config=self.dataset_config)
            
        # make a ros2 timer
        self.state = State.WAITING
        self.nh.get_logger().info('Waiting for new episode to begin...')
        self.timer = self.nh.create_timer(1.0 / self._rate, self.TimerCore)
        
    def ResetEp(self):
        self.nb_eps += 1
        self.nb_total_steps += self.nb_steps_in_this_ep
        self.nb_steps_in_this_ep = 1 # starts with the first step
        
    def Restart(self):
        self.timestep = dm_env.restart(observation=self.GetNextObservation())
        data = StepData(timestep=self.timestep, action=self.GetNextAction())
        self.backend.record_step(data, True)
        
    def Transition(self):
        self.timestep = dm_env.transition(reward=self.GetNextReward(), observation=self.GetNextObservation())
        data = StepData(self.timestep, self.GetNextAction())
        self.backend.record_step(data=data, is_new_episode=False)
        
    def Termination(self):
        self.timestep = dm_env.termination(reward=self.GetNextReward(), observation=self.GetNextObservation())
        data = StepData(self.timestep, self.GetNextAction())
        self.backend.record_step(data, False)
        
    """ Core interaction with the TFDS backend writer. """
    def TimerCore(self):
        # simple finite state machine
        if self.state == State.WAITING:
            if self.HasEpisodeBegun():
                self.state = State.BEGIN
                
        if self.state == State.BEGIN:
            self.nh.get_logger().info('Begin New Episode.')
            self.ResetEp()
            self.Restart()
            self.state = State.RUNNING
        
        if self.state == State.RUNNING:
            if not self.timestep.last():
                if self.HasEpisodeEnded():
                    self.Termination()
                else:
                    self.Transition()
                self.nb_steps_in_this_ep += 1
            else:
                self.nh.get_logger().info('Episode Ended.')
                self.nh.get_logger().info('# Steps for this Ep: {}.'.format(self.nb_steps_in_this_ep))
                self.state = State.END
                
        if self.state == State.END:
            self.nh.get_logger().info('Waiting for new episode to begin...')
            self.state = State.WAITING
            
    def EndTimer(self):
        self.nh.get_logger().info('Saving final dataset. DO NOT KILL THE PROCESS UNTIL COMPLETE...')
        self.nb_total_steps += self.nb_steps_in_this_ep
        self.timer.destroy()
        self.backend.close()
        self.nh.get_logger().info('Data Collection Ended.')
        self.nh.get_logger().info('Dataset written to {}'.format(self.TRAJECTORIES_DIR))
        self.nh.get_logger().info('Total # Steps: {}.'.format(self.nb_total_steps))
        self.nh.get_logger().info('Total # Eps: {}.'.format(self.nb_eps))
        self.nh.get_logger().info('Process now safe to kill.')