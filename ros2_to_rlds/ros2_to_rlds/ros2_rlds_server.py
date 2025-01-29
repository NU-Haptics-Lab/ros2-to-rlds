import os, sys, pdb, uuid, datetime, yaml, json

# ROS stuff
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rqt_py_common import message_helpers

# local import (same folder)
from .ros2_rlds import ROS2_RLDS

### ROS2 Messages
from std_msgs.msg import Int32, Bool, Float64

### ROS2 Srvs
from std_srvs.srv import Trigger
from dexterity_master.srv import StringSrv, FloatSrv, SetTopics, GetAllFrames
###



DEBUG = True

class TFLookup():
    def __init__(self, tf_buffer):
        self.tf_buffer = tf_buffer
        
    def safe_tf_lookup(self, target_frame, source_frame):
        # target_frame = parent frame
        # source_frame = child frame
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame, # from this frame (the parent)
                source_frame, # to this frame (the child)
                rclpy.time.Time()) # latest available
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {target_frame} to {source_frame}: {ex}')
            return
        return t
    
    def GetAllFrames(self):
        all_frames_str = self.tf_buffer.all_frames_as_yaml()
        # convert all_frames_str into a dict
        all_frames_dict = yaml.safe_load(all_frames_str)
        
        # check for empty
        if len(all_frames_dict) == 0:
            all_frames = []
        
        # extract all the keys
        else:
            all_frames = list(all_frames_dict.keys())
            
        # return the frames
        return all_frames

class ROS2Data():
    def __init__(self, node, descriptor, tf_wrapper, required=True):
        # inputs
        self.node = node
        self.descriptor = descriptor
        self.tf_wrapper = tf_wrapper
        
        # members
        self.topics = [] # list of [topic_name1, ...]
        self.data = {} # dict of topic name : msg
        self.tfs = {} # parent frame : child frame
        
        # if this data is not required, then set a default Float64 msg
        if not required:
            self.data["default_topic"] = Float64()
        
    def TopicCallback(self, msg, topic_name):
        self.data[topic_name] = msg
        
    def Get(self):
        # augment self.data with GetTFs dict
        tf_dict = self.GetTFs(self.tfs)
        self.data.update(tf_dict)
        if len(self.data) == 0:
            self.node.get_logger().warn("CAREFUL: Empty {} data.".format(self.descriptor))
        return self.data
    
    def GetTFs(self, spec_dict):
        out = {}
        for parent, child in spec_dict.items():
            out[parent+"__to__"+child] = self.tf_wrapper.safe_tf_lookup(parent, child)
        return out
    
    def Dump(self):
        dd = {}
        dd["topics"] = self.topics
        dd["tfs"] = self.tfs
        
        if (DEBUG):
            self.node.get_logger().info("{} dumped: ".format(self.descriptor))
            self.node.get_logger().info(json.dumps(dd))
        
        return dd
    
    def Load(self, dd1):
        if self.descriptor not in dd1:
            return
        
        dd2 = dd1[self.descriptor]
        self.topics = dd2["topics"]
        self.tfs = dd2["tfs"]
        
        if (DEBUG):
            self.node.get_logger().info("{} loaded: ".format(self.descriptor))
            self.node.get_logger().info(json.dumps(dd2))


class ROS2ToRLDSServer(Node):
    """
        Server for setting up data collection instances using the ROS2_RLDS class.
        Frontend should specify which topics are 'actions' and which are 'observations'
        Frontend should let user start and stop data collection
    """
    def __init__(self):
        super().__init__('ROS2ToRLDSServer')
        # parameters
        self.user_str = os.path.expanduser('~')
        self.data_dir = os.path.join(self.user_str, 'data')
        self.data_dir_local_name = uuid.uuid4().hex # randomly generated folder name
        self.data_dir_full_name = os.path.join(self.data_dir, self.data_dir_local_name)
        
        # ROS2 parameters
        self.declare_parameter('enable_topic', "~/in/enable")
        self.enable_topic = self.get_parameter('enable_topic').get_parameter_value().string_value

        # vars
        self.subscribers = {}
        self.alltopics = self.GetROSTopicsAndTypes() # topic name : topic type
        
        # Data Collector Logger Things
        self.rate = 10 # default hz
        
        # movestate subscriber
        self.sub1 = self.create_subscription(Bool, self.enable_topic, self.EnableCB, 10)
        self._enabled = False
        
        # pubs
        self.pub1 = self.create_publisher(Bool, '~/episode_active', 10)
        
        # ros2 srvs
        # self.srv1 = self.create_service(StringSrv, 'addtopic', self.AddTopic)
        # self.srv2 = self.create_service(StringSrv, 'removetopic', self.RemoveTopic)
        self.srv3 = self.create_service(Trigger, '~/start', self.StartWriter)
        self.srv4 = self.create_service(Trigger, '~/stop', self.StopWriter)
        self.srv5 = self.create_service(StringSrv, '~/change_data_dir', self.ChangeDataDir)
        self.srv6 = self.create_service(FloatSrv, '~/change_rate', self.ChangeRate)
        self.srv7 = self.create_service(SetTopics, '~/set_topics', self.SetTopics)
        self.srv8 = self.create_service(Trigger, '~/reset', self.ResetSrv)
        self.srv9 = self.create_service(Trigger, '~/refresh', self.Refresh)
        self.srv10 = self.create_service(GetAllFrames, '~/get_all_frames', self.GetAllFrames)
        self.srv11 = self.create_service(StringSrv, '~/load_profile', self.LoadProfile)
        self.srv12 = self.create_service(StringSrv, '~/save_profile', self.SaveProfile)
        
        # tf listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_wrapper = TFLookup(self.tf_buffer)
        
        # Debug info
        if (DEBUG):
            # print all parameters
            print("user_str: " + self.user_str)
            print("data_dir: " + self.data_dir)
            print("data_dir_local_name: " + self.data_dir_local_name)
            print("data_dir_full_name: " + self.data_dir_full_name)
            print("alltopics: " + str(self.alltopics))
        
        # qos
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # init
        self.Reset()
            
        self.get_logger().info("ROS2ToRLDSServer initialized.")

            
    def Reset(self):
        self.action_class = ROS2Data(self, "actions", self.tf_wrapper)
        self.observation_class = ROS2Data(self, "observations", self.tf_wrapper)
        self.rewards_class = ROS2Data(self, "rewards", self.tf_wrapper, required=False)
        self.collecting_data = False
        self.DestroyDataSubscribers()
        
    def EnableCB(self, msg):
        self._enabled = msg.data
        self.pub1.publish(Bool(data=self._enabled))
        
        """
        Get the actual python class of the msg from the topic name (str)
        """
    def GetMsgTypeClass(self, topic):
        if topic not in self.alltopics:
            print("Topic {} not found in alltopics".format(topic))
            return None
        msgtype_string = self.alltopics[topic] # this will be of format "std_msgs/msg/String", but we need it in "std_msgs/String"
        prefix = msgtype_string.split("/")[0] # get the first part of the string
        suffix = msgtype_string.split("/")[-1] # get the second part of the string
        final = prefix + "/" + suffix # combine them
        msgtypeclass = message_helpers.get_message_class(final) # get the class
        return msgtypeclass

    def CreateSubscription(self, name, topic_type):
        msgtypeclass = self.GetMsgTypeClass(name)
        
        if msgtypeclass == None:
            print("Topic {} not found in get_message_type".format(name))
            return
        else:
            self.subscribers[name] = self.create_subscription(
                msgtypeclass,
                name,
                lambda msg: self.TopicCallback(msg, name, topic_type),
                self.qos)
        
    """
    topic_type: one of "actions", "observations", "rewards"
    """
    def CreateDataSubscribers(self, topic_type):
        if topic_type == "actions":
            topics = self.action_class.topics
        elif topic_type == "observations":
            topics = self.observation_class.topics
        elif topic_type == "rewards":
            topics = self.rewards_class.topics

        # iterate through topics and create subscribers
        for topic in topics:
            self.CreateSubscription(topic, topic_type)

        if (DEBUG):
            if len(topics) > 0:
                print("CreateDataSubscribers {}".format(topic_type))
                # print self.addtopics
                for key in topics:
                    print("Topic: ", key)
            else:
                print("No topics to subscribe to for {}".format(topic_type))
                
    def DestroyDataSubscribers(self):
        print("Destroying Data Subscribers.")
        for subscription in self.subscribers:
            self.destroy_subscription(subscription)
        self.subscribers = {}

    def TopicCallback(self, msg, topic_name, topic_type):
        # save topic info to data dictionary
        if topic_type == "actions":
            self.action_class.TopicCallback(msg, topic_name)
        elif topic_type == "observations":
            self.observation_class.TopicCallback(msg, topic_name)
        elif topic_type == "rewards":
            self.rewards_class.TopicCallback(msg, topic_name)
    
    def GetROSTopicsAndTypes(self):
        # The first element of each tuple is the topic name and the second element is a list of topic types.
        topics_tup = self.get_topic_names_and_types() # list of [topic-name -> [list of topic types]]
        # put into dictionary
        dd = {}
        for topic in topics_tup:
            dd[topic[0]] = topic[1][0] # always take the first topic type
        # if (DEBUG):
        #     print("GetROSTopicsAndTypes: " + str(dd))
        return dd

    def MakeDataDir(self):
        if (DEBUG):
            print("MakeDataDir: " + self.data_dir_full_name)
        os.makedirs(self.data_dir_full_name)
        
    def MakeNewDataDir(self):
        self.data_dir_local_name = uuid.uuid4().hex # randomly generated folder name
        self.data_dir_full_name = os.path.join(self.data_dir, self.data_dir_local_name)
        self.MakeDataDir()
        
        
    ##### Services #####
    # def AddTopic(self, req, res):
    #     name = req.string
    #     if name not in self.addtopics:
    #         self.addtopics.append(name)
    #         if (DEBUG):
    #             print("Added topic: " + name)
    #         res.success = True
    #     else:
    #         res.success = False
    #     return res

    # def RemoveTopic(self, req, res):
    #     name = req.string
    #     if name in self.addtopics:
    #         del self.addtopics[name]
    #         if (DEBUG):
    #             print("Removed topic: " + name)
    #         res.success = True
    #     else:
    #         res.success = False
    #     return res

    def StartWriter(self, req, res):
        # only one collection instance at a time
        if (self.collecting_data):
            res.success = False
        else:
            self.MakeNewDataDir()
            
            # ensure action_cb and observation_cb are not empty
            if len(self.action_class.Get()) == 0 or len(self.observation_class.Get()) == 0:
                res.success = False
                self.get_logger().error("No Action Data or Observation Data. Not starting writer.")
                return res
            
            # make the data collector
            self.data_collector = ROS2_RLDS(
                self,
                self.action_class.Get,
                self.observation_class.Get,
                self.HasEpisodeBegun,
                self.HasEpisodeEnded,
                self.rate,
                TRAJECTORIES_DIR=self.data_dir_full_name,
                reward_cb=self.rewards_class.Get
            )
            
            # start the data collector
            self.data_collector.RunTimer()
            
            self.collecting_data = True
            res.success = True
        return res
        
    def StopWriter(self, req, res):
        if (self.collecting_data):
            # stop the data collector
            self.data_collector.EndTimer()
            # destroy subscribers
            self.DestroyDataSubscribers()
            res.success = True
            self.collecting_data = False
        else:
            res.success = False
        return res
    
    def ChangeDataDir(self, req, res):
        self.data_dir = req.data
        self.data_dir_full_name = os.path.join(self.data_dir, self.data_dir_local_name)
        res.success = True
        return res
    
    def ChangeRate(self, req, res):
        self.rate = req.data
        res.success = True
        return res
    
    def SetTopicsForClass(self, req, class_):
        class_.topics = req.topics
        self.CreateDataSubscribers(req.type)
        for tf_spec in req.tf_specs:
            class_.tfs[tf_spec.parent] = tf_spec.child
    
    def SetTopics(self, req, res):
        self.Refresh(req, res) # refresh topics
        # clear existing subscriptions
        self.DestroyDataSubscribers()
        # save selected topics
        topics = req.topics
        topic_type = req.type
        if topic_type == "actions":
            self.SetTopicsForClass(req, self.action_class)
        elif topic_type == "observations":
            self.SetTopicsForClass(req, self.observation_class)
        elif topic_type == "rewards":
            self.SetTopicsForClass(req, self.rewards_class)
        #
        res.success = True
        
        if (DEBUG):
            print("SetTopics for {}: ".format(req.type) + str(topics))
            print("SetTopics for {}, transforms specs: ".format(req.type) + str(req.tf_specs))
        return res
    
    def ResetSrv(self, req, res):
        self.StopWriter(req, res) # protection within
        self.Reset()
        res.success = True
        
        if DEBUG:
            print("ResetSrv.")
        return res
    
    def Refresh(self, req, res):
        self.alltopics = self.GetROSTopicsAndTypes()
        res.success = True
        
        if DEBUG:
            print("Refresh.")
        return res
    
    def GetAllFrames(self, req, res):
        all_frames = self.tf_wrapper.GetAllFrames()
            
        # return the frames
        res.frames = all_frames
        res.success = True
        
        # if (DEBUG):
        #     print("All TFs yaml: " + str(y))
        res.success = True
        return res
    
    def SaveProfile(self, req, res):
        filename = os.path.expanduser(req.data) 
        
        # TODO: check extension on filename for .json
        if len(filename) == 0:
            self.get_logger().error("Invalid profile path: {}".format(filename))
            res.success = False
            return res
        else:
            self.get_logger().info("Profile Saved at {}".format(filename))
        
        
        # construct the json dict
        json_dict = {}
        json_dict["actions"] = self.action_class.Dump()
        json_dict["observations"] = self.observation_class.Dump()
        json_dict["rewards"] = self.rewards_class.Dump()
        
        # open and save
        with open(filename, "w") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
            
        # done
        res.success = True
        return res
    
    def LoadProfile(self, req, res):
        # first, reset
        self.Reset()
        
        # now load in the profile
        filename = os.path.expanduser(req.data) 
        self.get_logger().info("Profile Loaded from {}".format(filename))
        
        # TODO: check extension on filename for .json
        
        # open and load
        with open(filename) as f:
            json_dict = json.load(f)
            
        # load into classes
        self.action_class.Load(json_dict)
        self.observation_class.Load(json_dict)
        self.rewards_class.Load(json_dict)
        
        # start subscribers
        self.CreateDataSubscribers("actions")
        self.CreateDataSubscribers("observations")
        self.CreateDataSubscribers("rewards")
            
        # done
        res.success = True
        return res
    
    ############## Data Collector Callbacks ##############
    def HasEpisodeBegun(self):
        return self._enabled
    
    def HasEpisodeEnded(self):
        return not self._enabled
    
    
        

def main(args=None):
    # make the ROS class
    rclpy.init(args=args)
    node = ROS2ToRLDSServer()

    # start the ros node
    print("Spin")
    rclpy.spin(node)
    
    print("Shutdown.")
    rclpy.shutdown()
    
    # clean up
    print("Clean Up.")
    pass

if __name__ == '__main__':
    main()