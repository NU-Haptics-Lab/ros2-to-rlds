import pdb
import ros2_numpy as rnp
import numpy as np
import tensorflow_datasets as tfds
import json
import copy

from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from rosidl_runtime_py import message_to_ordereddict


ALLOW_STRING_DATA = True
ALLOW_SEQUENCE_FEATURES = False # https://github.com/google-deepmind/envlogger/issues/14

def ros2_to_dict(msg):
    ordered = message_to_ordereddict(msg, no_str=not ALLOW_STRING_DATA)
    # out = dict(ordered)
    return ordered

def ros2_to_json(msg, pretty=False):
    dd = ros2_to_dict(msg)
    if pretty:
        out = json.dumps(dd, indent=4)
    else:
        out = json.dumps(dd)
    return out

def PrettyPrintMsg(msg):
    p = ros2_to_json(msg, pretty=True)
    print(p)

def PrettyPrintMsgNp(msg):
    p = ros2_to_json(msg, pretty=True)
    print(p)

def IsBuiltinNotString(var):
    return type(var) in (int, float, bool, bytes)

"""
dd: the output of ros2_to_dict
recall: in ROS2 a message value will either be a builtin, or another ros2 msg, or a list of built-ins, or a list of the SAME ros2 msg
"""
def ConvertDictToTFDS(dd):
    assert( isinstance(dd, dict) )
    
    out_dd = {}
    for k, v in dd.items():
        """
        A value can either be a builtin non-string, string, or a dict, or a list
        """
                
        # if the value is a builtin NOT STRING(scalar)
        if IsBuiltinNotString(v):
            out_dd[k] = tfds.features.Scalar(dtype=np.array(v).dtype) # scalar
                
        # is a string
        elif isinstance(v, str):
            out_dd[k] = tfds.features.Text()
            # out_dd[k] = tfds.features.Scalar(dtype=np.dtype("<U"))
        
        # if the value is a dictionary, recurse
        elif isinstance(v, dict):
            out_dd[k] = ConvertDictToTFDS(v)
            
        # if the value is a list
        elif isinstance(v, list):
            """
            List elements can either be all builtins non-string, or all strings, or dicts. We assume it CAN NOT have a combination of builtins and dicts
            """
            # if list is empty, put in zero length feature
            if len(v) == 0:
                out_dd[k] = tfds.features.Tensor(shape=(0,), dtype=np.float64)
                continue
            
            element0 = v[0]
            dtype = np.array(element0).dtype
            shape = (len(v), ) # 1-d shape of length len(v)
            
            # if the first element is builtin, we assume all are builtin. So we simply use a tensor/array
            if IsBuiltinNotString(element0):
                out_dd[k] = tfds.features.Tensor(shape=shape, dtype=dtype, encoding=tfds.features.Encoding.ZLIB)
                    
            elif isinstance(element0, str):
                if ALLOW_STRING_DATA:
                    out_dd[k] = tfds.features.Tensor(shape=(len(v), ), dtype=np.dtype("<U"))
                else:
                    # raise Exception("Sequences of text incompatible with TFDS right now. Use a different ROS2 message")
                    print("WARNING: Sequences of text incompatible with TFDS right now. Skipping this feature.")
                    
                
            # list contains dictionaries, so we recurse on that    
            elif isinstance(element0, dict):
                recursed = ConvertDictToTFDS(element0)
                                
                if ALLOW_SEQUENCE_FEATURES:
                    out_dd[k] = tfds.features.Sequence(recursed, length=len(v)) 
                else:
                    raise Exception("Sequences of dict's incompatible with TFDS right now.")
                    # print("WARNING: Sequences of dict's incompatible with TFDS right now. Skipping this feature.")
                    
            else:
                print("ConvertDictToTFDS: unknown element0 value:")
                print(element0)
        
        else:
            print("ConvertDictToTFDS: unknown v value:")
            print(v)
            
    # finally, return the correct type
    return tfds.features.FeaturesDict(out_dd)

def ConvertRos2DictToDict(dd):
    """
    dd: dictionary of topic_names -> ros2 messages
    """
    out_dd = {}
    for k, v in dd.items():
        out_dd[k] = ros2_to_dict(v)
        
    return out_dd