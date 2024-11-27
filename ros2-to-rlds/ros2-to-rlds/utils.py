import pdb
import ros2_numpy as rnp
import numpy as np
import tensorflow_datasets as tfds
import json
import copy

from dm_env import specs
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from ros2_numpy.registry import converts_to_numpy
from rosidl_runtime_py import message_to_ordereddict


ALLOW_STRING_DATA = True

@converts_to_numpy(Float32)
def convert(my_msg):
    return np.array(my_msg.data, dtype=np.float32)
@converts_to_numpy(Bool)
def convert(my_msg):
    return np.array(my_msg.data, dtype=np.bool_)
@converts_to_numpy(JointState)
def convert(my_msg):
    return np.array(my_msg.position, dtype=np.float64)
@converts_to_numpy(TransformStamped)
def convert(my_msg):
    return np.array(my_msg.position, dtype=np.float64)

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
    


"""
msgs confirmed to work with:
Float64
sensor_msgs/Image
std_msgs/TransformStamped
"""
def is_builtin_fcn(var):
    if ALLOW_STRING_DATA:
        return type(var) in (int, float, bool, str, bytes)
    else:
        return type(var) in (int, float, bool, bytes)
    # strings don't play nicely
    # return type(var) in (int, float, bool, bytes)

def IsBuiltinNotString(var):
    return type(var) in (int, float, bool, bytes)

"""
    Converts actual ROS2 message to numpy (NOT THE SPEC)
"""
def convert(input):
    # recursive convert for any arbitrary ROS2 message
    
    if type(input) == str:
        if ALLOW_STRING_DATA:
            # tfds.features.Text requires type to be "object"
            return np.array(input, dtype=object)
        else:
            return None

    # check if input is a builtin type
    if is_builtin_fcn(input):
        return np.array(input)
    
    # else, is a ROS2 msg
    else:
        msg = input
        out = {}
        for field, value in msg.get_fields_and_field_types().items():
            nested = getattr(msg, field)
            
            is_list = "sequence" in value
            is_builtin = "/" not in value
            
            # strings are special?
            if type(nested) == str:
                if ALLOW_STRING_DATA:
                    out[field] = convert(nested)
                else:
                    continue
            
            # this is a single entry, ROS2 message or builtin
            elif not is_list:
                out[field] = convert(nested)
                
            # this is a list or ROS2 messages or builtins
            elif is_list:
                out[field] = np.array([convert(nest) for nest in nested])
                
            # default
            else:
                print("dexterity_utils.convert: UNKNOWN INPUT: {}".format(input))
        return out

def ConvertROS2ToNumpy(dd):
    # dd: dictionary of topic_names -> ROS2 messages using the ros2_numpy lib
    out = {}
    for k, v in dd.items():
        if (type(v) == dict):
            out[k] = ConvertROS2ToNumpy(v)
        else:
            out[k] = convert(v)
    return out


def convert2(input):
    # recursive convert2 for any arbitrary ROS2 message or builtin type
    
    if type(input) == str:
        if ALLOW_STRING_DATA:
            # keep strings as raw strings, not wrapped in np array
            return input
        else:
            return None

    # check if input is a builtin type
    if is_builtin_fcn(input):
        return input
    
    # else, is a ROS2 msg
    else:
        out = {}
        
        # iterate through keys
        for key in input:
            val = input[key]
            
            # if it's a nested dict
            if isinstance(val, dict):
                # recurse and replace value with converted var
                converted = convert2(val)
                if len(converted) != 0:
                    out[key] = converted
            
            # this is a list of anything
            elif isinstance(val, list):
                new_list = []
                for va in val:
                    # convert
                    to_add = convert2(va)
                    
                    # confirm that to_add is non-zero
                    if isinstance(to_add, list) or isinstance(to_add, dict):
                        non_zero = len(to_add) != 0
                    else:
                        non_zero = True
                        
                    # confirm that to_add is not None and non_zero
                    if to_add is not None and non_zero:
                        new_list.append(to_add)
                if len(new_list) == 0:
                    continue
                else:
                    out[key] = np.array(new_list, dtype=object)
                
            # this is a scalar built-in
            elif is_builtin_fcn:
                out[key] = val
                
            # default
            else:
                print("dexterity_utils.convert2: UNKNOWN INPUT: {}".format(input))
        return out

def ConvertROS2ToNumpy2(dd):
    # dd: dictionary of topic_names -> ROS2 messages
    out = {}
    for k, v in dd.items():
        # ensure v is a ros2 message (aka has field and types)
        assert(hasattr(v, "get_fields_and_field_types"))
        
        # convert to dict
        v = ros2_to_dict(v)
        
        # convert to numpy
        converted = convert2(v)
        if len(converted) != 0:
            out[k] = converted
    return out

def IsNumpyObjectAString(np_obj):
    if type(np_obj) == np.ndarray:
        is_scalar = np.ndim(np_obj) == 0
        is_string = is_scalar and np_obj.dtype == object
    else:
        is_string = type(np_obj) == str
    return is_string

def ConvertNumpyToTFDSOrSpec(dd, use_tfds_features=True):
    out_dd = {}
    for k, v in dd.items():
        
        # if the value is a dictionary, recurse
        if (type(v) == dict):
            out_dd[k] = ConvertNumpyToTFDSOrSpec(v, use_tfds_features)
            
        # if the value is a numpy array
        elif type(v) == np.ndarray:
            v_np = v
            
            # includes object, bool, float, etc. Doesn't include dtype('<U2') aka string
            is_np_builtin = v_np.dtype.isbuiltin == 1
            
            # recommended by https://numpy.org/devdocs/reference/generated/numpy.isscalar.html
            is_scalar = np.ndim(v_np) == 0
            
            # https://stackoverflow.com/questions/10790312/numpy-check-array-for-string-data-type
            # is_string = v_np.dtype.type == np.str_
            is_string = IsNumpyObjectAString(v_np)
            
            # for now we assume that all elements of a non-scalar array are the same type, or the same dictionary layout, etc.
            is_objects = not is_scalar and v_np.dtype == object
            
            is_strings = is_objects and IsNumpyObjectAString(v_np[0]) 
            
            # check for a single string
            if is_string:
                if ALLOW_STRING_DATA:
                    # tfds.features.Tensor() does not support `encoding=` when dtype is string.
                    # out_dd[k] = tfds.features.Tensor(shape=v_np.shape, dtype=v_np.dtype)
                    if use_tfds_features:
                        out_dd[k] = tfds.features.Text()
                    else:
                        out_dd[k] = specs.StringArray(shape=())
                else:
                    continue
                    
            elif is_strings:
                if ALLOW_STRING_DATA:
                    if use_tfds_features:
                        # docs: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Sequence#at_construction_time
                        out_dd[k] = tfds.features.Sequence(tfds.features.Text(), length=len(v_np))
                    else:
                        out_dd[k] = specs.StringArray(shape=(len(v_np),))
                else:
                    continue
            
            # dtype object means it's an array (aka sequence) of different messages / types
            elif is_objects:
                # in order to recurse, the type MUST be a dict
                assert(type(v_np[0]) == dict)
                
                # only need to put in the first element since each element should have the same feature spec, RECURSE!
                if use_tfds_features:
                    # docs: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Sequence#at_construction_time
                    out_dd[k] = tfds.features.Sequence(ConvertNumpyToTFDSOrSpec(v_np[0], use_tfds_features), length=len(v_np)) 
                else:
                    out_dd[k] = [ConvertNumpyToTFDSOrSpec(val, use_tfds_features) for val in v_np]
                
            # this SHOULD be non-str built-in types
            elif is_np_builtin:
                if use_tfds_features:
                    out_dd[k] = tfds.features.Tensor(shape=v_np.shape, dtype=v_np.dtype, encoding=tfds.features.Encoding.ZLIB) # native numpy->tf conversion
                else:
                    out_dd[k] = specs.Array(shape=v_np.shape, dtype=v_np.dtype, name=k)
                
        # if the value is a builtin (scalar)
        elif v is not None:
            if use_tfds_features:
                # little trick to get the numpy dtype
                t = specs.Array(shape=(), dtype=type(v)) # scalar
                out_dd[k] = tfds.features.Scalar(dtype=t.dtype) # scalar
            else:
                out_dd[k] = specs.Array(shape=(), dtype=type(v), name=k) # scalar
            
    if use_tfds_features:
        return tfds.features.FeaturesDict(out_dd)
    else:
        return out_dd

def ConvertNumpyToSpec(dd):
    # dd: nested dictionary of topic_names -> np.array
    # out: dictionary of 
    return ConvertNumpyToTFDSOrSpec(dd, use_tfds_features=False)

def ConvertNumpyToTFDSFeatures(dd):
    """ Convert a nested dictionary of topic_names -> np.arrays to a nested dictionary of topic_names -> TFDS features. """
    return ConvertNumpyToTFDSOrSpec(dd, use_tfds_features=True)

"""
dd: the output of ros2_to_dict
recall: in ROS2 a message value will either be a builtin, or another ros2 msg, or a list of built-ins, or a list of the SAME ros2 msg
"""
def ConvertDictToTFDSOrSpec(dd, use_tfds_features=True):
    assert( isinstance(dd, dict) )
    
    out_dd = {}
    for k, v in dd.items():
        """
        A value can either be a builtin non-string, string, or a dict, or a list
        """
                
        # if the value is a builtin NOT STRING(scalar)
        if IsBuiltinNotString(v):
            if use_tfds_features:
                # little trick to get the numpy dtype
                t = specs.Array(shape=(), dtype=type(v)) # scalar
                out_dd[k] = tfds.features.Scalar(dtype=t.dtype) # scalar
            else:
                out_dd[k] = specs.Array(shape=(), dtype=type(v), name=k) # scalar
                
        # is a string
        elif isinstance(v, str):
            if use_tfds_features:
                out_dd[k] = tfds.features.Text()
            else:
                out_dd[k] = specs.StringArray(shape=(), name=k) # scalar
        
        # if the value is a dictionary, recurse
        elif isinstance(v, dict):
            out_dd[k] = ConvertDictToTFDSOrSpec(v, use_tfds_features)
            
        # if the value is a list
        elif isinstance(v, list):
            """
            List elements can either be all builtins non-string, or all strings, or dicts. We assume it CAN NOT have a combination of builtins and dicts
            """
            # if list is empty, put in zero length feature
            if len(v) == 0:
                if use_tfds_features:
                    out_dd[k] = tfds.features.Tensor(shape=(0,), dtype=np.float64)
                else:
                    out_dd[k] = specs.Array(shape=(0,), dtype=np.float64, name=k)
                continue
            
            element0 = v[0]
            dtype = np.array(element0).dtype
            shape = (len(v), ) # 1-d shape of length len(v)
            
            # if the first element is builtin, we assume all are builtin. So we simply use a tensor/array
            if IsBuiltinNotString(element0):
                if use_tfds_features:
                    out_dd[k] = tfds.features.Tensor(shape=shape, dtype=dtype, encoding=tfds.features.Encoding.ZLIB)
                else:
                    out_dd[k] = specs.Array(shape=shape, dtype=dtype, name=k)
                    
            elif isinstance(element0, str):
                if use_tfds_features:
                    # raise Exception("Sequences of text incompatible with TFDS right now. Use a different ROS2 message")
                    # out_dd[k] = tfds.features.Sequence(tfds.features.Text(), length=len(v))
                    
                    # test by wrapping in a dict
                    # out_dd[k] = tfds.features.Sequence({k: tfds.features.Text()}, length=len(v))
                    
                    # test by using a simple list
                    out_dd[k] = [tfds.features.Text()] * len(v)
                    
                else:
                    out_dd[k] = specs.StringArray(shape=shape, name=k)
                
            # list contains dictionaries, so we recurse on that    
            elif isinstance(element0, dict):
                recursed = ConvertDictToTFDSOrSpec(element0, use_tfds_features)
                                
                if use_tfds_features:
                    # raise Exception("Sequences of text incompatible with TFDS right now.")
                    out_dd[k] = tfds.features.Sequence(recursed, length=len(v)) 
                else:
                    out_dd[k] = [recursed] * len(v) # len(v) long
                    
            else:
                print("ConvertDictToTFDSOrSpec: unknown element0 value:")
                print(element0)
        
        else:
            print("ConvertDictToTFDSOrSpec: unknown v value:")
            print(v)
            
    # finally, return the correct type
    if use_tfds_features:
        return tfds.features.FeaturesDict(out_dd)
    else:
        return out_dd

def ConvertDictToSpec(dd):
    # dd: nested dictionary of topic_names -> dicts
    # out: dictionary of 
    return ConvertDictToTFDSOrSpec(dd, use_tfds_features=False)

def ConvertDictToTFDSFeatures(dd):
    """ Convert a nested dictionary of topic_names -> dicts to a nested dictionary of topic_names -> TFDS features. """
    return ConvertDictToTFDSOrSpec(dd, use_tfds_features=True)

def ConvertRos2DictToDict(dd):
    """
    dd: dictionary of topic_names -> ros2 messages
    """
    out_dd = {}
    for k, v in dd.items():
        out_dd[k] = ros2_to_dict(v)
        
    return out_dd