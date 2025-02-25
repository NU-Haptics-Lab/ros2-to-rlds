# To run
`ros2 run ros2-to-rlds ros2_rlds_server`

# LIMITATIONS
Can't use any text in the actions AT ALL, see https://github.com/google-deepmind/envlogger/issues/15

Can't use any sequences, see https://github.com/google-deepmind/envlogger/issues/14

I haven't fully tested it, but I think that if a message has any empty array (ex: `JointState.effort = []`) then the dataset will crash when you try to load it. See `./test/test_ds_load_joint_state_msg.py`.

# Tested & Working ROS2 message types
```
stds_msgs/msg/Int32
std_msgs/msg/Float64
sensor_msgs/msg/Image
```

# DOES NOT WORK
```
std_msgs/msg/*MultiArray
std_msgs/msg/Bool
*String
sensor_msgs/msg/JointState
```