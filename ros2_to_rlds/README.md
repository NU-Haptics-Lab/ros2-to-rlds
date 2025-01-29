# To run
`ros2 run ros2-to-rlds ros2_rlds_server`

# LIMITATIONS
Can't use any text in the actions, see https://github.com/google-deepmind/envlogger/issues/15

Can't use any sequences, see https://github.com/google-deepmind/envlogger/issues/14

# Tested & Working ROS2 message types
```
stds_msgs/msg/Int32
std_msgs/msg/Float64
```

# DOES NOT WORK
```
std_msgs/msg/*MultiArray
std_msgs/msg/Bool
*String
```