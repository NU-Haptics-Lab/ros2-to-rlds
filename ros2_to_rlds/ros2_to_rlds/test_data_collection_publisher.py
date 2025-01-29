import rclpy
from rclpy.node import Node
from std_msgs.msg import *
from ros2_to_rlds_msgs.msg import *
import math

class Test(Node):

    def __init__(self):
        super().__init__('test_data_collection_publisher')

        # Make publisher for most common standard msg besides string
        self.pub1 = self.create_publisher(Int32, "test/data_collection/int32", 10)
        self.pub2 = self.create_publisher(Float64, "test/data_collection/float64", 10)
        self.pub3 = self.create_publisher(Bool, "test/data_collection/bool", 10)
        
        # array publishers
        self.pub4 = self.create_publisher(Int32array, "test/data_collection/int32array", 10)
        self.pub5 = self.create_publisher(Float64array, "test/data_collection/float64array", 10)
        
        # header
        self.pub6 = self.create_publisher(Header, "test/data_collection/header", 10)
        
        
        # Timer
        period = 1.0 / 10.0
        self.timer = self.create_timer(period, self.TimerCB)
        
        # Parameters
        self.bias = 0.0
        self.amplitude = 0.5
        
    def GetNow(self):
        now = self.get_clock().now().seconds_nanoseconds()
        seconds = now[0]
        nanoseconds = now[1]
        out = float(seconds) + float(nanoseconds) / 1e9
        return out
        
    def TimerCB(self):
        # calculate sinusoid value
        val = self.bias + self.amplitude * math.sin(self.GetNow())
        intval = int(10.0 * val)
        
        # publish to int32
        msg = Int32()
        msg.data = intval
        self.pub1.publish(msg)
        
        # publish to float64
        msg = Float64()
        msg.data = val
        self.pub2.publish(msg)
        
        # publish to bool
        msg = Bool()
        msg.data = val > 0.0
        self.pub3.publish(msg)
        
        # publish to int32 multi array
        msg = Int32array()
        msg.data = 3 * [intval]
        self.pub4.publish(msg)
        
        # publish to float64 multi array
        msg = Float64array()
        msg.data = 3 * [val]
        self.pub5.publish(msg)
        
        # header
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = "test_data_collection_publisher"
        self.pub6.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    test_ = Test()

    rclpy.spin(test_)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    test_.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
