"""Small ROS parameter service for the pinned FAST-LIVO2 camera loader."""
import rclpy
from rclpy.node import Node


def main():
    rclpy.init()
    node=Node('parameter_blackboard',automatically_declare_parameters_from_overrides=True)
    print('CAMERA_PARAMETER_SERVICE_READY',flush=True)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
