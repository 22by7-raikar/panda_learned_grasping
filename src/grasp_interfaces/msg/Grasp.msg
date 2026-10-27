# A parallel-jaw grasp for the Franka Hand.
# pose is the panda_hand_tcp pose: z = approach direction, y = closing direction,
# origin between the fingertips.
geometry_msgs/PoseStamped pose
float64 width                 # object width between the contacts [m]
float64 score                 # [0, 1], higher is better
int32 object_id
string source                 # "efd" (classical) or "frcnn" (learned)
geometry_msgs/Point contact_a
geometry_msgs/Point contact_b
# Axis-aligned bounding box of the object in the pose's frame (used to attach it to the hand)
geometry_msgs/Point object_center
geometry_msgs/Vector3 object_dimensions
