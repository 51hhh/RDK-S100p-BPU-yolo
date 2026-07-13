std_msgs/Header header
uint32 source_epoch
uint64 sequence
int64 offset_ns
uint64 uncertainty_ns
bool synchronized
uint8 servo_state
string clock_source

uint8 SERVO_UNKNOWN=0
uint8 SERVO_UNLOCKED=1
uint8 SERVO_LOCKED=2
uint8 SERVO_HOLDOVER=3
