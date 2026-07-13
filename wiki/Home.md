# RDK 远近场排球接球系统 Wiki

本 Wiki 记录 Jetson NX 远场双目、RDK D435i 近场视觉与独立接球控制节点的确定接口。

- [系统架构与状态机](远近场接球系统设计.md)
- [ROS2 接口定义](ROS2接口定义.md)
- [千兆网与低延迟配置](千兆网低延迟配置.md)

基本原则：NX 和 D435i 视觉节点只发布数据，`catch_controller` 是唯一状态选择、
轨迹计算、复位和 `/auto/goal_pose` 发布者。
