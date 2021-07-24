## 1. 从BVH中提取ROOT point
```
python get_root.py [bvh_file]
```

## 2. 求LiDAR轨迹到Mocap轨迹的RT矩阵
- 需要明确各自在第几帧
- 由于LiDAR轨迹的朝向和人的hip的朝向是一致的，因此仅需要***加上一个平移向量***，可使之位于hip的位置
- 目前平移向量在函数中固定了，可以根据实际情况输入
- 该函数仅用来求对应单帧的变换关系
```
python lidar_to_mocap.py [lidar_traj] [mocap_file]
```

## 3. Mocap和pose互相转换
- 下载工具[bvh-toolbox](https://github.com/OlafHaag/bvh-toolbox)
- 其中的BVH to CSV tables可将bvh转换成绝对坐标的pose

## TO DO
- [x] 从BVH中提取ROOT point <br>
- [x] 求LiDAR轨迹到Mocap轨迹的RT矩阵 <br>
- [x] 将Mocap转换到绝对pose的指令 <br>
- [ ] 将绝对pose转换到Mocap的指令 <br>
- [ ] 求整段LiDAR轨迹到Mocap轨迹的RT矩阵
- [ ] 将绝对位置的pose每帧分别乘不同的RT <br>
- [ ] 利用相机内外参，将绝对位置的pose投影回对应的相机
- [ ] 将投影结果可视化