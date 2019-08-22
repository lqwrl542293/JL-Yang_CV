# 环境配置
Python 2.7
Pytorch 0.4.0以及以上
torchvision 0.2.0 （其他版本会出运行问题）
[KITTI stereo dataset (2015 & 2012)](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
[Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

## 注意事项
```
Scene flow 下载内容
下载三个子集（FlyingThings3D, Driving, and Monkaa）的RGB cleanpass images和对应的disparity 。把它们放到一个父目录下
重新命名为：”driving_frames_cleanpass”, “driving_disparity”, “monkaa_frames_cleanpass”, “monkaa_disparity”, “frames_cleanpass”, “frames_disparity”
```

- 如果用0.4.1+版本的话，在upsample 函数参数重需要加入“align_corner=True”
- 如果使用torchvision=0.2.0+的话，RGB图片读取时不能加上”.astype(float32’)”。
- 输出视差最好乘1.17，see [#135](https://github.com/JiaRenChang/PSMNet/issues/135)  and  [#113](https://github.com/JiaRenChang/PSMNet/issues/113) 

# 网络结构
本程序基于PSMNet的特征提取部分进行的更改，将DeeplabV3的空洞卷积的思想应用到了立体匹配特征提取中，总体的特征提取部分架构如下：

![](https://s2.ax1x.com/2019/08/22/mdY0ds.png)

# 训练
在”run.sh”文件中有样例代码。
