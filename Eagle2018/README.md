# Eagle_System
Initially created in December, 2018 for the Innovation Project for College Students of Beijing University of Posts and Telecommunications. The project is fully named as **"The Eagle" Remote Sensing Reconnaissance System**.

![](https://ws3.sinaimg.cn/large/006tNc79ly1g2ckg16156j30u00u0mzz.jpg)

It's a demo designed with GUI capable to access the picture database on a remote terminal and do simple search based on given conditions. The remote terminal can also run deep learning algorithms independently when necessary to get semantic information of an image.

## Overall Procedure

The system is designed to have 2 different modes concerning different demands and user scenarios: personal mode and corporate mode. 

There are totally 2 ends involved, namely local PC and remote TX2 (short for NVIDIA Jetson TX2). It is worth noting that TX2 can be replaced by any embedded platform with stronger computational capability, such as latest NVIDIA Jetson Xavier.

### Personal Mode

Personal mode has not completed.

### Corporate Mode

The procedure of corporate mode is described as the sequece graph below. 

![](https://ws4.sinaimg.cn/large/006tKfTcly1g0ztdhzf3kj31050u0k4n.jpg)

## File Structure

### GUI on Local PC

Code for GUI: `/PC/GUI.py`

Code to connect PC to TX2: `/PC/connection_local.py`

### Supporting Code on TX2

Code executed remotely on TX2: `/TX2/connection_remote.py`

### Picture Database

Supplementary code for P-DB: `/TX2/PDB/utils/`.

Overall record for the whole P-DB is stored as `/TX2/PDB/PDB.txt`. The content format of each line is shown as below.
```
10 75 1 /TX2/PDB/pics/2_12_11_RGB.png /TX2/PDB/pics/2_12_11_SS.png /TX2/PDB/pics/2_12_11_OD.png /TX2/PDB/pics/2_12_11_INFO.txt
```
`10 75 1` stand for building-covering rate, vegetation rate and number of vehicles, respectively.

Within `/TX2/PDB/pics/` is the picture database of the project. So far we only 100 sets of data and each set consists of a 512x512 urban aerial image (from ISPRS Potsdam dataset), the result of semantic segmentation and that of object detection . Semantic segmentation algorithm is  trained on Potsdam dataset itself while object detection is trained on NWPU VHR-10 dataset thus performance is unsatisfactory.

- Original RGB images are named as `X_XX_RGB.png`.
- Images segmented as `X_XX_SS.png`.
- Images detected as `X_XX_OD.png`.
- Single image's semantic information retrieved as `X_XX_X_INFO.txt`, with content formatted below.

```
房屋占地比： 7
草地占地比： 80
汽车数量： 0
```

For convenience, a preliminary dataset is provided for demonstration of demo. `/TX2/PDB/pics/` and `/TX2/PDB/PDB.txt` is attached with [Baidu Netdisk](https://pan.baidu.com/s/1mLTFbAPvMdDI5UempxEX5w) (PW: 0yj6). This dataset is based on Potsdam published by ISPRS. You can check [this website](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html) for more information. If you are interested in human-labelled object detection ground truth of Potsdam, feel comfortable to contact.

## To Ultilize "The Eagle"

1. Have all the necessary site-packages installed.
2. Put P-DB files at the remote server and have the path recorded.
3. Add your SSH configurations and recorded path to the preamble area of corresponding code files.
4. Put your PC and TX2 under the same LAN.
5. Run `GUI.py` locally.


## What's Next

- [x] Replace current object detection reults with better ones
- [ ] Refine the code
- [ ] Better GUI design
- [ ] Complete personal mode

## Acknowledgement

Sincere tribute to all memebers that have contributed to this repo (names not listed in order): *Haoran Cui*, *Yumin Zheng*, *Qingtian Zhu*, *Xi Xia*, *Xingyi Li*.

Additional, tribute to members having contributed to the dataset used to train object detection algorithms which is absent in the officially published dataset (names not listed in order): *Jingjing Wei*, *Kai Kang*, *Zhaofeng Wang*, *Yijian Liu*, *Wenqian Cui*.