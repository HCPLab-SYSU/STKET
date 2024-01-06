# Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation

Implementation of papers: 

- [Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation](https://ieeexplore.ieee.org/document/10375886)  
  IEEE Transactions on Image Processing (IEEE TIP)，2024.   
  Tao Pu, Tianshui Chen, Hefeng Wu, Yongyi Lu, Liang Lin

![](./figures/framework.png)

## Usage
Firstly, we download the directory of data and fasterRCNN in [Yrcong' repository](https://github.com/yrcong/STTran).

Then, we follow the instructions to compile some code for bbox operations.
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```

For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch


## Citation
```
@article{Pu2024STKET,
  author={Pu, Tao and Chen, Tianshui and Wu, Hefeng and Lu, Yongyi and Lin, Liang},
  title={Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation},
  journal={IEEE Transactions on Image Processing},
  volume={33},
  pages={556-568},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TIP.2023.3345652}
}

@inproceedings{Pu2023VidSGG,
  author={Pu, Tao},
  title={Video Scene Graph Generation with Spatial-Temporal Knowledge},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023},
  pages={9340--9344},
  publisher={Association for Computing Machinery},
  doi={10.1145/3581783.3613433}
}
```
  
## Contributors
For any questions, feel free to open an issue or contact us:    

* putao537@gmail.com
