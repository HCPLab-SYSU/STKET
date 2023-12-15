# Spatial-Temporal Knowledge-Embedded Transformer 

Implementation of papers: 

- [Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation](https://arxiv.org/pdf/2309.13237.pdf)  
  IEEE Transactions on Image Processing (T-IP)，2024.   
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
  title={Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation},
  author={Pu, Tao and Chen, Tianshui and Wu, Hefeng and Lu, Yongyi and Lin, Liang},
  journal={IEEE Transactions on Image Processing},
  year={2024}
}
```
  
## Contributors
For any questions, feel free to open an issue or contact us:    

* putao537@gmail.com
