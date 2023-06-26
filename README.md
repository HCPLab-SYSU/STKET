# Spatial-Temporal Knowledge-Embedded Transformer 

Implementation of papers: 

- Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation  
  Technical Report.   
  Tao Pu, Tianshui Chen, Hefeng Wu, Yongyi Lu, Liang Lin

## Usage
We construct this project based on [Yrcong' repository](https://github.com/yrcong/STTran).

Firstly, we download the directory of data and fasterRCNN in [Yrcong' repository](https://github.com/yrcong/STTran).

Then, we follow the instruction to compile some code for bbox operations.
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
Coming Soon
```
  
## Contributors
For any questions, feel free to open an issue or contact us:    

* putao537@gmail.com
