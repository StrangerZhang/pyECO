# Python Implementation of ECO

## Run demo
```bash
cd pyECO/eco/features/

python setup.py build_ext --inplace

cd pyECO/

python bin/demo_ECO_hc.py --video_dir path/to/video
```

## Benchmark results
#### OTB100  

| Tracker  | AUC           |
| -------- | ------------- |
| ECO_deep | 68.7(vs 69.1) |
| ECO_hc   | 65.2(vs 65.0) |

![](./figure/otb100.png)

## Note

we use ResNet50 feature instead of the original imagenet-vgg-m-2048

code tested on mac os 10.13 and python 3.6, ubuntu 16.04 and python 3.6 

## Reference
[1] Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael
    ECO: Efficient Convolution Operators for Tracking
    In Conference on Computer Vision and Pattern Recognition (CVPR), 2017
