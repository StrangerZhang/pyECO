# Python Implementation of ECO

## Run demo
cd pyECO/eco/features/

python setup.py build_ext --inplace

cd pyECO/

python bin/demo_ECO_hc.py --video_dir path/to/video

## Benchmark results
#### OTB100  

| Tracker           | AUC           |
| ----------------- | ------------- |
| ECO_deep          | 67.0(vs 69.1) |

## Note
we use ResNet50 feature instead of the original imagenet-vgg-m-2048

code tested on mac os 10.13 and python 3.6, ubuntu 16.04 and python 3.6 

## Citation
	@InProceedings{Danelljan_2017_CVPR,
		author = {Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael},
		title = {ECO: Efficient Convolution Operators for Tracking},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {July},
		year = {2017}
	}
