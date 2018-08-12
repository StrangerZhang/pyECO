# Python implementation of ECO

## Run demo
cd /path/to/features/

python setup.py build_ext --inplace

cd /project/root

python bin/demo_ECO_hc.py

## benchmark results
#### OTB100  

| Tracker           | AUC           |
| ----------------- | ------------- |
| ECO_deep          | 68.8(vs 70.0) |
| ECO_deep_original | 60.3(vs 65.0) |

## Note
we use ResNet50 feature instead of the original imagenet-vgg-m-2048

code tested on mac os 10.13 and python 3.6

## Citation
	@InProceedings{Danelljan_2017_CVPR,
		author = {Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael},
		title = {ECO: Efficient Convolution Operators for Tracking},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {July},
		year = {2017}
	}
