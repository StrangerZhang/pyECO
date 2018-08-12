# python ECO

# Note
cd /path/to/features/

python setup.py build_ext --inplace

cd /project/root

python bin/demo_ECO_hc.py

# benchmark results
OTB100  AUC 68.8(vs 70.0 original paper result)

# Citation
	@InProceedings{Danelljan_2017_CVPR,
		author = {Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael},
		title = {ECO: Efficient Convolution Operators for Tracking},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {July},
		year = {2017}
	}
