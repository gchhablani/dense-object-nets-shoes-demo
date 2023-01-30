# Dense Object Nets Shoes Demo
This repository contains the demo for dense object nets for the course - Deep Learning for Robotics, Spring'23 at Georgia Tech.


## Steps

1. Clone the git repository:
```sh
git clone https://github.com/RobotLocomotion/pytorch-dense-correspondence.git
```

2. Copy the Python3 version of the download script to `pytorch-dense-correspondence` repo.
```sh
cp scripts/download_pdc_data.py <path to repo>/config/download_pdc_data.py
```

3. Download the data
```sh
cd <path to repo>
python config/download_pdc_data.py ./config/dense_correspondence/dataset/composite/shoes_all.yaml /content/pytorch-dense-correspondence/
```

4. Download model checkpoint and unzip
```sh
wget https://data.csail.mit.edu/labelfusion/pdccompressed/trained_models/stable/shoes_consistent_M_background_0.500_3.tar.gz
tar -xvf shoes_consistent_M_background_0.500_3.tar.gz
rm shoes_consistent_M_background_0.500_3.tar.gz
```
5. 