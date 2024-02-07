# Project of Course CS577
### To run on your system
* Make a virtual enviroment using python 3.9 (using a virtualenv, not conda because we specific pip packages).
* Clone the repo
```
git clone https://github.com/FloatyDev/project577.git 
```
* Install dependencies using the following command.
```
pip install -r requirements.txt
```
### Scripts
The main scripts of this project are **mnist_finch.py** and **segment_finch.py**
#### mnist_finch.py
Offers estimation performance of FINCH on MNIST_10K dataset (with plots).
The metrics provided are:
* Normalized Mutual Information (NMI) Score
* Adjuster Rand Index (ARI) 
* Silhouette Score
* Calinski-Harabasz Score
* Davies-Bouldin Score
  
After running the script results will be saved on mnist_results directory. The script evaluates FINCH with both image raw pixels as well as CNN classification features (From ResNet50).
These weights are pre-extracted and saved on **data** directory of the project.
#### segment_finch.py
This script uses the foundation model [Segment Aything](https://github.com/facebookresearch/segment-anything) to produce quality binary masks of the provided image (some test images are put on **images** directory).
After that [FINCH](https://github.com/ssarfraz/FINCH-Clustering) is applied to the features made by either Hu-Moment or HOG extracting methods to cluster the binary masks by shape.
To run the script you have to specify the argument --algo that takes 2 values:
* hu (use Hu-Moments as the feature extraction method)
* hog (use hog as the feature extraction method, with predefined parameters)
So for example if you want to run the scripot using hu-moments as the feature extraction method you execute as:
```
python segment_finch.py --algo hu
```
Masks from segment-anything model are saved on **images** directory. The cluster results are saved on **sam_finch_results** directory. 

**Note**

Weights for SAM are downloaded automatically if they are not found.
