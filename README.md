# Learning detection with diverse proposals
A Tensorflow implementation of [Learning detection with diverse proposals](http://openaccess.thecvf.com/content_cvpr_2017/papers/Azadi_Learning_Detection_With_CVPR_2017_paper.pdf) by Nuri Kim. This repository is based on the Faster R-CNN implementation available [here](https://github.com/endernewton/tf-faster-rcnn).

### Performance
Here, Tested on VOC 2007 test set and VGG16 is used as a backbone network.
The crowd sets consist of images containing at least one object having overlap with other object in the same category over the threshold (0.3).

#### Trained with VOC2007 trainval set:
| Method | mAP | mAP on Crowd |
|:-:|:-:|:-:|
| Faster R-CNN | <b>71.4%</b> | 57.7% |
| Faster R-CNN + LDPP | 70.9% | <b>61.8%</b> |

#### Trained with VOC0712 trainval set:
| Method | mAP | mAP on Crowd |
|:-:|:-:|:-:|
| Faster R-CNN | 75.8% | 62.0% |
| Faster R-CNN + LDPP | <b>76.6%</b> | <b>64.5%</b> |


### Prerequisites
  - A basic Tensorflow installation. I used tensorflow 1.7.
  - Python packages you might not have: `cython`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)).

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/bareblackfoot/lddp-tf-faster-rcnn.git
  ```

3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

4. Install the [Python COCO API](https://github.com/pdollar/coco). The code requires the API to access COCO dataset.
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```

### Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets (Part of COCO is done). The steps involve downloading data and optionally creating soft links in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

### Test with pre-trained models
1. Download pre-trained model
  - Google drive [here](https://drive.google.com/drive/folders/1lrB8inqhWvK6_xhkbgbt0XBVsEZvbECl?usp=sharing).
  if you want to test the model trained on VOC 2007, the trained model is [here](https://drive.google.com/drive/folders/1DK0fBcilf450jeq5He0grefuWiMishJ9?usp=sharing).

2. Create a folder and a soft link to use the pre-trained model
  ```Shell
  NET=res101
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  cd ../../..
  ```
3. Test with pre-trained VGG16 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_lddp.sh $GPU_ID pascal_voc_0712 vgg16
  ```

### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_lddp.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_lddp.sh
  # Examples:
  ./experiments/scripts/train_lddp.sh 0 pascal_voc_0712 vgg16
  ./experiments/scripts/train_lddp.sh 1 coco res101
  ```

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_lddp.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_lddp.sh
  # Examples:
  ./experiments/scripts/test_lddp.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_lddp.sh 1 coco res101
  ```

5. You can use ``tools/reval.sh`` for re-evaluation


By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```
