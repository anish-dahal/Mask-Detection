Mask-Detection using pre-trained RestNet model from pytorch

## Dataset
I have created a model that detects face mask trained on 7553 images with 3 color channels (RGB).

Data set consists of 7553 RGB images in 2 folders as with_mask and without_mask. Images are named as label with_mask and without_mask. Images of faces with mask are 3725 and images of faces without mask are 3828. 


Dataset path: [Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
```
data/
    -with_mask/
        -with_mask_1.jpg
        -with_mask_2.jpg
        -....
    -without_mask/
        -without_mask_1.jpg
        -without_mask_2.jpg
        -....
```
We use Transfer learning for mask detection and the network used for the classification is ResNet34

## ResNet
Resnet34 is one pre-trained model that is basically a 34-layer convolutional neural network that has been pre-trained using the ImageNet dataset. The models are trained on the 1.28 million training images, and evaluated on the 50k validation images for 1000 distinct classes. However, it differs from standard neural networks in that it incorporates residuals from each layer in the succeeding connected layers.
Input size: (3, 224, 224)

Paper. ([source](https://arxiv.org/pdf/1512.03385.pdf))

## Demo

MTCNN is used for face detection and the face in the image has been as an input for the model after using transformation which was used to train pre-trained model ResNet34 model.

