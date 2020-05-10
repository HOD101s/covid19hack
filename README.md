# 19Hack Project
An AI based identification of non-complying or Pneumonia-infected individuals by classifying CT scan images. It can be used as preliminary test for people showing symptoms to cut the strain on existing medical facilities. 

# Algorithms implemented
We've implemented transfer learning using pre trained models like **InceptionV3** and **VG19**. They speed up the time it takes to develop and train a model by reusing these pieces or modules of already developed models. This helps speed up the model training process and accelerate results. 

## InceptionV3
This image recognition model consists of two parts:
 - Feature extraction part with a convolutional neural network
 - Classification part with fully-connected and softmax layers
In transfer learning, we build a new model to classify the original dataset, where we reuse the feature extraction part and re-train the classification part with your dataset.
![InceptionV3 model](https://user-images.githubusercontent.com/40513848/81470679-a7744f80-9209-11ea-87f4-6a0ba1bbf1a5.png)    
#### Model Summary
```
Model: "sequential_41"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
inception_v3 (Model)         (None, 8, 8, 2048)        21802784  
_________________________________________________________________
global_average_pooling2d_41  (None, 2048)              0         
_________________________________________________________________
dense_119 (Dense)            (None, 32)                65568     
_________________________________________________________________
dropout_69 (Dropout)         (None, 32)                0         
_________________________________________________________________
dense_120 (Dense)            (None, 16)                528       
_________________________________________________________________
dropout_70 (Dropout)         (None, 16)                0         
_________________________________________________________________
dense_121 (Dense)            (None, 2)                 34        
=================================================================
Total params: 21,868,914
Trainable params: 21,834,482
Non-trainable params: 34,432
_________________________________________________________________
```

## VGG16
The VGG16 model was developed by the Visual Graphics Group (VGG) at Oxford. By default, the model expects color input images to be rescaled to the size of 224×224. The model is specifically trained for a more than a million type of images
![VGG16 model](https://user-images.githubusercontent.com/40513848/81470734-08038c80-920a-11ea-8ce9-b0916e1fce3c.png)    
#### Model Summary
```
Model: "sequential_43"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 9, 9, 512)         14714688  
_________________________________________________________________
global_average_pooling2d_43  (None, 512)               0         
_________________________________________________________________
dense_125 (Dense)            (None, 32)                16416     
_________________________________________________________________
dropout_73 (Dropout)         (None, 32)                0         
_________________________________________________________________
dense_126 (Dense)            (None, 16)                528       
_________________________________________________________________
dropout_74 (Dropout)         (None, 16)                0         
_________________________________________________________________
dense_127 (Dense)            (None, 2)                 34        
=================================================================
Total params: 14,731,666
Trainable params: 16,978
Non-trainable params: 14,714,688
_________________________________________________________________
```


## VGG19
Similar to VGG16, the main difference between the “VGG-19 Neural Network” and the “VGG-16 Neural Network” is that, this type of network is 19 layers deep and that type of network was 16 layers deep respectively.
![VGG19 model](https://user-images.githubusercontent.com/40513848/81470765-31bcb380-920a-11ea-8e0c-49245858443d.png)    
#### Model Summary
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg19 (Model)                (None, 7, 7, 512)         20024384  
_________________________________________________________________
average_pooling2d_1 (Average (None, 2, 2, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               262272    
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 258       
=================================================================
Total params: 20,286,914
Trainable params: 262,530
Non-trainable params: 20,024,384
_________________________________________________________________
```

# Comparison of Techniques
<table>
    <thead>
        <tr>
            <th rowspan=2>Metrics</th>
            <th colspan=2>InceptionV3</th>
            <th colspan=2>VGG16</th>
            <th colspan=2>VGG19</th>
        </tr>
        <tr>
          <th>0</th>
          <th>1</th>
          <th>0</th>
          <th>1</th>
          <th>0</th>
          <th>1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Precision</td>
            <td>0.41</td>
            <td>0.63</td>
            <td>0.37</td>
            <td>0.62</td>
            <td>0.37</td>
            <td>0.62</td>
        </tr>
        <tr>
            <th>Recall</td>
            <td>0.23</td>
            <td>0.80</td>
            <td>0.20</td>
            <td>0.79</td>
            <td>0.19</td>
            <td>0.81</td>
        </tr>
        <tr>
            <th>F1-score</td>
            <td>0.30</td>
            <td>0.71</td>
            <td>0.26</td>
            <td>0.70</td>
            <td>0.25</td>
            <td>0.70</td>
        </tr>
    </tbody>
</table>
