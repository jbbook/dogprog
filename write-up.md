### Project definition

#### Project overview 
In this project, different algorithms of computer vision are explored to identify humans, dogs, and specific dog breeds in images.
I show that convolutional neural networks (CNNs) can obtain very good prediction accuracy with simple architectures.

At a result of this project, a web app is supplied that takes any user-supplied image input. If a dog is detected in the image, the algorithm will provide an estimate of the dog's breed. If instead a human is detected, the algorithm will provide the breed that is most similar to the human provided. If neither dog nor human is detected, an error message is displayed. 

#### Problem Statement
There is the need of three distinct algorithms needed. 
- A human detector, which is implemented using the OpenCV python library
- A dog detector, which is implemented using a pre-trained CNN.
- An algorithm that can provid a best estimate for the dogs breed, which is implemented using a custom CNN building on ResNet. 

#### Metrics
The defining metric is the accuracy, i.e. how often the correct breed is predicted from a user image. The accuracy is given in percent, and the final algorithms provides over 80% accuracy on the test data set.

### Analysis

#### Data Exploration & Visualization
Exloratory analysis has given that the provided dog image data set consitsts of 8351 dog images and 133 dog breed categories. The dataset is split 80/10/10% for training/validation/test.

### Methodology

#### Data Preprocessing
The given dog data set is very clean, so minimal data preprocessing is needed. Images are converted into tensors for usage in tensorflow. Using keras.preprocessing library, images are converted to 3d arrays (RGB) and then to 4D tensors using np.expand_dims.

#### Implementation 
Human detection
Human detection is implemented using the opencv library and a cascade classifier called haarscascades. This classifier identifies typical facial patterns in images and can detect human faces in frontal face with very high sensitivity. (code cells 3 and 4 in dog_app.ipynb)

Dog detection
Dog detection is implemented using the pre-trained ResNet50 model keras supplies. ResNet50 is trained on the ImageNet data set and can classify images into 1000 different classes. A model output between 151 and 268 corresponds to different dog breeds. The dog detector function is provided in code cell 9 of dog_app.ipynb.

Dog breed classification
For classifications of dog breeds, first, a custom CNN is trained, using 3 convolutional layers, 1 max pooling layers, 1 dropout layer and 1 dense layer. The exact specifications are defined in code cell 13. With this simple CNN, an accuracy of over 10% can be reached.

#### Refinement
For improvement of the accuracy, transfer learning is used. To use transfer learning, a pre-trained model in used (in this case ResNet50), and its bottleneck features are extracted. Then, a GlobalaveragePooling2D layer is applied and finally a dense layer with 133 outputs for final classification task. By using a pre-trained network that is trained on very similar data as our data set, we can benefit from the extracted features in its convolutional layers and reduce computational time significantly while maintaining high accuracy. Because the pre-trained ResNet50 is a much more complicated and deeper CNN than the custom-made simple CNN previously used for breed classification, accuracy is raised to over 80% with minimal training time needed.
To summarize, transfer learning is accomplished by
i - loading a pre-trained CNN model, in this case ResNet50 trained on ImageNet
ii - compute the bottleneck features of training images using the pre-trained model.  
iii - training a new model that takes the bottleneck features as input and provides the classification (in this case, dog breed) as output.

### Results
#### Model Evaluation and Validation
The CNNs are evaluated using dedicated validation and test sets. The final CNN reaches impressive 80% accuracy using pre-trained ResNet50 and transfer learning.

### Conclusion

#### Reflection
CNNs are a really impressive tool for image classification tasks. Using transfer learning, a new model can be trained from an existing model with very little computational cost. One major downside is the tendency to overfit data. One clear sign of overfitting is that the accuracy gets worse 
#### Improvements
There are lots of ways how accuracy could be further refined, including
- Trying out different pre-trained CNNs for bottleneck extraction.
- Extensive hyper-paratemer tuning.
- Performing data augmentation.