# Image classfication with CNN's
In this notebook, we are trying to classify the ten different object images from the 'CIFAR-10' dataset and improve the accuracy rate of VGG-16. VGG has only convolutional layers and max-pooling layers. We add some deep learning techniques like dropout early stopping data augmentation, and Batch Normalization. We search for how these techniques effect on accuracy rate. We analyze our accuracy with a confusion matrix and f1-score. Lastly, we search what we will do to get better accuracy. 

1. Introduction
Over the past ten years, artificial intelligence and machine learning have become a subject that attracts everyone's attention. Why ten years? This question has two answers: Better GPU unit and more data. With better GPU units, we can train our models very quickly. With more data, our model accuracy rate will be better.

 We can use ML (Machine Learning) for repeating tasks. Nowadays, most people using ML for Computer Vision, NLP (Natural Language Processing), Robotics, Data Mining, and Genetic Algorithms.
 Image Classification is a CV (Computer vision) problem. Image classification is used in many areas such as medicine, the defense industry, agriculture, and autonomous vehicles.
  DL has multiple layers of artificial neural networks, imitating the human brain's working system. Therefore, Deep Learning is a beneficial technique for image classification. Also, Convolutional neural networks were created for image classification.

CNN's can take in an input image, assign importance (learnable weights and biases) to various aspects in the picture, and differentiate one from the other. 

2. Dataset
In this paper, CNN models have been used for image classification over the 'CIFAR-10' dataset. CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. This paper aims to classify the ten different classes using a CNN. 

3. Used Techniques and Environment
In DL, there are many techniques. These techniques reduce overfitting, increases both accuracy and validation rate. In this paper, these techniques are used: Dropout, L2 regularization, Early Stopping, Data Augmentation, and Batch Normalization. I used Adam optimizer for the base model, and sparse categorical cross-entropy was used for the loss function. 
For the environment, I used Jupyter Notebook with Python3. Also, I used Tensorflow and Keras.

3.1 Convolutional Layer
You can visit my other repo:  https://github.com/caglarsevinc/Face-Mask-Detection-with-Pytorch/blob/main/README.md#6-modelling

3.2 Dropout
The dropout technique is one of the best regularization techniques for deep learning. Dropout is a simple algorithm; in the training phase, every neuron has a probability. Dropout meaning is ignored neurons with respect to hyperparameter known as dropout rate. 

3.3 L2 Regularization
L2 regularization is also known as ridge regression. L2 regularization is regularized version of linear regression. A regularization term added to the cost function. If the hyperparameter is very large, then all weights end up very close to zero, and the result is the flat line going through the data's mean. Increasing leads to flattening predictions; this reduces model variance but increases bias. 

3.4 Early Stopping
When the difference between accuracy score and validation score increases, the model leads to overfitting. If we observe overfitting, early stopping is one of the techniques that reduce overfitting. In this case, when the validation error starts to increase, training is stopped, and the previous step is returned.

3.5 Data Augmentation
If we have more data, our model will learn better. With better learning, the model's training accuracy and validation accuracy will improve so that they provide overfitting. We can augment data with scaling, cropping, flipping, rotation, and color augmentations. If we apply the data augmentation like in the below, instead of one image, we got nine pictures. 

3.6 Batch Norm
If data are not normalized, a model needs more data, and the learning rate has to be slow. As we see in the counterplot below, if we don’t use normalization and the model’s learning rate is big, our model job will be harder to find local minima. In neural networks, batch norm normalizes after layers. 

3.7 Optimizer Model
Usually, optimizer models are used to find local minima in nonlinear problems. We can also use gradient descent-based adaptive methods. But some of the optimizer models work much better than gradient descent-based adaptive methods. In this paper, we used Adam for the optimizer model. Adam is the combination of RMSprop and stochastic gradient descent with momentum. It uses squared gradients to scale learning rates like RMSprop instead of the gradient itself like SGD with momentum. 

4. Result and Comparison 
![image](https://user-images.githubusercontent.com/45899874/152239404-eeca8ca5-113a-4a31-aeac-0a9a5961437a.png)
In the application part of this paper, we understand what techniques are used and how to apply them. In this part, we will search how these techniques are effective to other techniques. 

4.1 Confusion Matrix 
The confusion matrix is used to measure the performance of the algorithm for classification tasks. In the confusion matrix for two label classification, we create a table with two variables, ‘actual’ and ‘predicted’. The table has four conditions: False positives, false negatives, true positives, and true negatives. Nevertheless, in our task, we have multilabel classification. In multilabel classification’s confusion matrix, we only have two conditions: positive and negative. But we need the other conditions while calculating the F1 score. 

![image](https://user-images.githubusercontent.com/45899874/152239596-6ab44ade-cc30-484a-9ac7-91abf51db04e.png)
4.2 F1 Score
For calculating the F1 score, we need two variables. These are precision and recall. Precision is the proportion of correct predictions to all predictions. The recall is the proportion of true positives to all true labeled examples. F1 score uses harmonic means of precision and recall. 

5.CONCLUSION 
We can use DL in many areas, and it is a beneficial technique for repeating tasks, but for specific areas like medicine, the defense industry, or the stock market, DL has to be more accurate. Because in these areas, our error proportion has to be significantly less, and we do not have enough information about what the algorithm thinks. We can give an example about the defense industry. Think about what can happen if the algorithm makes a wrong prediction about a worker with a hoe. We can improve our model with Pre-Trained Algorithms, Transfer Learning, Ensemble Learning, Semantic Segmentation, etc. 

5.1 Pre-Trained Algorithms and Transfer Learning 
In computer vision, there are so many algorithms and models. We can use these models with two techniques. Those are pre-trained models and Transfer Learning. For pre-trained models, all we have to do is import a library and call a function name. The purpose of using pre-trained models is the cost of GPU. For Transfer Learning, we take a trained model for one task and re-use it for the related task. So Transfer Learning can be very useful for image classification. 

5.2 Ensemble Learning
We can use many machine learning techniques for solving problems. Ensemble Learning permits merging the models and create a new model with better accuracies. For image classification, we can use Ensemble Learning.

5.3 What We Achive
We show that our DL techniques help both the accuracy and validation scores by adding and removing techniques from the model. For example, we made comparisons like without Dropout Technique or with Dropout Technique without Batch Normalization Technique. We made these comparisons in fifty epochs. In our best model, we got %86,5 on accuracy score and %85,3 on validation score. Then we take the best model and train with 100 epochs. In the last train, we got %89,3 accuracy score and %87,8 on the validation score. When we look at the confusion matrix, we can see that our model cannot precisely distinguish between cats and dogs. Our model is successful in recognizing automobiles, ships, trucks, frogs, and horses. 














