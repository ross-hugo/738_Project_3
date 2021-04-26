# 738_Project_3
Machine Learning Project creating a neural network

Says One Neuron To Another

Datasets Used:

• https://deepai.org/dataset/mnist

• https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv

**Datasets**
For this project, I decided to mainly focus on solving classification problems. As a starting example, I decided to try and solve MNIST, a 
series of graphical representations of hand drawn numbers, by identifying which number they are trying to represent. MNIST is a classic 
machine learning problem, like the Iris dataset.

Next, I decided to try a different classification problem from standard and decided to use the adult census dataset and then determined
whether the person makes either above or below 50k in income with a number of features. 

_Thoughts on Datasets_
The MNIST dataset is a standard in the ML field, for this reason I believe that I should get a relatively high accuracy. Especially since
the MINST dataset has many samples and examples in it.

Next, with the adult census dataset, this is less standard and there are less samples than the MNIST dataset so I don't expect to do as 
well with classifying this dataset even though it is a binary classification.

**Results**
After creating a dynamic Neural Network model and able to change the number of inputs, outputs and hidden layers I was able to train and 
test different models on the two datasets. With a little bit of experimenting I found some models that perform relatively well on the 
two datasets. On the MNIST dataset I was able to acheieve a result of 96% and 75% for the MNIST and census datasets respectively.

One peculiar result of my neural network is that it takes many many epochs for the model to train, which is not in line with my expirience
when using other machine learning libraries (Tensorflow, PyTorch). This is most likely due to the activation function used (Stocahstic
Gradient Descnet or SGD) because the most standard now is Adaptive Moment Estimation or Adam. 

**Improvements**
For the MNIST dataset, I could have achieved a better result by implementing a CNN (Convelutional Neural Network) to gain insights from 
the iteraction between the pixels to view the higher level patterns among the illustration. While this might be a good improvement, 
the accuracy already achieved was 95% of the out of sample data on a multi-class classification. 

With the census data, it only achieved a 75% accuracy on the out of sample data on a binary classification. This can be greatly improved 
with a number of different changes. One of the biggest changes would be more feature engineering. For example, labeling the education of
each person in a standard manner so that the higher the number the higher the educaiton would help provide an insight to the model of how
those education numbers should be interpreted. There are other ways of improving the feature engineering as well. Another way to improve 
the census model would be to ensure that the data was balanced or by doing random sampling of the data because the distribution of the 
incomes is inherently not uniform. Finally, I could have kept some of the data that did not have a value for one of the cells by using a
filler value instead of just tossing it.
