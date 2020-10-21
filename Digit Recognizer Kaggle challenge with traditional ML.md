# Part one: Digit Recognizer with traditional ML models

The Digit Recognizer challenge on Kaggle is a computer vision challenge where the goal is to classify images of hand written digits correctly.

While working through this computer vision project, I will follow a slightly adjusted Machine Learning project check list from Aurelien Geron's book "Hands-On Machine Learning with Scikit_Learn, Keras & TensorFlow". (Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (p. 35). O'Reilly Media. Kindle Edition.)

1.	Look at the big picture
2.	Get the data
3.	Discover and visualize the data to gain insights
4.	Prepare the data for Machine Learning algorithms
5.	Select, train and fine-tune models
6.	Conclusion

As with all coding posts, the full jupyter notebook can be found in my github repo below:

<https://github.com/John-Land/Digit-Recognizer-ML-competition-Kaggle>

In this first attempt we will use traditional Machine Learning algorithms to tackle the Digit Recognizer challenge. In part two below, we will use Deep Learning with tensorflow to try and improve on our results with a fully connected Neural Network and with a Convolutional Neural Network.

Part two can be found under below links.

Project page: <https://john-land.github.io/Digit-Recognizer-ML-competition-Kaggle-with-deep-learning-tensorflow>

Github: <https://github.com/John-Land/Digit-Recognizer-ML-competition-Kaggle-with-deep-learning-tensorflow>

## 1. Look at the big picture

Before looking deeper into the dataset, it will first be helpful to understand how image features are represented as numbers. The Kaggle page has a good expiation in the data tab.

The MNIST dataset we will be working on consist of 28 x 28 pixel grey-scale images (black and white images). Therefore one image consists of 28 x 28 = 784 pixels. Each pixel is considered a unique feature of an image, therefore each image in our dataset has 784 features. The values of each pixel range from 0 to 255 inclusive, with higher values indicating darker coloured pixels.

Note that due to the fact that these images are grey-scale images, we have 28 x 28 x 1 = 784 pixels per image. If these were coloured RGB images, one image would have three difference values for each pixel (red, green and blue pixel intensity values), and the features space per image would be 28 X 28 X 3 pixels.


## 2. Get the data

The data is provided on the Kaggle challenge page. <https://www.kaggle.com/c/digit-recognizer/data>

We will first import the data and check for any missing values and some basic information.


```python
# linear algebra
import numpy as np     

# data processing
import pandas as pd    

#data visualization
import matplotlib.pyplot as plt 
```


```python
training_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')

training_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 785 columns</p>
</div>



### 2.1. Data Structure


```python
training_data.shape, testing_data.shape
```




    ((42000, 785), (28000, 784))



Training data: 42000 rows and 785 columns -> Data on 42000 images, 784 pixel values and 1 label per image.

Testing data: 28000 rows and 784 columns -> Data on 28000 images, 784 pixel values and no labels per image.

Our predictions for the labels in the test set will be submitted to Kaggle later.


```python
print("Training Data missing values:"), training_data.isna().sum()
```

    Training Data missing values:
    




    (None,
     label       0
     pixel0      0
     pixel1      0
     pixel2      0
     pixel3      0
                ..
     pixel779    0
     pixel780    0
     pixel781    0
     pixel782    0
     pixel783    0
     Length: 785, dtype: int64)




```python
print("Testing Data missing values:"), testing_data.isna().sum()
```

    Testing Data missing values:
    




    (None,
     pixel0      0
     pixel1      0
     pixel2      0
     pixel3      0
     pixel4      0
                ..
     pixel779    0
     pixel780    0
     pixel781    0
     pixel782    0
     pixel783    0
     Length: 784, dtype: int64)



There are no missing values in the training and test set.

## 3. Discover and visualize the data to gain insights

The MNIST dataset we will be working on consist of 28 x 28 pixel grey-scale images (black and white images). Each row in our data set consists of all 784 pixels of one image and the label of the image.


```python
training_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 785 columns</p>
</div>



The below code visualizes one individual image by first reshaping the row in the data table for the individual image back into its original 28x28x1 pixel matrix, and then visualizing the pixel matrix for the image with matplotlib.


```python
photo_id = 1
image_28_28 = np.array(training_data.iloc[photo_id, 1:]).reshape(28, 28)
plt.imshow(image_28_28)
print("original image is a:", training_data.iloc[photo_id, 0])
```

    original image is a: 0
    


    
![png](output_15_1.png)
    



```python
photo_id = 50
image_28_28 = np.array(training_data.iloc[photo_id, 1:]).reshape(28, 28)
plt.imshow(image_28_28)
print("original image is a:", training_data.iloc[photo_id, 0])
```

    original image is a: 7
    


    
![png](output_16_1.png)
    



```python
X_train = training_data.iloc[:, 1:]
Y_train = training_data[['label']]
X_test = testing_data
X_train.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 784 columns</p>
</div>




```python
Y_train.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Prepare the data for Machine Learning algorithms

Before training our Machine learning algorithms we will do three pre-processing steps.

1.	The data will be scaled with the MinMaxScaler so all features fall in a range between 0 and 1.
2.	We will use PCA to reduce the feature space considerable, but still keeping 95% of the variance in the data, ensuring that not too much information is lost while reducing the feature space. This reduction in feature space will help reduce training time later.
3.	After PCA, we will use the StandardScaler to ensure that all new features from PCA have mean 0 and standard deviation 1.



```python
from sklearn.preprocessing import MinMaxScaler

#fit standard scaler to training set
scaler = MinMaxScaler().fit(X_train)

#transform training set
X_train_norm = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

#transform test set
X_test_norm = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
```


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95,svd_solver = 'full').fit(X_train_norm)
X_train_reduced = pd.DataFrame(pca.transform(X_train_norm))
X_test_reduced = pd.DataFrame(pca.transform(X_test_norm))
X_train_reduced.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>150</th>
      <th>151</th>
      <th>152</th>
      <th>153</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.594493</td>
      <td>-2.742397</td>
      <td>0.718753</td>
      <td>0.472985</td>
      <td>-0.317968</td>
      <td>1.919458</td>
      <td>-2.680278</td>
      <td>0.335527</td>
      <td>1.366855</td>
      <td>0.795994</td>
      <td>...</td>
      <td>-0.067206</td>
      <td>-0.228756</td>
      <td>0.062035</td>
      <td>-0.036426</td>
      <td>-0.072468</td>
      <td>-0.072641</td>
      <td>-0.022245</td>
      <td>-0.132045</td>
      <td>-0.213704</td>
      <td>-0.038427</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.672360</td>
      <td>-1.413927</td>
      <td>-1.967865</td>
      <td>1.315386</td>
      <td>-1.734820</td>
      <td>2.895702</td>
      <td>2.564217</td>
      <td>-0.692553</td>
      <td>-0.029491</td>
      <td>0.266064</td>
      <td>...</td>
      <td>-0.170333</td>
      <td>-0.142983</td>
      <td>0.160735</td>
      <td>-0.041835</td>
      <td>0.259718</td>
      <td>0.126622</td>
      <td>0.042587</td>
      <td>-0.072806</td>
      <td>0.046261</td>
      <td>0.160544</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.478017</td>
      <td>-1.152023</td>
      <td>0.263354</td>
      <td>0.306917</td>
      <td>-1.857709</td>
      <td>-1.268787</td>
      <td>1.716859</td>
      <td>-1.197560</td>
      <td>-0.765865</td>
      <td>-0.100494</td>
      <td>...</td>
      <td>-0.054321</td>
      <td>0.147798</td>
      <td>0.115108</td>
      <td>-0.050983</td>
      <td>-0.236928</td>
      <td>0.021208</td>
      <td>0.044639</td>
      <td>0.051697</td>
      <td>-0.053665</td>
      <td>0.080733</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.650022</td>
      <td>1.177187</td>
      <td>-0.251551</td>
      <td>2.979240</td>
      <td>-1.669978</td>
      <td>0.617218</td>
      <td>-1.192546</td>
      <td>1.083957</td>
      <td>-0.179872</td>
      <td>-1.158735</td>
      <td>...</td>
      <td>-0.018113</td>
      <td>0.267320</td>
      <td>0.372053</td>
      <td>-0.061314</td>
      <td>-0.377481</td>
      <td>0.186922</td>
      <td>-0.163961</td>
      <td>-0.041613</td>
      <td>-0.080466</td>
      <td>0.046466</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.543960</td>
      <td>-1.761384</td>
      <td>-2.151424</td>
      <td>0.739431</td>
      <td>-2.555829</td>
      <td>3.882603</td>
      <td>2.213753</td>
      <td>-1.003590</td>
      <td>0.489861</td>
      <td>0.696341</td>
      <td>...</td>
      <td>0.199446</td>
      <td>0.042179</td>
      <td>0.039327</td>
      <td>0.088611</td>
      <td>0.384118</td>
      <td>-0.297355</td>
      <td>0.003738</td>
      <td>0.197964</td>
      <td>-0.175501</td>
      <td>-0.041155</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 154 columns</p>
</div>




```python
X_train_reduced.shape
```




    (42000, 154)



PCA has reduced the feature space from 784 to 154 features, but still retained 95% of the variance and therefore most of the information in the data.


```python
from sklearn.preprocessing import StandardScaler

#fit standard scaler to training set
scaler = StandardScaler().fit(X_train_reduced)

#transform training set
X_train = pd.DataFrame(scaler.transform(X_train_reduced), columns = X_train_reduced.columns)

#transform test set
X_test = pd.DataFrame(scaler.transform(X_test_reduced), columns = X_test_reduced.columns)
```


```python
X_train.describe().round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>150</th>
      <th>151</th>
      <th>152</th>
      <th>153</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>...</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.9</td>
      <td>-2.9</td>
      <td>-3.1</td>
      <td>-3.5</td>
      <td>-3.1</td>
      <td>-3.2</td>
      <td>-3.3</td>
      <td>-3.4</td>
      <td>-3.5</td>
      <td>-3.6</td>
      <td>...</td>
      <td>-4.0</td>
      <td>-4.6</td>
      <td>-4.8</td>
      <td>-5.1</td>
      <td>-4.8</td>
      <td>-4.4</td>
      <td>-4.7</td>
      <td>-4.4</td>
      <td>-4.6</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.6</td>
      <td>...</td>
      <td>-0.7</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>-0.7</td>
      <td>-0.6</td>
      <td>-0.6</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.1</td>
      <td>-0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>0.0</td>
      <td>...</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.7</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.7</td>
      <td>0.7</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.7</td>
      <td>...</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.6</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.1</td>
      <td>3.0</td>
      <td>3.2</td>
      <td>3.6</td>
      <td>3.3</td>
      <td>3.3</td>
      <td>3.8</td>
      <td>3.8</td>
      <td>4.9</td>
      <td>3.6</td>
      <td>...</td>
      <td>4.7</td>
      <td>5.4</td>
      <td>5.1</td>
      <td>5.5</td>
      <td>5.5</td>
      <td>4.4</td>
      <td>5.5</td>
      <td>4.7</td>
      <td>4.6</td>
      <td>5.1</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 154 columns</p>
</div>



### 5. Select, train and fine-tune models

Now it's finally time to train our machine learning models.
As this is a classification task, we will take into considering below models.
1. Naive Bayes
2. LDA
3. QDA
4. Logistic Regression
5. Decision Tree
6. Ensemble of models 1-5

As we have no labels on the testing data, we will use the training data with 5 fold cross validation, to get an estimation of the out of sample accuracy. The model with the highest mean accuracy on all cross validation sets will be selected for prediction and submission to kaggle.


```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
import time

start = time.process_time()

clf_dummy = DummyClassifier(strategy = 'most_frequent')
cv_score = cross_val_score(clf_dummy, X_train, Y_train, cv=5).round(3).mean()


end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best score (accuracy): ', cv_score.round(3))
```

    training time in seconds: 0.28
    training time in minutes: 0.0
    Grid best score (accuracy):  0.112
    


```python
from sklearn.naive_bayes import GaussianNB

start = time.process_time()

clf1 = GaussianNB()
clf1.fit(X_train, Y_train.values.ravel());
cv_score_clf1 = cross_val_score(clf1, X_train, Y_train.values.ravel(), cv=5).mean()

end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best score (accuracy): ', cv_score_clf1.round(3))
```

    training time in seconds: 2.86
    training time in minutes: 0.05
    Grid best score (accuracy):  0.857
    


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

start = time.process_time()

clf2 = LinearDiscriminantAnalysis()
clf2.fit(X_train, Y_train.values.ravel());
cv_score_clf2 = cross_val_score(clf2, X_train, Y_train.values.ravel(), cv=5).mean()

end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best score (accuracy): ', cv_score_clf2.round(3))
```

    training time in seconds: 11.95
    training time in minutes: 0.2
    Grid best score (accuracy):  0.869
    


```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

start = time.process_time()

clf3 = QuadraticDiscriminantAnalysis()
clf3.fit(X_train, Y_train.values.ravel());
cv_score_clf3 = cross_val_score(clf3, X_train, Y_train.values.ravel(), cv=5).mean().round(3)

end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best score (accuracy): ', cv_score_clf3.round(3))
```

    training time in seconds: 11.22
    training time in minutes: 0.19
    Grid best score (accuracy):  0.945
    


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

start = time.process_time()

clf4 = LogisticRegression(max_iter = 10000)
grid_values = {'C': [0.01, 0.1, 1, 10, 100]}

# default metric to optimize over grid parameters: accuracy
grid_clf4 = GridSearchCV(clf4, param_grid = grid_values, cv=5)
grid_clf4.fit(X_train, Y_train.values.ravel());

end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best parameter (max. accuracy): ', grid_clf4.best_params_)
print('Grid best score (accuracy): ', grid_clf4.best_score_.round(3))
```

    training time in seconds: 453.69
    training time in minutes: 7.56
    Grid best parameter (max. accuracy):  {'C': 100}
    Grid best score (accuracy):  0.919
    


```python
from sklearn.tree import DecisionTreeClassifier

start = time.process_time()

clf5 = DecisionTreeClassifier()
grid_values = {'max_depth': [11, 13, 15, 17]}

# default metric to optimize over grid parameters: accuracy
grid_clf5 = GridSearchCV(clf5, param_grid = grid_values, cv=5)
grid_clf5.fit(X_train, Y_train.values.ravel())

end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best parameter (max. accuracy): ', grid_clf5.best_params_)
print('Grid best score (accuracy): ', grid_clf5.best_score_.round(3))
```

    training time in seconds: 260.5
    training time in minutes: 4.34
    Grid best parameter (max. accuracy):  {'max_depth': 15}
    Grid best score (accuracy):  0.821
    


```python
from sklearn.ensemble import RandomForestClassifier

start = time.process_time()


clf6 = RandomForestClassifier()
grid_values = {'max_depth': [11, 13, 15, 17], 'n_estimators': [75, 100, 125, 150]}

# default metric to optimize over grid parameters: accuracy
grid_clf6 = GridSearchCV(clf6, param_grid = grid_values, cv=5)
grid_clf6.fit(X_train, Y_train.values.ravel())

end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best parameter (max. accuracy): ', grid_clf6.best_params_)
print('Grid best score (accuracy): ', grid_clf6.best_score_.round(3))
```

    training time in seconds: 4913.48
    training time in minutes: 81.89
    Grid best parameter (max. accuracy):  {'max_depth': 17, 'n_estimators': 150}
    Grid best score (accuracy):  0.941
    


```python
from sklearn.ensemble import VotingClassifier

start = time.process_time()

eclf = VotingClassifier(estimators=[('GaussianNB', clf1),
                                    ('lda', clf2), 
                                    ('qda', clf3), 
                                    ('lr', clf4), 
                                    ('tree', clf5), 
                                    ('rf', clf6)],
                         voting='hard')
    
cv_score = cross_val_score(eclf, X_train, Y_train.values.ravel(), cv=5).mean().round(3)

end = time.process_time()

print('training time in seconds:', np.round(end - start,2))
print('training time in minutes:', np.round((end - start)/60, 2))
print('Grid best score (accuracy): ', cv_score.round(3))
```

    training time in seconds: 494.3
    training time in minutes: 8.24
    Grid best score (accuracy):  0.94
    

### 6. Conclusion

Based on the average cross validation error from five-fold cross validation, we get the below accuracies for each model.


```python
classifiers = ['GaussianNB', 
               'LinearDiscriminantAnalysis',
               'QuadraticDiscriminantAnalysis',
               'LogisticRegression', 
               'DecisionTreeClassifier', 
               'RandomForestClassifier', 
               'VotingClassifier']

scores = [cv_score_clf1.round(3), 
          cv_score_clf2.round(3), 
          cv_score_clf3.round(3), 
          grid_clf4.best_score_.round(3), 
          grid_clf5.best_score_.round(3), 
          grid_clf6.best_score_.round(3),
          cv_score.round(3)]

model_scores = pd.DataFrame(data= scores, columns = ['CV_Accuracy'], index = classifiers)
model_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CV_Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GaussianNB</th>
      <td>0.857</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.869</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.945</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.919</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.821</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.941</td>
    </tr>
    <tr>
      <th>VotingClassifier</th>
      <td>0.940</td>
    </tr>
  </tbody>
</table>
</div>



Based on the highest cross validation accuracy of 94.45%, we will use the QuadraticDiscriminantAnalysis classifier to submit our predictions to Kaggle.


```python
predictions = clf3.predict(X_test)
ImageId = np.array(range(1, X_test.shape[0]+1, 1))

output = pd.DataFrame({'ImageId': ImageId, 'Label': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
```

    Your submission was successfully saved!
    
