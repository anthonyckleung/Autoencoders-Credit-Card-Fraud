# Credit card fraud detection with Semi-supervised learning

**Task**: Demonstrate the use of autoencoders to train a highly imbalance and 
high-dimensional data set of credit card records

This demonstration is done on a jupyter notebook.

### Prerequisites:
* Python 3
* Jupyter notebook
* Keras (TensorFlow backend)
* sklearn
* numpy, seaborn, pandas
* data source: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)



### Description
This notebook demonstrates the use of autoencoders as part of a semi-supervised 
approach to model a highly imbalance and high-dimensional data set. 
Exploratory analysis shows that the two target classes (fraudulent and non-frauduent)
are highly inseparable. It is by extracting latent variables of a trained autoencoder
on the original dataset shows the target classes can be separable. 

