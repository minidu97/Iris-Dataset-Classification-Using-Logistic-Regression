# Iris-Dataset-Classification-Using-Logistic-Regression
This project implements a multi-class logistic regression model to classify Iris flower types. It uses Python libraries to load and prepare the Iris dataset, train the model, evaluate its accuracy, calculate decision scores, and visualize the results graphically.

About:-

This project is a multi-class logistic regression model designed to classify types of Iris flowers based on sepal and petal dimensions. Using numpy, pandas, and matplotlib libraries, the code creates this machine learning model using a dataset of Iris flowers.

The project starts by loading and preparing the dataset - the dataset columns are labelled, the classes of the flower are mapped to numerical data, the features are normalised, and the dataset is split into training and test datasets. This is done through the "load_data" and "prepare_data" functions.

There is a custom class called "MultiClassLogisticRegression" which contains the core functionality of the learning model. This includes methods to fit the model to the training data; using a binary identification method, compute the sigmoid function for logistic regression, calculate class scores, and predict outcomes based on the fitted model.

After training the model using the training dataset, the accuracy of the model is tested using both the training and test datasets. The "predict" method is applied to estimate the class of the flowers based on the input features. The accuracy of these predictions is then printed to the console.

The model also calculates the decision function scores for each test data point which illustrate the likelihood of each class. These decision score are also printed to the console.

Finally, the "plot_data" function creates a graphical plot showing the classifications made by the model when applied to the training data, providing a visual representation of the model's learning and decision making processes.

Overall, this project forms a solid example of how to implement a multiclass logistic regression machine learning model in Python with practical data visualization strategies.

Instructions to setup:-

Simply set the path of the downloaded dataset file in the code.
