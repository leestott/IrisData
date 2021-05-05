# IrisData
## Iris Data Example Python Numpy
### https://blogs.msdn.microsoft.com/uk_faculty_connection/2017/07/04/how-to-implement-the-backpropagation-using-python-and-numpy/?WT.mc_id=academic-0000-leestott

### Azure Jupyter Notebook Demo Interactive Experiment https://notebooks.azure.com/n/ln6ojzL3dZY/notebooks/IrisDataDemo.ipynb?WT.mc_id=academic-0000-leestott

The Iris Data Set has over 150 item records. Each item has four numeric predictor variables (often called features): sepal length and width, and petal length and width, followed by the species ("setosa," "versicolor" or "virginica"). 

- The demo program uses 1-of-N label encoding, so setosa = (1,0,0) and versicolor = (0,1,0) and virginica = (0,0,1). The goal is to predict species from sepal and petal length and width. 

- The 150-item dataset has 50 setosa items, followed by 50 versicolor, followed by 50 virginica. Before writing the demo program, There is a 120-item file of training data (using the first 30 of each species) and a 30-item file of test data (the leftover 10 of each species). 

- The demo program creates a simple neural network with four input nodes (one for each feature), five hidden processing nodes (the number of hidden nodes is a free parameter and must be determined by trial and error), and three output nodes (corresponding to encoded species). The demo loaded the training and test data into two matrices. 

- The back-propagation algorithm is iterative and you must supply a maximum number of iterations (50 in the demo) and a learning rate (0.050) that controls how much each weight and bias value changes in each iteration. Small learning rate values lead to slow but steady training. Large learning rates lead to quicker training at the risk of overshooting good weight and bias values. The max-iteration and leaning rate are free parameters. 

- The demo displays the value of the mean squared error, every 10 iterations during training. As you'll see shortly, there are two error types that are commonly used with back-propagation, and the choice of error type affects the back-propagation implementation. After training completed, the demo computed the classification accuracy of the resulting model on the training data (0.9333 = 112 out of 120 correct) and on the test data (0.9667 = 29 out of 30 correct). The classification accuracy on a set of test data is a very rough approximation of the accuracy you'd expect to see on new, previously unseen data. 
