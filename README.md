# Predictive-Modeling-with-R
This project covers the fundamental statistical learning and real world business problem using the R programming language. Using the dataset `insurance.csv` we perform the project objectives to understand how the response variable **charges** is affected by different predictor variables. We will also find out which model best performs for business goals and purposes.

## Project Objective
- Perform data feature selection, feature elimination, feature importance using techniques such as Recursive Feature Elimination (RFE), Principal Component Analysis (PCA), and Random Forest.
- Develop models using supervised, unsupervised, and semi-supervised learning techniques such as decision trees, regression trees, neural networks, and support vector machines.
- Tune model parameters, estimate prediction eros, and model validation.
- Compare and ensemble multiple models in pipeline and automatically select the best model.

## Data Preparation
First, the dataset `insurance.csv` was loaded into memory. Before continuing, we transformed the variable `charges` by using the `log()` function. Transforming `charges` allowed us to normalize its dataset since it was a highly skewed variable.

Next, we used `model.matrix()` to create another dataset that uses dummy variables in place of the categorical variables. To split the data we used the `sample()` function and `set.seed()` to 1 we generated row indexes for the training and test sets. 2/3 of the row indexes for the training set and 1/3 for the test set. This allowed us to then create training and test datasets from the dataset created previously. And also we created training and test datasets from the dummy variables.

## Build a Multiple Linear Regression Model
In this section we performed **multiple linear regression** `lm` with variable `charges` as the reponse and the predictors are: `age`, `sex`, `bmi`, `children`, `smoker`, and `region`. We can call `summary()` function to understand the model.

 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158248988-167ea3b6-5ba1-4585-bd90-8a60bab3b8b6.png" /p>
 </p>
 
In the `summary()` function's output, we can determine the relationship looking at the **p-value**. If the p-value is greater than 5%, their will be no indication of a relationship. If the p-value is less than 5% than their is a relationship. As the p-value is 2.2e-16%, there is a relationship between the predictors and the response variables. For example, we can see that `sex` has a significant relationship to the response `charges`. We determine this by looking at the **Pr(>|t|)** value. `Sex`, specifically for male, is 0.027847%, which means that `sex` has a significant relationship with response `charges`.

From the multiple linear regression model we predicted that the best model will include the following predictors: `age`,`sex`,`bmi`,`children`,`smoker`, and `region`. We will use `stepAIC()` function to perform the best subset selection and choose the best model based on AIC.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158249755-e57656e2-bc79-4e17-83ef-2211141858d0.png" /p>
 </p>
 
 After performing `stepAIC()` we computed the test error of the best model using `LOOCV()` using `trainControl()` and `train()` from the caret library. These functions from the caret library allow us to streamline the model training process for complex regrssion and classification problems. The `LOOCV()` is a type of cross-validation in which each observation is considered as the validation set and the rest (N-1) observations are considered as the training set. In LOOCV, fitting of the model is done and predicting using one observation validation set. The **MSE** for the **LOOCV** model was **0.1833683**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158250633-72870eed-0e0a-44f8-9824-14f8f2546e86.png" /p>
 </p>
 
 We were also able to calculate the test error of the best model based on AIC using **10-fold cross validation (CV)**. With the 10-fold cross validation we achieved a **MSE** of **0.1792372**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158251156-485be981-bab6-4cd5-a56e-6eb1e45d3e4b.png" /p>
 </p>

Between both models we will use the **LOOCV**. The MSE for this final model was **0.2347613**. In summary, we used the test dataset to compare the predictions to the actual response variable in the test data so we are able to evaluate the model's accuracy.

## Build a Regression Tree Model
Using `tree()` we will build a regression tree model where `charge` will be the response variable and the following variables are the predictors: `age`,`sex`,`bmi`,`children`,`smoker`, and `region`. The `tree()` function is grown by binary recursive partitioning using the repsponse variable in the specified formula and choosing splits from the predictors.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158256617-bc3c9985-5b22-4572-907d-7f766b04e721.png" /p>
 </p>

We can then find the optimal tree by using `cv.tree()` which runs a k-fold cross-validation experiment to find the deviance or number of misclassifications as a function of the cost-complexity parameter 'k'.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158257051-378b07c6-3fe3-4ca8-9122-a70d9f580bbc.png" /p>
 </p>
 
 The best size for the optimal model is **6**. We would like to choose the size that has the least amount of error rate because we want to first test how well our model is able to get trained by some data and then predict the data it hasn't seen. If our test error is high after pruning, that just means we can continue with a subtree to find a simpler model and lower test error. We want to find a balance between variance and bias, with respect to how simple our tree can be. It is important to do cross validation so we can select the optimal value that controls the trade-off between a subtree's complexity and its fit to the training data. After pruning we can plot the best tree model and see what predictors are most important to the response variable `charges`. Using the **regression tree model** the **MSE** is **0.2019634**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158257650-28038df5-503a-4f38-82ce-4e1f954fb2d1.png" /p>
 </p>
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158257922-7452924a-7ec1-4599-9c5b-e4a692d49669.png" /p>
 </p>
 
## Build a Random Forest Model
The next model we built was the **Random Forest** model using the `randomForest()` model. We can see that the MSR and % variance are based on **OOB estimates**, a device to get honest error estimates.

 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158258629-f009ae43-02ad-4f5c-a6f2-198de2f96481.png" /p>
 </p>

We were also able to extract the variable importance measure using the `importance()` function.

 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158258900-0b874e7b-1a29-4d4c-ae11-3bd34c55d38d.png" /p>
 </p>

Visually seeing the variable importance using the `varImpPlot()` function we can see that the top 2 important predictors in this model are **smoker** and **age**. The third important predictor is between **children** and **bmi** depending on the graphs. This plot ranks the usefulness of the variables. This means that **smoker** and **age** are the variables that will give the prediction and contribute most to the model. **children** may be an important variable because if an insurance policy holder has more children, the `charges` would be higher. On the other hand, **bmi** may be an important variable because a policy holder's health background may affect the insurance `charges`.

 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158259551-5797ce40-b6a1-4fb9-9e99-5cfe263c00de.png" /p>
 </p>

## Build a Support Vector Machine
**Support Vector Machines (SVM)** are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Performing SVM using the `svm()` we then were able to perform a grid search to find the best model's parameters of potential `cost`, `gamma` and `kernel`.

 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158260399-d66666b3-7da4-4698-b5d8-47fd0f9602f0.png" /p>
 </p>

Applying these parameters to the best model then using the `predict()` function on the test dataset we were able to achieve a **MSE** for the SVM model of **0.2259622**.

## Perform K-Means Cluster Analysis
**K-Means Clustering** is a common unsupervised machine learning algorithm for partitioning a given dataset into a set of k groups, where k represents the number of groups pre-specified by the analyst. For the `insurance` dataset we work with the numerical variables: `age`, `bmi`, `children`, and `charges`. We determined the optimal number of clusters using the libraries `cluster` and `facoextra`. Using the `fviz_nbclust()` function the optimal number of clusters is **3**. This means that when k=2, this is considered a good cluster for which the within-cluster variation is small as possible.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158261263-13c369a7-63cb-461b-828b-8de51c43f9ae.png" /p>
 </p>
 
Now performing `kmeans()` with the optimal number of clusters we can visualize our results with `fviz_cluster()`. This will be helpful to assess the choice of number of clusters as well as compare two different cluster analyses. `fviz_cluster()` takes the k-means results and the original data as arugments. In the resulting plot below the observations are represented by points using principal components to find a low-dimensional representation of the observations that explains a good fraction of the variance. For the insurance data, we clustered it into 3 sub-groups that visualize all 4 columns of data observations. The sum of squares summary looks at the deviations of each point around its cluster mean. For our percentage from between sum of squares divided by total sum it is 74.7%. This is a percent variance that explains the cluster means. It is pretty high, which means that the k we chose is meant to be the most optimal, is doing a good job.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158262092-d3d394c6-04c3-489e-8158-88387e3c480b.png" /p>
 </p>

## Build a Neural Network Model
For this model we standardized the inputs using the `scale()` function and converted the inputs to a dataframe using the `as.data.frame()` function. We then split the data into train and test datasets. Like other models we were then able to plot, predict and calculate the **MSE** which was **0.6754152**.

## Conclusion
In conclusion, we have the following model types we created with their respective MSE. For this data, `insurance` we can suggest that the best model to use is the **Random Forest Model** with a MSE of 0.1780. Our goal is to find the model with the smallest MSE. The smallest MSE will represent that the predicted responses are very close to the true responses. We calculate based on MSE because it takes into consideration of the bias-variance trade off. The random forest model is the best because while achieving a MSE closer to zero, it simultaneously has low variance and low bias which are both non-negative.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/158262662-55a8b9ae-eabe-4b89-8bab-568f11472cbf.png" /p>
 </p>

We can use the **Random Forest Model** to explain to clients potential costs could be different for customers. 
The benefits of using the random forest model are: 
  1. Random Forest can be used to solve both classification and regression problems. 
  2. Random Forest works well with both categorical and continuous variables. 
  3. Random Forest creates many trees on the subset of the data & combines the output of all the trees, which means it reduces the overfitting problem which reduces the variance and improves accuracy. 
  4. The model output using the `varImpPlot()` function is a great visual when talking to potential clients and how each variable may affect their charges. 

The disadvantages of using the random forest model are: 
  1. Random Forest creates a lot of trees which means it will use a lot of computational power and resources. The complexity of the model is not great. 
  2. Random Forest require more time to train which means a longer running period.
