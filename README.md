# Predictive-Modeling-with-R
This project covers the fundamental statistical learning and real world business problem using the R programming language.

## Project Objective
- Perform data feature selection, feature elimination, feature importance using techniques such as Recursive Feature Elimination (RFE), Principal Component Analysis (PCA), and Random Forest.
- Develop models using supervised, unsupervised, and semi-supervised learning techniques such as decision trees, regression trees, neural networks, and support vector machines.
- Tune model parameters, estimate prediction eros, and model validation.
- Compare and ensemble multiple models in pipeline and automatically select the best model.

### Data Preparation
First, the dataset `insurance.csv` was loaded into memory. Before continuing, we transformed the variable `charges` by using the `log()` function. Transforming `charges` allowed us to normalize its dataset since it was a highly skewed variable.

Next, we used `model.matrix()` to create another dataset that uses dummy variables in place of the categorical variables. To split the data we used the `sample()` function and `set.seed()` to 1 we generated row indexes for the training and test sets. 2/3 of the row indexes for the training set and 1/3 for the test set. This allowed us to then create training and test datasets from the dataset created previously. And also we created training and test datasets from the dummy variables.

### Build a multiple linear regression model
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

### Build a regression model tree
