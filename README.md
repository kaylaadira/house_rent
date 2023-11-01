# HOUSE RENT PREDICTION USING MACHINE LEARNING MODELS WITH HYPERPARAMETER TUNING
## Description
The "House Rent Prediction Using Machine Learning Models with Hyperparameter Tuning" project is a cutting-edge application of artificial intelligence and data science in the real estate industry. This project aims to develop a robust and accurate system that can predict rental prices for houses based on a range of relevant features. By utilizing various machine learning models and employing hyperparameter tuning techniques, this project strives to provide valuable insights for property owners, renters, and real estate professionals.

## Data
The dataset used for this project is obtained from Kaggle with the title “House Rent Prediction Dataset” that can be accessed through https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset. In this project, five machine learning models were considered for house rent prediction; Linear Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Gaussian Process (GP), and XGBoost.

## EDA & Preprocessing
Exploratory Data Analysis (EDA), was applied to analyze the characteristics or features in the dataset. The result of this EDA tells that the number of bathrooms in a house/apartment have significant relation with the rent of the house/apartment. One hot encoding and standard scaler was applied in preprocessing step. One hot encoding was applied to convert categorical value into indicator variable while a standard scaler is applied because the range of the value in the dataset is very varied. Standard scaler make it more standardized by using StandardScaler() which can scale each feature/variable to unit variance.

## Training
The data was trained on five machine learning models, which are Linear Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Gaussian Process, and XGBoost. Each model was evaluated on four metrics,  mean score 5-fold cross-validation (CV(5)), mean absolute error (MAE), root mean squared error (RMSE), and R-Squared Score. As seen on Result Table, XGBoost (before tuning) achieved the highest accuracy compared to other models with an R-Squared score of 0.8011082769077092. 

The XGBoost then tuned to optimize the model's performance and search the best parameters for this case. The tuned XGBoost model achieve the accuracy of 0.803864006778024 on R-Squared metric and on 'reg_lambda': 0.1, 'reg_alpha': 0.1, 'n_estimators': 100, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.1 for the best parameters.


## Result
<img alt="image" src="https://github.com/kaylaadira/house_rent/assets/130166504/28f4ac17-e0d8-4b37-b686-c77049a7250a">
Based on these results, the XGBoost model, especially the tuned version, appears to be the best performer among the evaluated models. It shows better performance in terms of cross-validation score, MAE, RMSE, and R-squared, indicating a better fit to the data.

## Conclusion
The XGBoost model outperforms the other models compared in terms of cross-validation score, MAE, RMSE, and R-squared. Tuning the model's hyperparameters has resulted in improved accuracy and generalization, making it a better choice for house rent prediction. However, there is still room for improvement to increase the performance and accuracy.
