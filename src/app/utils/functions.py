from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from typing_extensions import Literal
import re
import joblib

# Clean a dataframe to format it
def clean_df(dataframe, str_column, drop_list=None, cleaning=True, not_encoder=False):
    """
    Cleans a given dataframe by performing various operations such as renaming columns, encoding categorical variables, dropping specified columns, and removing unwanted rows.

    Parameters:
       dataframe: Pandas DataFrame object. The dataframe to be cleaned.
       str_column: string. The name of the column containing categorical variables to be encoded.
       drop_list: list, optional. The list of column names to be dropped from the dataframe.
       cleaning: bool, default=True. If True, performs cleaning operations such as dropping columns and removing unwanted rows.
       not_encoder: bool, default=False. If True, returns the original dataframe without encoding.

    Returns:
       cleaned_df: Pandas DataFrame object. The cleaned dataframe after performing all the cleaning operations.
    """

    # Rename names of columns which have a years at the end of their names
    dataframe.columns = [re.sub(r"_\d{4}$", '', col) for col in dataframe.columns]

    df = dataframe.copy()

    # Encode categorical variables
    encoder=OrdinalEncoder()
    df[[str_column]] = encoder.fit_transform(df[[str_column]])

    # If cleaning==True drop specified columns
    if cleaning==True:
        df.drop(drop_list, axis=1, inplace=True)

        df_features = pd.read_csv("data/df_features.csv", sep=";", index_col=0)
        df.drop(df_features[6:].index, axis=1, inplace=True)

    # if not_encoder==True return the dataframe without encoding
    if not_encoder==True:

        dataframe = dataframe[df.columns]

        return dataframe
    else:
        return df

# Get prediction
def get_prediction(model, dataframe, str_column, drop_list=None, cleaning=True, not_encoder=False):
    """Get prediction from the model.
    
    Args:
        model (object): The trained model.
        dataframe (DataFrame): The input dataframe.
        str_column (str): The name of the column containing categorical data.
    
    Returns:
        prediction (array-like): The prediction.
    """

    # Get dataframe cleaned
    df = clean_df(dataframe, str_column, drop_list, cleaning, not_encoder)

    # Get prediction
    predction = model.predict(df)

    return predction

 # Polynomial Linear Regresion pipeline
def pipelines():
    """
    Creates the pipelines for the different models
        
    returns:
        pipelines: list of pipelines, list of names of models, numbers of clusters
    """

    plr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ("PolynomialFeatures", PolynomialFeatures()),
        ('lr', LinearRegression())
    ])

    # ElasticNet pipeline
    enet_pipe = Pipeline([
        ('ElasticNet', ElasticNet()),
    ])

    # KNN pipeline
    knn_pipe = Pipeline([
        ('KNeighborsRegressor', KNeighborsRegressor()),
    ])

    # SVR pipeline
    svr_pipe = Pipeline([
        ('svr', SVR()),
    ])

    # Random Forest pipeline
    rdf_pipe = Pipeline([
        ('RandomForestRegressor', RandomForestRegressor()),
    ])

    # Gradient Boosting pipeline
    gb_pipe = Pipeline([
        ('GradientBoostingRegressor', GradientBoostingRegressor()),
    ])

    # AdaBoost pipeline
    ab_pipe = Pipeline([
        ('AdaBoostRegressor', AdaBoostRegressor()),
    ])

    # XGBoost pipeline
    xgb_pipe = Pipeline([
        ('XGBRegressor', xgb.XGBRegressor()),
    ])

    return ([plr_pipe, enet_pipe, knn_pipe, svr_pipe, rdf_pipe, gb_pipe, ab_pipe, xgb_pipe],
            ["Polynomial Regression", "ElasticNet", "KNeighborsRegressor", "SuperVectorMachine",
              "Random Forest", "GradientBoosting", "AdaBoost", "XGBoost"])

# Obtain a baseline of Regression models
def get_baseline(df, target_column, pipes, pipe_name, X_train, Y_train, cv:int = 5, sort_metric: Literal["MAE", "MAPE", "MSE", "RMSE", "R2", None] = None):
    """
    Obtain a baseline of Regression models
    
    args:
        pipes: list of pipelines
        pipe_name: list of names of models
        X_train: X_train dataframe
        Y_train: Y_train dataframe
        cv: number of folds for cross validation
        sort_metric: metric to sort the models by. Must be one of these: MAE, MAPE, MSE, RMSE, R2, None
        
    returns:
        grid_serch.best_estimator_: best pipeline of the baseline 
    """

    pipe_list = pipes.copy()

    # List of MAE of all pipe_list
    mae_results = [abs(np.mean(cross_val_score(pipe, X_train,
                        Y_train, cv=cv, scoring="neg_mean_absolute_error"))) for pipe in pipes.copy()]

    # List of MAPE of all pipe_list
    mape_results = [abs(np.mean(cross_val_score(pipe, X_train,
                        Y_train, cv=cv, scoring="neg_mean_absolute_percentage_error"))) for pipe in pipes.copy()]

    # List of MSE of all pipe_list
    mse_results = [abs(np.mean(cross_val_score(pipe, X_train,
                        Y_train, cv=cv, scoring="neg_mean_squared_error"))) for pipe in pipes.copy()]

    # List of RMSE of all pipe_list
    rmse_results = [np.sqrt(abs(np.mean(cross_val_score(pipe, X_train, Y_train,
                                cv=cv, scoring="neg_mean_squared_error")))) for pipe in pipes.copy()]

    # List of R2 of all pipe_list
    r2_results = [np.mean(cross_val_score(pipe, X_train, 
                            Y_train, cv=cv, scoring="r2")) for pipe in pipes.copy()]

    # Obtein a dataframe of all pipeline and theirs metrics sorting by R2
    if sort_metric == "R2":
        baseline = pd.DataFrame({
            "MAE": mae_results,
            "MAPE": mape_results,
            "MSE": mse_results,
            "RMSE": rmse_results,
            "R2": r2_results
        }, index=pipe_name).sort_values(by=sort_metric, ascending=False)

    # Obtein a dataframe of all pipeline and theirs metrics without sorting
    elif sort_metric == None:
        baseline = pd.DataFrame({
            "MAE": mae_results,
            "MAPE": mape_results,
            "MSE": mse_results,
            "RMSE": rmse_results,
            "R2": r2_results
        }, index=pipe_name)

    # # Obtein a dataframe of all pipeline and theirs metrics by other metrics
    else:
        baseline = pd.DataFrame({
            "MAE": mae_results,
            "MAPE": mape_results,
            "MSE": mse_results,
            "RMSE": rmse_results,
            "R2": r2_results
        }, index=pipe_name).sort_values(by=sort_metric, ascending=True)

    # Parameters grid of Polynomial Linear Regression
    if baseline.index[0] == "Polynomial Regression":
        param_grid = {
            "PolynomialFeatures__degree": range(1, 5),
            "lr__fit_intercept": [False, True]
        }

        model = pipe_list[0]

    # Parameters grid of ElasticNet
    if baseline.index[0] == "ElasticNet":
        param_grid = {
            "ElasticNet__alpha": np.logspace(-4, 3, 20),
            "ElasticNet__l1_ratio": range(0, 1, 0.1)
        }

        model = pipe_list[1]

    # Parameters grid of KNN
    if baseline.index[0] == "KNeighborsRegressor":
        param_grid = {
            "KNeighborsRegressor__n_neighbors": range(1, 6)
        }

        model = pipe_list[2]

    # Parameters grid of SVR
    if baseline.index[0] == "SVR":
        param_grid = {
            "svr__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "svr__degree": range(1, 5),
            "svr__C": np.power(10, np.float_(np.arange(-1, 6))),
            "svr__epsilon": [0.01, 0.1, 0.5, 1],
            "svr__gamma": ["scale", "auto"]
        }

        model = pipe_list[3]

    # Parameters grid of Random Fores
    if baseline.index[0] == "Random Forest":
        param_grid = {
            "RandomForestRegressor__n_estimators": range(1, 100, 10),
            "RandomForestRegressor__max_depth": range(3, 10),
            "RandomForestRegressor__min_samples_leaf": range(1, 20),
            "RandomForestRegressor__max_leaf_nodes": range(1, 20)
        }

        model = pipe_list[4]

    # Parameters grid of Gradient Boosting
    if baseline.index[0] == "GradientBoosting":
        param_grid = {
            "GradientBoostingRegressor__n_estimators": range(1, 100, 10),
            "GradientBoostingRegressor__max_depth": range(3, 10),
            "GradientBoostingRegressor__learning_rate": [0.01, 0.1, 0.5]
        }

        model = pipe_list[5]

    # Parameters grid of AdaBoost
    if baseline.index[0] == "AdaBoost":
        param_grid = {
            "AdaBoostRegressor__n_estimators": range(1, 100, 10),
            "AdaBoostRegressor__learning_rate": [0.01, 0.1, 0.5]
        }

        model = pipe_list[6]

    # Parameters grid of XGBoost
    if baseline.index[0] == "XGBoost":
        param_grid = {
            "XGBRegressor__n_estimators": range(1, 100, 10),
            "XGBRegressor__max_depth": range(3, 10),
            "XGBRegressor__alpha": [0, 0.01, 0.1],
            "XGBRegressor__lambda": [0, 0.01, 0.1]
        }

        model = pipe_list[7]

    # Traslation of Metrics to introduce as argument in GridSearchCV function
    scorings = {"MAE": "neg_mean_absolute_error",
                "MAPE": "neg_mean_absolute_percentage_error",
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error",
                "R2": "r2"}

    final_model = model
    
    final_model.fit(df.drop(columns=[target_column]), df[target_column])

    return (r2_results[0], final_model)


# Obtein a baseline test of Regression models
def model_tests(model, X_test, Y_test, sort_metric):
    """
    Calculates the specified sort metric for evaluating a machine learning model.

    Parameters:
        model (object): The trained machine learning model.
        X_test (array-like): The input features for testing the model.
        Y_test (array-like): The true labels for testing the model.
        sort_metric (str): The sort metric to calculate. Can be one of "MAE", "MAPE", "MSE", "RMSE", or "R2".

    Returns:
        The model with the best metric.
    """

    # Get prediction
    prediction = model.predict(X_test)

    # Sort metric depend on the metric
    if sort_metric == "MAE":
        mean_absolute_error(Y_test, prediction)
    
    elif sort_metric == "MAPE":
        mean_absolute_percentage_error(Y_test, prediction)
    
    elif sort_metric == "MSE":
        mean_squared_error(Y_test, prediction)
    
    elif sort_metric == "RMSE":
        np.sqrt(mean_squared_error(Y_test, prediction))
    
    elif sort_metric == "R2":
        r2_score(Y_test, prediction)

    return prediction[0]

# Get best regression model of machine learning for one dataframe
def get_best_model(df, column_taget, X_train, Y_train, X_test, Y_test, cv:int = 5, sort_metric: Literal["MAE", "MAPE", "MSE", "RMSE", "R2", None] = None):
    """
    Retrieves the best model based on the given dataset and target column.

    Parameters:
        df (pandas.DataFrame): The dataset to train and test the models.
        column_target (str): The target column to be predicted by the models.
        X_train (numpy.ndarray): The training dataframe.
        Y_train (numpy.ndarray): The training target values.
        X_test (numpy.ndarray): The testing dataframe.
        Y_test (numpy.ndarray): The testing target values.
        cv (int, optional): The number of folds for cross-validation. Defaults to 5.
        sort_metric (Literal["MAE", "MAPE", "MSE", "RMSE", "R2", None], optional): The metric used to sort the models. Defaults to None.

    Returns:
        Tuple[float, float, Any]: A tuple containing the model's training score, test score, and the final model.
    """

    # Get baseline and the best model from this baseline
    model_train_score, final_model = get_baseline(df, column_taget, pipelines()[0], pipelines()[1], X_train, Y_train, cv, sort_metric)

    # Get metrics of the best model from the test data
    test_metrics = model_tests(final_model, X_test, Y_test, sort_metric)
    
    return  (round(model_train_score, 2), round(test_metrics, 2), final_model)