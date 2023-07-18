# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from IPython.display import display
from typing import Literal
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import silhouette_score
import joblib

# Function to obtain an univariant analysis
def get_univariate_analysis(self, df_no_outliers=None, graphics=True):

    if df_no_outliers is None:
        df_no_outliers = self

    normal_var = 0
    no_normal_var = 0
    univar_analysis = pd.DataFrame(
        {}, columns=["Mean", "Median", "Mode", "Variance", "Standart_desv", "Percentil_25", "Percentil_75", "K_test", "p_value", "Distribution"])

    for col in self.columns:
        print(f"\033[1mAnálisis univariante de {col}:\033[0m")
        # Realiza un análisis si la variable es categórica
        if (self[col].dtype == object) or ((col == "Codigo_municipio") or (col == "Codigo_provincia")):
            print(f"Variable categórica:")
            print(f"-Valores únicos:\n{self[col].value_counts()}")
            print(f"-Número de valores únicos: {self[col].nunique()}")
            print("\n\n\n")

        # Realiza un análisis si la variable es numérica
        else:
            if graphics:
                # Crea un histograma y un boxplot de la variable
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                sns.histplot(df_no_outliers[col], kde=True, ax=axes[0])
                axes[0].set_title("Histograma")
                sns.boxplot(df_no_outliers[col], ax=axes[1])
                axes[1].set_title("Boxplot")
                fig.suptitle(f"Análisis de {col}")
                plt.show()

            # Comprueba estadisticamente con el test Kolmogorov-Smirnov si la variable sigue una distribución normal
            stat, p = ss.kstest(self[col], 'norm')
            alpha = 0.05

            # Añade los datos al DataFrame dependiendo de si se acepta H0 o no
            if p < alpha:
                no_normal_var += 1
                univar_analysis = pd.concat([univar_analysis, pd.DataFrame({"Features": col, "Mean": self[col].mean(), "Median": self[col].median(
                ), "Mode": self[col].mode().iloc[0], "Variance": self[col].var(), "Standart_desv": self[col].std(), "Percentil_25": self[col].quantile(0.25), "Percentil_75": self[col].quantile(0.75), "K_test": stat, "p_value": p, "Distribution": "Not standart"}, index=[0])])  # type: ignore
                print(
                    f"La columna {col} no presenta una distribución normal\n\n\n")

            else:
                normal_var += 1
                univar_analysis = pd.concat([univar_analysis, pd.DataFrame({"Features": col, "Mean": self[col].mean(), "Median": self[col].median(
                ), "Mode": self[col].mode().iloc[0], "Varianza": self[col].var(), "Desviacion_estandar": self[col].std(), "Percentil_25": self[col].quantile(0.25), "Percentil_75": self[col].quantile(0.75), "K_test": stat, "p_value": p, "Distribution": "Standart"}, index=[0])])  # type: ignore
                print(
                    f"La columna {col} presenta una distribución normal\n\n\n")

    # Establece la columna Municipio como índice
    univar_analysis.set_index("Features", inplace=True)

    # Imprime el número de variables que siguen una distribución normal y el que no
    print(
        f"\033[1mNúmero de variables que siguen una distribución normal:\033[0m {normal_var}")
    print(
        f"\033[1mNúmero de variables que no siguen una distribución normal:\033[0m {no_normal_var}")
    return univar_analysis


# Function to obtain bivariate analysis
def get_bivariate_analysis(self, annot=False):
    """
    Obtiene el análisis bivariante de un dataframe

    args:
        list_series: lista de series o dataframes
        annot: boolean, si se quiere mostrar la correlación entre las variables
    """
    if annot:
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.corr(numeric_only=True), annot=True)
    else:
        sns.heatmap(self.corr(numeric_only=True), annot=False)
    sns.pairplot(self, diag_kind='kde')

# Function to obtain bivariate analysis
def get_bivariate_analysis(self, annot=False):
    """
    Obtiene el análisis bivariante de un dataframe

    args:
        list_series: lista de series o dataframes
        annot: boolean, si se quiere mostrar la correlación entre las variables
    """
    if annot:
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.corr(numeric_only=True), annot=True)
    else:
        sns.heatmap(self.corr(numeric_only=True), annot=False)
    sns.pairplot(self, diag_kind='kde')

 # Polynomial Linear Regresion pipeline
def pipelines(kmeans=False, n_clusters=None):
    if kmeans==False:
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
    
        n_clusters=None
    else:
        
        # Polynomial Linear Regresion pipeline
        plr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ("PolynomialFeatures", PolynomialFeatures()),
            ('lr', LinearRegression())
        ])

        # ElasticNet pipeline
        enet_pipe = Pipeline([
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ('ElasticNet', ElasticNet()),
        ])

        # KNN pipeline
        knn_pipe = Pipeline([
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ('KNeighborsRegressor', KNeighborsRegressor()),
        ])

        # SVR pipeline
        svr_pipe = Pipeline([
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ('svr', SVR()),
        ])

        # Random Forest pipeline
        rdf_pipe = Pipeline([
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ('RandomForestRegressor', RandomForestRegressor()),
        ])

        # Gradient Boosting pipeline
        gb_pipe = Pipeline([
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ('GradientBoostingRegressor', GradientBoostingRegressor()),
        ])

        # AdaBoost pipeline
        ab_pipe = Pipeline([
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ('AdaBoostRegressor', AdaBoostRegressor()),
        ])

        # XGBoost pipeline
        xgb_pipe = Pipeline([
            ("KMeans", KMeans(n_clusters=n_clusters)),
            ('XGBRegressor', xgb.XGBRegressor()),
        ])


    return ([plr_pipe, enet_pipe, knn_pipe, svr_pipe, rdf_pipe, gb_pipe, ab_pipe, xgb_pipe],
            ["Polynomial Regression", "ElasticNet", "KNeighborsRegressor", "SuperVectorMachine",
              "Random Forest", "GradientBoosting", "AdaBoost", "XGBoost"], n_clusters)


# Obtein a baseline of Regression models
def get_baseline(pipes, pipe_name, X_train, Y_train, cv:int = 5, sort_metric: Literal["MAE", "MAPE", "MSE", "RMSE", "R2", None] = None):

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

    print("Baseline")
    display(baseline)
    print(f"The best methods based on {sort_metric} is {baseline.index[0]}")

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

    # GridSearchCV of best model
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv,
                               scoring=scorings[sort_metric])

    grid_search.fit(X_train, Y_train)

    print("\n\n")

    print(
        f"Using the {baseline.index[0]} model, the best params to aply are these {grid_search.best_params_}. ")
    print(f"It has been obteing a {sort_metric} of {abs(grid_search.best_score_)}")


    print("\n\n\n")

    return grid_search.best_estimator_


def get_baseline_Kmeans(pipes, pipe_name, n_clusters, X_train, Y_train, cv:int = 5, sort_metric: Literal["MAE", "MAPE", "MSE", "RMSE", "R2", None] = None):

    pipe_list = pipes.copy()

    # List of MAE of all pipelines
    mae_results = [abs(np.mean(cross_val_score(pipe, X_train,
                       Y_train, cv=cv, scoring="neg_mean_absolute_error"))) for pipe in pipes.copy()]

    # List of MAPE of all pipelines
    mape_results = [abs(np.mean(cross_val_score(pipe, X_train,
                        Y_train, cv=cv, scoring="neg_mean_absolute_percentage_error"))) for pipe in pipes.copy()]

    # List of MSE of all pipelines
    mse_results = [abs(np.mean(cross_val_score(pipe, X_train,
                       Y_train, cv=cv, scoring="neg_mean_squared_error"))) for pipe in pipes.copy()]

    # List of RMSE of all pipelines
    rmse_results = [np.sqrt(abs(np.mean(cross_val_score(pipe, X_train,
                        Y_train, cv=cv, scoring="neg_mean_squared_error")))) for pipe in pipes.copy()]

    # List of R2 of all pipelines
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

    print("Baseline")
    display(baseline)
    print(f"The best methods based on {sort_metric} is {baseline.index[0]}")

    # Parameters grid of Polynomial Linear Regression
    if baseline.index[0] == "Polynomial Regression":
        param_grid = {
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
            "PolynomialFeatures__degree": range(1, 5),
            "lr__fit_intercept": [False, True]
        }

        model = pipe_list[0]

    # Parameters grid of ElasticNet
    if baseline.index[0] == "ElasticNet":
        param_grid = {
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
            "ElasticNet__alpha": np.logspace(-4, 3, 20),
            "ElasticNet__l1_ratio": range(0, 1, 0.1)
        }

        model = pipe_list[1]

    # Parameters grid of KNN
    if baseline.index[0] == "KNeighborsRegressor":
        param_grid = {
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
            "KNeighborsRegressor__n_neighbors": range(1, 6)
        }

        model = pipe_list[2]

    # Parameters grid of SVR
    if baseline.index[0] == "SVR":
        param_grid = {
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
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
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
            "RandomForestRegressor__n_estimators": range(1, 100, 20),
            "RandomForestRegressor__max_depth": range(5, 10),
            "RandomForestRegressor__min_samples_leaf": range(1, 20),
            "RandomForestRegressor__max_leaf_nodes": range(1, 20)
        }

        model = pipe_list[4]

    # Parameters grid of Gradient Boosting
    if baseline.index[0] == "GradientBoosting":
        param_grid = {
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
            "GradientBoostingRegressor__n_estimators": range(1, 100, 10),
            "GradientBoostingRegressor__max_depth": range(3, 10),
            "GradientBoostingRegressor__learning_rate": [0.01, 0.1, 0.5]
        }

        model = pipe_list[5]

    # Parameters grid of AdaBoost
    if baseline.index[0] == "AdaBoost":
        param_grid = {
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
            "AdaBoostRegressor__n_estimators": range(1, 100, 10),
            "AdaBoostRegressor__learning_rate": [0.01, 0.1, 0.5]
        }

        model = pipe_list[6]

    # Parameters grid of XGBoost
    if baseline.index[0] == "XGBoost":
        param_grid = {
            "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
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

    # GridSearchCV of best model
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv,
                               scoring=scorings[sort_metric])

    grid_search.fit(X_train, Y_train)

    print("\n\n")
    
    print(
        f"Using the {baseline.index[0]} model, the best params to aply are these {grid_search.best_params_}. ")
    print(f"It has been obteing a {sort_metric} of {abs(grid_search.best_score_)}")

    print("\n\n\n")
    
    return grid_search.best_estimator_

# Obtein a baseline test of Regression models
def model_tests(pipes, pipe_name, X_train, Y_train, X_test, Y_test, sort_metric, cv, n_clusters=None):

    pipe_list = pipes.copy()

    # List of fitting
    pipe_fitted = [pipe.fit(X_train, Y_train) for pipe in pipes.copy()]

    # List of predictions
    pipe_predicts = [pipe.predict(X_test) for pipe in pipe_fitted]

    # List of MAE of all pipelines
    mae_results = [mean_absolute_error(Y_test, pipe_test) for pipe_test in pipe_predicts]

    # List of MAPE of all pipelines
    mape_results = [ mean_absolute_percentage_error(Y_test, pipe_test) for pipe_test in pipe_predicts]

    # List of MSE of all pipelines
    mse_results = [mean_squared_error(Y_test, pipe_test) for pipe_test in pipe_predicts]

    # List of RMSE of all pipelines
    rmse_results = [np.sqrt(mean_squared_error(Y_test, pipe_test)) for pipe_test in pipe_predicts]

    # List of R2 of all pipelines
    r2_results = [r2_score(Y_test, pipe_test) for pipe_test in pipe_predicts]

    
    # Obtein a dataframe of all pipeline and theirs metrics sorting by R2
    if sort_metric == "R2":
        baseline_test = pd.DataFrame({
            "MAE": mae_results,
            "MAPE": mape_results,
            "MSE": mse_results,
            "RMSE": rmse_results,
            "R2": r2_results
        }, index=pipe_name).sort_values(by=sort_metric, ascending=False)

    # Obtein a dataframe of all pipeline and theirs metrics without sorting
    elif sort_metric == None:
        baseline_test = pd.DataFrame({
            "MAE": mae_results,
            "MAPE": mape_results,
            "MSE": mse_results,
            "RMSE": rmse_results,
            "R2": r2_results
        }, index=pipe_name)
    
    # # Obtein a dataframe of all pipeline and theirs metrics by other metrics
    else:
        baseline_test = pd.DataFrame({
            "MAE": mae_results,
            "MAPE": mape_results,
            "MSE": mse_results,
            "RMSE": rmse_results,
            "R2": r2_results
        }, index=pipe_name).sort_values(by=sort_metric, ascending=True)

    
    if n_clusters:
    
        # Parameters grid of Polynomial Linear Regression
        if baseline_test.index[0] == "Polynomial Regression":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "PolynomialFeatures__degree": range(1, 5),
                "lr__fit_intercept": [False, True]
            }

            model = pipe_list[0]

        # Parameters grid of ElasticNet
        if baseline_test.index[0] == "ElasticNet":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "ElasticNet__alpha": np.logspace(-4, 3, 20),
                "ElasticNet__l1_ratio": range(0, 1, 0.1)
            }

            model = pipe_list[1]

        # Parameters grid of KNN
        if baseline_test.index[0] == "KNeighborsRegressor":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "KNeighborsRegressor__n_neighbors": range(1, 6)
            }

            model = pipe_list[2]

        # Parameters grid of SVR
        if baseline_test.index[0] == "SVR":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "svr__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                "svr__degree": range(1, 5),
                "svr__C": np.power(10, np.float_(np.arange(-1, 6))),
                "svr__epsilon": [0.01, 0.1, 0.5, 1],
                "svr__gamma": ["scale", "auto"]
            }

            model = pipe_list[3]

        # Parameters grid of Random Fores
        if baseline_test.index[0] == "Random Forest":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "RandomForestRegressor__n_estimators": range(1, 100, 20),
                "RandomForestRegressor__max_depth": range(5, 10),
                "RandomForestRegressor__min_samples_leaf": range(1, 20),
                # "RandomForestRegressor__max_leaf_nodes": range(1, 20)
            }

            model = pipe_list[4]

        # Parameters grid of Gradient Boosting
        if baseline_test.index[0] == "GradientBoosting":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "GradientBoostingRegressor__n_estimators": range(1, 100, 10),
                "GradientBoostingRegressor__max_depth": range(3, 10),
                "GradientBoostingRegressor__learning_rate": [0.01, 0.1, 0.5]
            }

            model = pipe_list[5]

        # Parameters grid of AdaBoost
        if baseline_test.index[0] == "AdaBoost":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "AdaBoostRegressor__n_estimators": range(1, 100, 10),
                "AdaBoostRegressor__learning_rate": [0.01, 0.1, 0.5]
            }

            model = pipe_list[6]

        # Parameters grid of XGBoost
        if baseline_test.index[0] == "XGBoost":
            param_grid = {
                "KMeans__n_clusters": range(n_clusters-2, n_clusters+2),
                "XGBRegressor__n_estimators": range(1, 100, 10),
                "XGBRegressor__max_depth": range(3, 10),
                "XGBRegressor__alpha": [0, 0.01, 0.1],
                "XGBRegressor__lambda": [0, 0.01, 0.1]
            }

            model = pipe_list[7]
    else:
        # Parameters grid of Polynomial Linear Regression
        if baseline_test.index[0] == "Polynomial Regression":
            param_grid = {
                "PolynomialFeatures__degree": range(1, 5),
                "lr__fit_intercept": [False, True]
            }

            model = pipe_list[0]

        # Parameters grid of ElasticNet
        if baseline_test.index[0] == "ElasticNet":
            param_grid = {
                "ElasticNet__alpha": np.logspace(-4, 3, 20),
                "ElasticNet__l1_ratio": range(0, 1, 0.1)
            }

            model = pipe_list[1]

        # Parameters grid of KNN
        if baseline_test.index[0] == "KNeighborsRegressor":
            param_grid = {
                "KNeighborsRegressor__n_neighbors": range(1, 6)
            }

            model = pipe_list[2]

        # Parameters grid of SVR
        if baseline_test.index[0] == "SVR":
            param_grid = {
                "svr__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                "svr__degree": range(1, 5),
                "svr__C": np.power(10, np.float_(np.arange(-1, 6))),
                "svr__epsilon": [0.01, 0.1, 0.5, 1],
                "svr__gamma": ["scale", "auto"]
            }

            model = pipe_list[3]

        # Parameters grid of Random Fores
        if baseline_test.index[0] == "Random Forest":
            param_grid = {
                "RandomForestRegressor__n_estimators": range(1, 100, 10),
                "RandomForestRegressor__max_depth": range(3, 10),
                "RandomForestRegressor__min_samples_leaf": range(1, 20),
                "RandomForestRegressor__max_leaf_nodes": range(1, 20)
            }

            model = pipe_list[4]

        # Parameters grid of Gradient Boosting
        if baseline_test.index[0] == "GradientBoosting":
            param_grid = {
                "GradientBoostingRegressor__n_estimators": range(1, 100, 10),
                "GradientBoostingRegressor__max_depth": range(3, 10),
                "GradientBoostingRegressor__learning_rate": [0.01, 0.1, 0.5]
            }

            model = pipe_list[5]

        # Parameters grid of AdaBoost
        if baseline_test.index[0] == "AdaBoost":
            param_grid = {
                "AdaBoostRegressor__n_estimators": range(1, 100, 10),
                "AdaBoostRegressor__learning_rate": [0.01, 0.1, 0.5]
            }

            model = pipe_list[6]

        # Parameters grid of XGBoost
        if baseline_test.index[0] == "XGBoost":
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

    # GridSearchCV of best model
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv,
                               scoring=scorings[sort_metric])

    grid_search.fit(X_train, Y_train)

    print("Baseline test")
    display(baseline_test)
    print(f"The best methods based on {sort_metric} and after having done a test is {baseline_test.index[0]}")
        
    print("\n\n")
    
    print(
        f"Using the {baseline_test.index[0]} model, the best params to aply are these {grid_search.best_params_}. ")
    print(f"It has been obteing a {sort_metric} of {abs(grid_search.best_score_)}")

    return grid_search.best_estimator_

def get_best_model(X_train, Y_train, X_test, Y_test, cv:int = 5, sort_metric: Literal["MAE", "MAPE", "MSE", "RMSE", "R2", None] = None, kmeans=False):
    if kmeans == True:
          # if graphics_inputs==True:
        kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_train)
                    for k in range(1, 10)]
        inertias = [model.inertia_ for model in kmeans_per_k]
        plt.figure(figsize=(8, 3.5))
        plt.plot(range(1, 10), inertias, "bo-")
        plt.xlabel("$k$", fontsize=14)
        plt.ylabel("Inertia", fontsize=14)
        plt.show()

        silhouette_scores = [silhouette_score(X_train, model.labels_)
                            for model in kmeans_per_k[1:]]
        plt.figure(figsize=(8, 3))
        plt.plot(range(2, 10), silhouette_scores, "bo-")
        plt.xlabel("$k$", fontsize=14)
        plt.ylabel("Silhouette score", fontsize=14)
        plt.show()

        n_clusters = 3 #int(input("Input number of clusters: "))

        return (get_baseline_Kmeans(pipelines(kmeans=True, n_clusters=n_clusters)[0], pipelines(kmeans=True)[1], n_clusters, X_train, Y_train, cv, sort_metric),
                model_tests(pipelines(kmeans=True, n_clusters=n_clusters)[0], pipelines(kmeans=True)[1], X_train, Y_train, X_test, Y_test, sort_metric, cv, n_clusters))
    else:
        return (get_baseline(pipelines()[0], pipelines()[1], X_train, Y_train, cv, sort_metric),
                model_tests(pipelines()[0], pipelines()[1], X_train, Y_train, X_test, Y_test, sort_metric, cv))
    
def get_final_model(dataframe, str_column, target_column, ruta):
    
    df = dataframe.copy()

    encoder=OrdinalEncoder()
    df[[str_column]] = encoder.fit_transform(df[[str_column]])

    final_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=6, n_estimators=71)

    final_model = final_model.fit(df.drop(columns=[target_column]), df[target_column])
    joblib.dump(final_model, ruta)

    print(f"Se ha exportado el modelo final a la ruta: {ruta}")

    return final_model

def get_prediction(model, dataframe, str_column):

    df = dataframe.copy()

    encoder=OrdinalEncoder()
    df[[str_column]] = encoder.fit_transform(df[[str_column]])

    predction = model.predict(df)

    return predction

