import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# function that reads data from csv
def read_data():
    file_path = "../styrian_health_data.xlsx"
    sheet_name = "Sheet 1"
    data_df = pd.read_excel(file_path, sheet_name=sheet_name)

    return data_df


def fetch_weather_data(odata_df):
    data_df = odata_df.copy()
    weather_df = pd.read_csv("../Supplementary Datasets/Weather_data_Bruck_An_Der_Mur_2006_to_2007.csv", sep=",")

    # rename the column datetime to zeit:
    weather_df = weather_df.rename(columns={"datetime": "zeit"})

    # extract rows from april 2006 to december 2006 from weather_df:
    weather_df = weather_df[weather_df["zeit"] >= "2006-04-27"]
    weather_df = weather_df[weather_df["zeit"] <= "2006-11-06"]

    data_df["zeit"] = pd.to_datetime(data_df["zeit"], format="%Y-%m-%d %H:%M:%S").dt.date

    # select only the temperature, zeit, name, tempmax, tempmin, humidity, pressure:
    weather_df = weather_df[["zeit", "temp", "tempmax", "tempmin", "humidity"]]

    weather_df["zeit"] = weather_df.zeit.astype(str)
    data_df["zeit"] = data_df.zeit.astype(str)

    # add the columns temp, tempmax, tempmin, humidity to the data_df on the same rows as the zeit:
    merged_df = pd.merge(data_df, weather_df, on="zeit", how="left")

    return merged_df


# function to format variable types, remove nans, shuffle data
def format_variables(data, to_filter, drop_values):
    data_df = data.copy()
    data_df.postleitzahl = data_df.postleitzahl.astype('str')
    data_df.geburtsjahr = data_df.geburtsjahr.astype('Int64')
    data_df.befinden = data_df.befinden.astype('Int64')
    data_df.messwert_bp_sys = pd.to_numeric(data_df.messwert_bp_sys)
    data_df.messwert_bp_dia = pd.to_numeric(data_df.messwert_bp_dia)
    data_df.schaetzwert_bp_sys = pd.to_numeric(data_df.schaetzwert_bp_sys)
    data_df.schaetzwert_by_dia = pd.to_numeric(data_df.schaetzwert_by_dia)
    data_df.befinden = data_df.befinden.astype('str')

    # adding variable for age
    age =  data_df["zeit"].dt.year - pd.to_datetime(data_df['geburtsjahr'], format='%Y').dt.year
    data_df["age"] = age

    data_df["month"] = data_df["zeit"].dt.month
    data_df["hour"] = data_df["zeit"].dt.hour
    data_df["day"] = data_df["zeit"].dt.day

    # seggregating terminal when instrument was changed
    data_df.loc[(data_df["month"].astype('int') <= 6) & (data_df["terminal"] == 3), "terminal"] = "3a"
    data_df.loc[(data_df["month"].astype('int') > 6) & (data_df["terminal"] == 3), "terminal"] = "3b"

    # adding temp info
    data_df = fetch_weather_data(data_df)
    data_df['temp'] = data_df['temp'].astype(float)
    data_df['humidity'] = data_df['humidity'].astype(float)
    data_df['temp_min'] = data_df['tempmin'].astype(float)
    data_df['temp_max'] = data_df['tempmax'].astype(float)
    del data_df["tempmin"]
    del data_df["tempmax"]

    # adding variable for is_local
    mask = data_df.gemeinde.isna() & data_df.bezirk.isna() & data_df.bundesland.isna()

    #replacing nans for variables
    data_df.loc[data_df.geschlecht.isna() == True, 'raucher'] = "unknown"
    data_df.loc[data_df.geschlecht.isna() == True, 'blutzucker_bekannt'] = "unknown"
    data_df.loc[data_df.geschlecht.isna() == True, 'cholesterin_bekannt'] = "unknown"
    data_df.loc[data_df.geschlecht.isna() == True, 'in_behandlung'] = "unknown"
    data_df.loc[data_df.geschlecht.isna() == True, 'befinden'] = "unknown"

    data_df.loc[mask, 'gemeinde'] = "not_applicable"
    data_df.loc[mask, 'bezirk'] = "not_applicable"
    data_df.loc[mask, 'bundesland'] = "not_applicable"
    data_df.loc[mask, 'postleitzahl'] = "not_applicable"
    data_df.loc[data_df.postleitzahl == "nan", 'postleitzahl'] = "unknown"
    data_df.loc[data_df.geschlecht.isna() == True, 'geschlecht'] = "unknown"
    data_df.terminal = data_df.terminal.astype('str')
    
    if drop_values:
    # removing useless variables
        data_df.drop(data_df[data_df.age > 100].index, inplace=True)
        data_df.drop(data_df[data_df.age < 15].index, inplace=True)

        # dropping nan values
        data_df = data_df.dropna()

        data_df["age"] = data_df["age"].astype("int")

    data_df['befinden'] = data_df['befinden'].astype(object)
    data_df['messwert_bp_sys'] = data_df['messwert_bp_sys'].astype(float)
    data_df['messwert_bp_dia'] = data_df['messwert_bp_dia'].astype(float)
    data_df['geschlecht'] = data_df['geschlecht'].astype(object)

    # dropping values from filter

    if drop_values:
        if len(to_filter) > 0:
            data_df = data_df.drop(to_filter, axis=1)

    # shuffling data with fixed seed
    n_rows = data_df.shape[0]
    
    # Create a seed for the random number generator based on the number of rows
    seed = hash(n_rows) % 2**32
    
    # Shuffle the DataFrame using the seed
    np.random.seed(seed)
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    # data_df = data_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # separating var types
    cat_feat_list = []
    num_feat_list = []

    for var in data_df.columns:
        if data_df[var].dtype == object:
            cat_feat_list.append(var)
        else:
            num_feat_list.append(var)

    return data_df, cat_feat_list, num_feat_list


# function that converts cat columns in df to one-hot encoding
def encode_data(df, cat_feat_list, num_feat_list):
    one_hot_data = pd.get_dummies(df[cat_feat_list], drop_first=True, dtype=int)

    for var in num_feat_list:
        one_hot_data[var] = df[var] 
    
    return one_hot_data


# function to separate target from dataframe
def separate_target(data, target):
    df = data.copy()
    Y = df[target]
    del df[target]
    X = df

    return X, Y

# computes adjusted R2
def adjusted_r2(r_2, n, k):
    return 1 - (1-r_2)*(n-1)/(n-k-1)


# function to compute metrics given target and predictions
def compute_metrics(pred, target, num_feats):
    r_2 = r2_score(target, pred)
    mse = mean_squared_error(target, pred)
    adj_r2 = adjusted_r2(r_2, len(pred), num_feats)
    return {
        "r_2": r_2,
        "adjusted_r_2": adj_r2,
        "mse": mse
    }

# method that fits and predicts regression tree based on model type
def fit_and_eval_regression_tree(X_train, Y_train, X_test, params, model_type):
    model = None
    if model_type == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(**params)
    elif model_type == "DecisionTreeRegressorRandomForest":
        model = RandomForestRegressor(**params)

    model.fit(X_train, Y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    return train_predictions, test_predictions, model


# method that fits, predicts, generates eval metrics for regression tree based on model type
def fit_model(X_train, Y_train, X_test, Y_test, model_type, params=None):
    num_feats = len(X_train.columns)
    train_predictions = None
    train_predictions = None
    train_results = None
    test_results = None
    model = None

    if model_type in ["DecisionTreeRegressor", "DecisionTreeRegressorRandomForest"]:
        train_predictions, test_predictions, model = fit_and_eval_regression_tree(X_train, Y_train, X_test, params, model_type)

    if model_type == "LinearRegression":
        train_predictions, test_predictions, model = fit_linear_model(X_train, Y_train, X_test)

    if train_predictions is not None and test_predictions is not None:
        train_results = compute_metrics(train_predictions, Y_train, num_feats)
        test_results = compute_metrics(test_predictions, Y_test, num_feats)


    return train_results, test_results, model


def generate_lin_formula(target, columns):
    pred_list = []
    for col in columns:
        if col != target:
            pred_list.append(col)
    
    return target + " ~ " + " + ".join(pred_list)


def fit_linear_model(X_train, Y_train, X_test):
    train_df = X_train.copy()
    train_df[Y_train.name] = Y_train

    formula = generate_lin_formula(Y_train.name, train_df.columns)
    model = smf.ols(formula, data=train_df).fit()
    train_pred = model.predict(X_train) 
    test_pred = model.predict(X_test)
    return train_pred, test_pred, model


def generate_qq_plot(Y):
    stats.probplot(Y, plot=plt)
    plt.show()


def generate_residual_plot(Y_train, model):
    sns.residplot(x=Y_train, y=model.resid, lowess=True, line_kws=dict(color="g"))


# method that performs best subset feat selection based on some creiterion like mse, adjusted_r_2 or r_2
def best_subset_selection(features, criterion, X_train, Y_train, X_test, Y_test, model_type, params, verbose):
    if criterion == "mse":
        best_val = np.inf
    elif criterion in ["adjusted_r_2", "r_2"]:
        best_val = -np.inf

    best_train_results = None
    best_model = None
    best_test_results = None
    best_features = None
    n_features = len(features)


    for i in range(1, n_features):
        if verbose > 1:
            print("\nNum features: ", i, "=======================================================")

        for j in range(n_features):
            current_features = features[j:j+i]
            if len(current_features) < i:
                break

            X_train_curr = X_train[current_features]
            X_test_curr = X_test[current_features]
            
            train_results, test_results, model = fit_model(X_train_curr, Y_train, X_test_curr, Y_test, model_type, params)

            if verbose > 1:
                print("\nFeatures: ", current_features)
                print("Train Results: ", train_results)
                print("Test Results: ", test_results)

            condition = False
            if criterion == "mse":
                condition = test_results[criterion] < best_val
            elif criterion in ["adjusted_r_2", "r_2"]:
                condition = test_results[criterion] > best_val   

            if condition:
                best_val = test_results[criterion]
                best_model = model
                best_features = current_features
                best_train_results = train_results
                best_test_results = test_results
    
    if verbose > 0:
        print("\nBest Model: ")
        print("Features: ", best_features)
        print("Train Results: ", best_train_results)
        print("Test Results: ", best_test_results)
    
    return best_model, best_train_results, best_test_results


# method that formats results of different models in a dataframe
def tabularize_model_metrics(train_result_list, test_result_list, model_names):
    train_df = pd.DataFrame(train_result_list)
    test_df = pd.DataFrame(test_result_list)
    train_df = train_df.rename(columns={"adjusted_r_2": "Train Adjusted R2", "r_2": "Train R2", "mse": "Train Mean Sq Error"})
    test_df = test_df.rename(columns={"adjusted_r_2": "Test Adjusted R2", "r_2": "Test R2", "mse": "Test Mean Sq Error"})
    df = pd.concat([train_df, test_df], axis=1)
    df["Model"] = model_names
    return df[["Model", "Train Mean Sq Error", "Test Mean Sq Error", "Train R2", "Test R2", "Train Adjusted R2", "Test Adjusted R2"]]
