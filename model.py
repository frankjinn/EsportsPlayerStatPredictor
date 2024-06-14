import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression

def prepare_data(df, target_column, feature_columns):
    """
    Prepare data for model training and evaluation.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.
        target_column (str): The name of the target column.
        feature_columns (list): List of feature column names.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets for training and testing.
    """
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train_model(X_train, y_train, model, param_grid, feature_selection=False):
    """
    Train a machine learning model with optional feature selection using RFECV.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        model: The machine learning model to train.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        feature_selection (bool): Whether to perform feature selection using RFECV.

    Returns:
        grid_search: Trained model using GridSearchCV.
    """
    steps = [('scaler', StandardScaler()), ('model', model)]

    if feature_selection:
        # Using RFECV for feature selection
        selector = RFECV(estimator=model, step=1, cv=TimeSeriesSplit(n_splits=5))
        steps.insert(1, ('selector', selector))

    pipeline = Pipeline(steps)

    time_split = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=time_split)
    grid_search.fit(X_train, y_train)

    return grid_search

def evaluate_model(grid_search, X_test, y_test):
    """
    Evaluate the trained model and print mean squared error and feature weights.

    Args:
        grid_search: Trained model using GridSearchCV.
        X_test (pd.DataFrame): Testing feature data.
        y_test (pd.Series): Testing target data.
    """
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 score          : {r2}")

    if hasattr(grid_search.best_estimator_, 'named_steps') and \
       'model' in grid_search.best_estimator_.named_steps:
        model = grid_search.best_estimator_.named_steps['model']
        if hasattr(model, 'coef_'):
            print("\nFeature Weights:")
            weights = model.coef_
            features = X_test.columns
            if isinstance(model, LogisticRegression):
                weights = weights[0]
            for feature, weight in zip(features, weights):
                print(f"{feature}: {weight:.4f}")
            return list(zip(features, weights))

def get_model_weights(grid_search):
    """
    Get model weights as JSON.

    Args:
        grid_search: Trained model using GridSearchCV.

    Returns:
        str: JSON representation of model weights.
    """
    if hasattr(grid_search.best_estimator_, 'named_steps') and \
       'model' in grid_search.best_estimator_.named_steps:
        model = grid_search.best_estimator_.named_steps['model']
        if hasattr(model, 'coef_'):
            weights = model.coef_.tolist()
            # print(dir(grid_search.best_estimator_))
            # features = grid_search.best_estimator_.named_steps['scaler'].get_feature_names_out().tolist()
            features = ['x' for x in range(len(weights))]
            intercept = model.intercept_.item()
            model_weights = {'features': dict(zip(features, weights)), 'intercept': intercept}
            return json.dumps(model_weights, indent=4)
    return json.dumps({})

def model_pipeline(df, stat, model, param_grid,feature_columns_input = None, feature_selection=False):
    """
    Train and evaluate a machine learning model for a specific statistic.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.
        stat (str): The name of the statistic.
        model: The machine learning model to use.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        feature_selection (bool): Whether to perform feature selection using RFECV.

    Returns:
        grid_search: Trained model using GridSearchCV.
    """
    df = df.copy()
    target_column = f"{stat}_expected"
    if feature_columns_input is None:
        feature_columns = [column_name for column_name in df.columns if
                           f"{stat}_" in column_name and "expected" not in column_name]
    else:
        feature_columns = feature_columns_input
    print(feature_columns)
    df = df.dropna(subset=[target_column])
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, feature_columns)
    grid_search = train_model(X_train, y_train, model, param_grid, feature_selection)
    print("Best Parameters:", grid_search.best_params_)
    feat_coef = evaluate_model(grid_search, X_test, y_test)
    """
    model_weights_json = get_model_weights(grid_search)
    with open("model.json", 'w') as file:
        json.dump(model_weights_json, file, indent=4)
    print("\nModel Weights (JSON):", model_weights_json)
    """
    return feat_coef
