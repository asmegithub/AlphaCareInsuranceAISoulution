import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap

def handle_missing_data(df):
    # Identify columns with missing data
    missing_cols = df.columns[df.isnull().any()]
    
    for col in missing_cols:
        if df[col].dtype == 'object':  # Categorical columns
            # Impute with the most frequent value
            imputer = SimpleImputer(strategy='most_frequent')
        else:  # Numerical columns
            # Impute with the median value
            imputer = SimpleImputer(strategy='median')
        
        df[col] = imputer.fit_transform(df[[col]])
    
    return df
def create_features(df):
    # Example: TotalPremium per Claim Ratio
    df['Premium_Per_Claim'] = df['TotalPremium'] / (df['TotalClaims'] + 1)
    # Example: Create interaction features or transformations
    df['Age_Squared'] = df['Age'] ** 2
    return df

def encode_categorical_data(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

def train_test_split(df, target_col, test_size=0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'MAE': mae, 'MSE': mse, 'R2': r2}
def get_feature_importance(model, feature_names):
    importances = model.feature_importances_  # For Random Forest or Decision Trees
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    return feature_importance

def interpret_model_with_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    # Visualize the SHAP summary plot
    shap.summary_plot(shap_values, X_train)
def compare_models(models, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        evaluation_metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = evaluation_metrics
    
    return pd.DataFrame(results)
