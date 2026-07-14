import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class FeatureSelector:

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.columns = X.columns
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        scaled_x = self.scaler_x.fit_transform(X)
        scaled_y = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        self.X = scaled_x
        self.y = scaled_y.flatten()
    
    def ANOVA_Fvalue(self, k):
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(self.X, self.y)
        selected_features = self.columns[selector.get_support()]
        return selected_features
    
    def mutual_info_reg(self, k):
        selector_mi = SelectKBest(score_func=mutual_info_regression, k=k)
        selector_mi.fit(self.X, self.y)
        selected_features = self.columns[selector_mi.get_support()]
        return selected_features
    
    def RFE(self, k):
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=k)
        selector.fit(self.X, self.y)
        selected_features = self.columns[selector.get_support()]
        return selected_features
    
    def lasso_selector(self, alpha, threshold):
        model = Lasso(alpha=alpha)
        model.fit(self.X, self.y)
        importance = model.coef_
        selected_indices = [i for i, coef in enumerate(importance) if coef >= threshold]
        selected_features = self.columns[selected_indices]
        return selected_features
    
    def tree_selector(self, k):
        model = RandomForestRegressor()
        model.fit(self.X, self.y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        selected_features = self.columns[indices]
        return selected_features