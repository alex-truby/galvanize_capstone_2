from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels
import statsmodels.api as sm
from scipy import stats
plt.style.use('ggplot')



#check for colinearity with VIF
def colinearity_check(X):
    for i, col_name in enumerate(list(X.columns)):
        print(f'{col_name}', variance_inflation_factor(X.values, i))

def regression_model(y, X):
    regression_model = sm.OLS(y, X)
    regression_model = regression_model.fit()
    return regression_model

def homoscedasticity_test_plot(model, y_train, X_train, y_test):
    y_preds = model.predict(X_test)
    resids = y_test - y_preds

    fig, ax = plt.subplots(figsize = (9,9))
    ax.scatter(y_test, resids)
    ax.set_title('Residuals v. Target Variable')
    ax.set_xlabel('Target Values')
    ax.set_ylabel('Model Residuals')
    #ax.plot(xx, [0]*100, color='red, lw=3')
    fig.savefig('./linear_reg_residualsplot_normal_target.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    post_pca_df = pd.read_csv('./post_pca_full_df.csv')

    y = post_pca_df.pop('CANCER_CrudePrev')
    X = post_pca_df
    X_std = (X - X.mean(axis=0))/X.std(axis=0, ddof=1)
    X_std = sm.add_constant(X_std)

    X_train, X_test, y_train, y_test = train_test_split(X_std, y, train_size  = 0.8, random_state = 34, shuffle=True)

    

    OLS_model = regression_model(y_train, X_train)
    #run if want to perform VIF test for collinearity of input features (after standardization)
    #colinearity_check(X_std)

    homoscedasticity_test_plot(OLS_model, y_train, X_train, y_test)