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
import pylab as py
import  statsmodels.formula.api as smf
import scipy.stats as stats
from statsmodels.formula.api import ols
from yellowbrick.regressor import CooksDistance
plt.style.use('ggplot')



#check for colinearity with VIF
def colinearity_check(X):
    for i, col_name in enumerate(list(X.columns)):
        print(f'{col_name}', variance_inflation_factor(X.values, i))

def regression_model(y, X):
    regression_model = sm.OLS(y, X)
    regression_model = regression_model.fit()
    return regression_model

def create_qq_plot(model, y_train, X_train, y_test):
    y_preds = model.predict(X_test)
    resids = y_test - y_preds
    sm.qqplot(resids, line='45')
    plt.show()
    plt.savefig('./qq_plot_before_removing_outliers.png', dpi=200)


def homoscedasticity_test_plot(model, y_train, X_train, y_test):
    y_preds = model.predict(X_test)
    resids = y_test - y_preds

    fig, ax = plt.subplots(figsize = (9,9))
    ax.scatter(y_test, resids)
    ax.set_title('Residuals v. Target Variable')
    ax.set_xlabel('Target Values')
    ax.set_ylabel('Model Residuals')
    #ax.plot(xx, [0]*100, color='red, lw=3')
    #fig.savefig('./linear_reg_residualsplot_normal_target.png', dpi=200)
    plt.show()



def check_for_outliers(y, X_col):
    X_train, X_test, y_train, y_test = train_test_split(X_col, y, train_size  = 0.8, random_state = 34, shuffle=True)

    model = sm.OLS(y_train, X_train)
    model = model.fit()
    y_preds = model.predict(X_test)
    resids = y_test - y_preds



#     # f = 'y ~ X_col'
#     # model = smf.ols(formula=f, data=df).fit()
#     # print('R-Squared:', model.rsquared)
#     # print(model.params)
#     # X_new = pd.DataFrame({'X-col': [df.X_col.min(), df.bedrooms.max()]})

#     # preds = model.predict(X_new)

#     # # df.plot(kind='scatter', x='X_col', y='y')
#     # # plt.plot(X_new, preds, c='blue', linewidth=2)
#     # # plt.show()

#     # # fig  = plt.figure(figsize=(15,8))
#     # # fig = sm.graphics.plot_regress_exog(model, 'X_col', fiig=fig)
#     # # plt.show()
#     # # fig=plt.figure(figsize = (15,8))



    # residuals=model.resid
    infl = model.get_influence()
    sm_fr = infl.summary_frame()

    visualizer = CooksDistance()
    visualizer.fit(X_col.values.reshape(-1,1), y)
    visualizer.show()
    return sm_fr

def drop_rows(y_df, X_df, index):
    y_df.drop(index, axis=0, inplace=True)
    X_df.drop(index, axis=0, inplace=True)
    return y_df, X_df

if __name__ == '__main__':
    post_pca_df = pd.read_csv('./post_pca_full_df.csv')

    # y = post_pca_df.pop('CANCER_CrudePrev')
    # X = post_pca_df
    #X = sm.add_constant(X)

    #check for collinearity within the data set
    #collineratity will all X features below. will drop DS_PM_pred & EP_MINRTY
    # has_superfund 1.00
    # ACETALDEHYDE_repiratory_HI 14.65
    # DIESEL PM_repiratory_HI 3.26
    # EP_MINRTY 9.22
    # EP_NOHSDP 4.73
    # EP_AGE65 4.16
    # DS_PM_pred 19.71
    
    final_lin_df = post_pca_df.drop(['DS_PM_pred', 'EP_MINRTY'], axis=1)
    y = final_lin_df.pop('CANCER_CrudePrev')
    X = final_lin_df

    #colinearity_check(X)
    # second collinearity check resulted in features all below a VIF of 10:
    # has_superfund 1.00
    # ACETALDEHYDE_repiratory_HI 6.91
    # DIESEL PM_repiratory_HI 3.22
    # EP_NOHSDP 2.37
    # EP_AGE65 3.51


    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size  = 0.8, random_state = 34, shuffle=True)
    # OLS_model = regression_model(y_train, X_train)

    #create_qq_plot(OLS_model, y_train, X_train, y_test)

    #homoscedasticity_test_plot(OLS_model, y_train, X_train, y_test)

    # print(regression_model(y_train, X_train).summary())

    # has_superfund = X['has_superfund']
    # acet = X['ACETALDEHYDE_repiratory_HI']
    # hs_dip = X['EP_NOHSDP']
    # diesel = X['DIESEL PM_repiratory_HI']
    # over_65 = X['EP_AGE65']

    # test_outlier = check_for_outliers(y, diesel)
    
    # for col in test_outlier.columns:
    #     print(test_outlier[col].max())
        
    #print(test_outlier['cooks_d'].values.argmax())

    largest_cooks_distance_indices = []
    largest_cooks_distance_sf_65 = []
    # for col in X.columns:
    #     test_outlier = check_for_outliers(y, X[col])
    #     largest_cooks_distance_indices.append(test_outlier['cooks_d'].values.argmax())

    # for col in ['has_superfund', 'EP_AGE65']:
    #     test_outlier = check_for_outliers(y, X[col])
    #     largest_cooks_distance_indices.append(test_outlier['cooks_d'].values.argmax())

    indices_to_drop = [18392, 15934, 5911]
    for index in indices_to_drop:
        drop_rows(y, X, index)
    print(y.shape, X.shape)
 
