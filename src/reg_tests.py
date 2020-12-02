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
plt.style.use('ggplot')


class PredictionModels():
    '''
    Prepare data for input into models.
    Run various prediction models.
    '''

    def __init__(self, input_df, target_col_name):
        self.input_df = input_df.copy()
        self.target_col = target_col_name
        self.y = self.input_df.pop(self.target_col).values
        self.X = self.input_df.values
        self.X_train = train_test_split(self.X, self.y, test_size = 0.3, train_size = 0.7, random_state=34, shuffle=True)[0]
        self.X_test = train_test_split(self.X, self.y, test_size = 0.3, train_size = 0.7, random_state=34, shuffle=True)[1]
        self.y_train = train_test_split(self.X, self.y, test_size = 0.3, train_size = 0.7, random_state=34, shuffle=True)[2]
        self.y_test = train_test_split(self.X, self.y, test_size = 0.3, train_size = 0.7, random_state=34, shuffle=True)[3]

    def run_random_forest_oob(self, input_df, n_estimators, max_features, random_state=34):
        rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state=random_state)
        rf.fit(self.X_train, self.y_train)
        oob_score = rf.score(self.X_test, self.y_test)
        #return rf.score(self.X_test, self.y_test)
        rf_oob_summary = pd.Series(data={'df_used':input_df,'test_type':'RF', 'n_estimators': n_estimators, 'max_features':max_features, 'oob_score':oob_score})
        # rf_scores_db.append(rf_results_to_append, ignore_index=True)
        # print(oob_score)
        return rf_oob_summary

    def run_random_forest_errors(self, input_df, n_estimators, max_features, n_folds, random_state=34):
        rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state=random_state)
        clf = make_pipeline(preprocessing.StandardScaler(), rf)
        mse = cross_val_score(clf, self.X_train, self.y_train, scoring='neg_root_mean_squared_error', cv=n_folds, n_jobs=-1) * -1
        r2 = cross_val_score(clf, self.X_train, self.y_train, scoring='r2', cv=n_folds, n_jobs=-1)
        mean_mse = mse.mean()
        mean_r2 = r2.mean()
        rf_errors_summary = pd.Series(data = {'df_used':input_df, 'n_estimators': n_estimators, 'max_features':max_features, 'n_folds':n_folds, 'random_state':random_state, 'mean_mse':mean_mse, 'mean_r2':mean_r2 })
        return rf_errors_summary

    def run_gradient_descent_errors(self, input_df, learning_rate, n_estimators, subsample, n_folds, random_state=34):
        #will use default loss function of least squares regression
        #subsample <1 will result in stochastic gradient boosting - leads to reduction of variance, increase of bias
        gd = GradientBoostingRegressor(learning_rate = learning_rate, n_estimators=n_estimators, subsample=subsample, random_state = random_state)
        clf = make_pipeline(preprocessing.StandardScaler(), gd)
        mse = cross_val_score(clf, self.X_train, self.y_train, scoring='neg_mean_squared_error', cv=n_folds, n_jobs=-1) * -1
        r2 = cross_val_score(clf, self.X_train, self.y_train, scoring='r2', cv=n_folds, n_jobs=-1)
        mean_mse = mse.mean()
        mean_r2 = r2.mean()
        gd_errors_summary = pd.Series(data = {'df_used':input_df, 'n_estimators': n_estimators, 'learning_rate':learning_rate, 
                                                'subsample':subsample, 'n_folds':n_folds, 'random_state':random_state, 'mean_mse':mean_mse, 'mean_r2':mean_r2 })
        return gd_errors_summary

    


#can probably delete this entire section before uploading to GITHUB!
if __name__ == '__main__':
    #full dataframe
    df = pd.read_csv('./combined_df.csv')

    #only demographic data + cancer rates
    demo = pd.read_csv('./demographic_df.csv')

    #all environmental columns + cancer rates
    enviro = pd.read_csv('./enviro_df.csv')

    #averaged chem HIs + remaining envrionmental columns + cancer rates
    env_avg_df = pd.read_csv('./env_avg_df.csv')

    #drop unnecessary targets from dataframe 
    df.drop(['CASTHMA_CrudePrev', 'CASTHMA_Crude95CI', 'COPD_CrudePrev', 'COPD_Crude95CI', 'StateAbbr', 'PlaceName', 'Population2010', 
        'ACETALDEHYDE_cancer_risk_per_million', 'BENZENE_cancer_risk_per_million', '1,3-BUTADIENE_cancer_risk_per_million', 
        'CYANIDE COMPOUNDS_cancer_risk_per_million', 'DIESEL PM_cancer_risk_per_million', 'TOLUENE_cancer_risk_per_million',
        'BENZENE_repiratory_HI', '1,3-BUTADIENE_repiratory_HI', 'CYANIDE COMPOUNDS_repiratory_HI', 'TOLUENE_repiratory_HI',
        'CANCER_Crude95CI', 'DS_PM_stdd', 'new_fips'], axis = 1, inplace=True)
    
    
    rf_scores_db = pd.DataFrame(columns=['test_type', 'n_estimators', 'max_features', 'oob_score'])


    post_pca_df1 = df.copy()
    post_pca_df1.drop(['EP_POV', 'EP_LIMENG'], axis=1, inplace=True)
    #post_pca_df1.to_csv (r'post_pca_full_df.csv', index=False, header=True)

    # test_df = post_pca_df1.copy()
    # test_df.drop(['has_superfund', 'ACETALDEHYDE_repiratory_HI', 'DIESEL PM_repiratory_HI',
    #                 'DS_PM_pred'], axis=1, inplace=True)
    # test_df.to_csv (r'test_df.csv', index=False, header=True)
