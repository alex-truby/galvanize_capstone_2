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


class StochasticModels():
    '''
    Prepare data for input into models.
    Run various prediction models.
    '''

    def __init__(self, input_df, target_col_name):
        self.input_df = input_df.copy()
        self.target_col = target_col_name
        self.y = self.input_df.pop(self.target_col).values
        self.X = self.input_df.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.15, train_size = 0.85, random_state=34, shuffle=True)


    def run_random_forest_oob(self, input_df, n_estimators, max_features, random_state=34):
        rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state=random_state)
        rf.fit(self.X_train, self.y_train)
        oob_score = rf.score(self.X_test, self.y_test)
        rf_oob_summary = pd.Series(data={'df_used':input_df,'test_type':'RF', 'n_estimators': n_estimators, 'max_features':max_features, 'oob_score':oob_score})
        return rf_oob_summary

    def run_random_forest_errors(self, input_df, n_estimators, max_features, n_folds, random_state=34):
        rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state=random_state)
        clf = make_pipeline(preprocessing.StandardScaler(), rf)

        #rf_to_plot = clf.fit(self.X_train, self.y_train)

        mse = cross_val_score(clf, self.X_train, self.y_train, scoring='neg_root_mean_squared_error', cv=n_folds, n_jobs=-1) * -1
        r2 = cross_val_score(clf, self.X_train, self.y_train, scoring='r2', cv=n_folds, n_jobs=-1)
        mean_mse = mse.mean()
        mean_r2 = r2.mean()
        rf_errors_summary = pd.Series(data = {'df_used':input_df, 'n_estimators': n_estimators, 'max_features':max_features, 'n_folds':n_folds, 'random_state':random_state, 'root_mean_mse':mean_mse, 'mean_r2':mean_r2 })
        #return clf.steps[1][1], rf_errors_summary, X_std, self.y_train
        return clf.steps[1][1], rf_errors_summary, self.X, self.y, self.X_train, self.y_train

    def run_gradient_boost_errors(self, input_df, learning_rate, n_estimators, subsample, n_folds, random_state=34):
        #will use default loss function of least squares regression
        #subsample <1 will result in stochastic gradient boosting - leads to reduction of variance, increase of bias
        gd = GradientBoostingRegressor(learning_rate = learning_rate, n_estimators=n_estimators, subsample=subsample, random_state = random_state)
        clf = make_pipeline(preprocessing.StandardScaler(), gd)
        mse = cross_val_score(clf, self.X_train, self.y_train, scoring='neg_mean_squared_error', cv=n_folds, n_jobs=-1) * -1
        r2 = cross_val_score(clf, self.X_train, self.y_train, scoring='r2', cv=n_folds, n_jobs=-1)
        mean_mse = mse.mean()
        mean_r2 = r2.mean()
        #puts out errors summary if want to keep running tab in a separate df/text file
        gd_errors_summary = pd.Series(data = {'df_used':input_df, 'n_estimators': n_estimators, 'learning_rate':learning_rate, 
                                                'subsample':subsample, 'n_folds':n_folds, 'random_state':random_state, 'root_mean_mse':mean_mse, 'mean_r2':mean_r2 })
        return clf, gd_errors_summary

    
def feature_importances_plot(col_list, model, X, y):
    #rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state=random_state)


    importances = model.feature_importances_
    std = np.std([iteration.feature_importances_ for iteration in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    #print(X[1])
    print("Feature Ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(X.shape[1]), 
           importances[indices], 
           yerr=std[indices], 
           color="r", 
           align="center")

    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(col_list, rotation = 45, fontsize=12)
    #ax.set_xlim([-1, number_features])
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title("Feature Importances", fontsize=18)
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    post_pca_df = pd.read_csv('./post_pca_full_df.csv')
    
    #test dropping features that were shown not to be needed in linear regression analysis (collinearity between variables) - model does better WITHOUT dropping these. will use original df
    final_lin_df = post_pca_df.drop(['DS_PM_pred', 'EP_MINRTY'], axis=1)


    rf_initiation = StochasticModels(post_pca_df, 'CANCER_CrudePrev')
    rf_model, rf_summary, features_X, features_y, train_X, train_y  = rf_initiation.run_random_forest_errors('post_pca_df', n_estimators = 100, max_features=3, n_folds=5, random_state=34)



    rf_model = rf_model.fit(train_X, train_y)

    feature_list = ['No HS Diploma', 'Diesel HI', '% Minority', 'Acetaldehyde HI', 'Superfund', '% Over 65 Yrs', 'test']
    feature_importances_plot(feature_list, rf_model, features_X, features_y)

    

    
    
    
  