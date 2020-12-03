import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition, datasets

plt.style.use('ggplot')
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}


demo = pd.read_csv('./demographic_df.csv')
y = demo.pop('CANCER_CrudePrev').values
X = demo
X.drop('new_fips', axis=1, inplace=True)

X_std = (X - X.mean(axis=0))/X.std(axis=0, ddof=1)

N = X.shape[0]
A = 1/(N-1)*np.dot(X_std.T, X_std)

eig_vals, eig_vecs = np.linalg.eig(A)

pc1 = np.array([[eig_vecs[0][0], eig_vecs[1][0]]]).T


#utilized to find out how many PC's to use as inputs into models
#initial EDA showed potential correlation between poverty, minority, lack of HS diploma, and limited english rates within census tracts
pca = decomposition.PCA() 
pca.fit(X_std)

total_variance = np.sum(pca.explained_variance_)
cum_variance = np.cumsum(pca.explained_variance_)
prop_var_expl = cum_variance/total_variance

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(prop_var_expl, color='red', linewidth=2, label='Explained variance')
ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)
ax.set_ylabel('Cumulative Proportion of Explained Variance')
ax.set_xlabel('Number of Principal Components')
ax.set_title('Number of Principal Components to Utilize \n From Demographic Dataset')
ax.legend()
plt.show()
#fig.savefig('./dem_pca.png', dpi=200)

