import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
# from sklearn.metrics import davies_bouldin_score
# from sklearn import metrics
# from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
# from sklearn.compose import ColumnTransformer

d1 = pd.read_csv('output.csv')
d4 = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataold = pd.read_csv('new - Sheet1.csv')
d7 = pd.read_csv('general_data.csv')
d8 = pd.read_csv('HRDataset_v13.csv')
d4 = d4.head(2749)
d7 = d7.head(2749)



d71 = d7['Education'].tolist()
d72 = d7['MaritalStatus'].tolist()
d41 = d4['customerID'].tolist()
d42 = d4['SeniorCitizen'].tolist()
d43 = d4['InternetService'].tolist()
d11 = d1['Citizen'].tolist()
d12 = d1['Insurance_claim'].tolist()
dataold['Education'] = d71
dataold['MaritalStatus'] = d72
dataold['customerID'] = d41
dataold['SeniorCitizen'] = d42
dataold['InternetService'] = d43
dataold['Citizen'] = d11
dataold['Insurance_claim'] = d12

labelencoder_X = LabelEncoder()
dataold.iloc[:, 1] = labelencoder_X.fit_transform(dataold.iloc[:, 1])

dataold.iloc[:, 6] = labelencoder_X.fit_transform(dataold.iloc[:, 6])

dataold.iloc[:, 10] = labelencoder_X.fit_transform(dataold.iloc[:, 10])

dataold.iloc[:, 9] = labelencoder_X.fit_transform(dataold.iloc[:, 9])

dataold.iloc[:, 4] = labelencoder_X.fit_transform(dataold.iloc[:, 4])

dataold.iloc[:, 0] = labelencoder_X.fit_transform(dataold.iloc[:, 0])

data = dataold[[' customerID','City', 'Gender', 'Age', 'Income', 'Illness', 'Education', 'MaritalStatus', 'SeniorCitizen', 'InternetService', 'Citizen', 'Insurance_claim']].copy()
ClusteringData = dataold[['City', 'Gender', 'Age', 'Income', 'Illness', 'Education', 'MaritalStatus', 'SeniorCitizen', 'InternetService', 'Citizen', 'Insurance_claim']].copy()


ClusteringData = ClusteringData.sample(frac=1).reset_index(drop=True)


#Data Cleaning

#Scaling the data to bring all the attributes to a comparable level

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(ClusteringData) 


#Normalizing the data so that the data approximately follows a Gaussian distribution

X_normalized = normalize(X_scaled) 

X_normalized = pd.DataFrame(X_normalized) 


#PCA (Principle Component analys

pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 


#Performing Clustering (DBSCAN)

db_default = DBSCAN(eps = 0.038, min_samples = 8).fit(X_principal) 
core_samples_mask = np.zeros_like(db_default.labels_, dtype=bool)
core_samples_mask[db_default.core_sample_indices_] = True
labels = db_default.labels_ 

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]

# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)
#     X = np.array(X_principal)
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=15)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
    
# plt.title('DBSCAN: Estimated number of clusters: %d' % n_clusters_)
# F = plt.gcf()
# Size = F.get_size_inches()
# F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)
# plt.show()


clusterValue_NewEntry = labels[-1]
# clusterValue_NewEntry

a = []
output = []

for i in range(0, len(labels)):
    if(labels[i] == clusterValue_NewEntry):
        a.append(i)

a.pop(-1)

for i in a:
    print("I is this", i)
    output.append(data.loc[i])
    # print(data.loc[i])
    # print('*****************')

output = pd.DataFrame(output)

# model = pickle.load(open('model.pkl','rb'))
