from flask import Flask, request, render_template, send_file,  Response
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
from sklearn.linear_model import LogisticRegression
# from sklearn.compose import ColumnTransformer
from itertools import product


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index_post():
    name = request.form['name']
    email = request.form['email']
    product = request.form['product']
    description = request.form['description']
    city = ', '.join(request.form.getlist('city'))
    gender = ', '.join(request.form.getlist('gender'))
    age = ', '.join(request.form.getlist('age'))
    income = ', '.join(request.form.getlist('income'))
    education = ', '.join(request.form.getlist('education'))
    marital_status = ', '.join(request.form.getlist('marital_status'))
    seni = ', '.join(request.form.getlist('senior'))
    IS = ', '.join(request.form.getlist('IS'))
    citizen = ', '.join(request.form.getlist('citizen'))
    insurance = ', '.join(request.form.getlist('insurance'))

    return render_template('output.html', n = name, e = email, p = product, d = description, c = city, g = gender, a = age, i = income, edu = education, ms = marital_status, s = seni, internet = IS, citizen = citizen, insurance = insurance)


@app.route('/result', methods=['POST'])
def result_output():

    city = ', '.join(request.form.getlist('city'))
    gender = ', '.join(request.form.getlist('gender'))
    age = ', '.join(request.form.getlist('age'))
    income = ', '.join(request.form.getlist('income'))
    education = ', '.join(request.form.getlist('education'))
    marital_status = ', '.join(request.form.getlist('marital_status'))
    seni = ', '.join(request.form.getlist('senior'))
    IS = ', '.join(request.form.getlist('IS'))
    citizen = ', '.join(request.form.getlist('citizen'))
    insurance = ', '.join(request.form.getlist('insurance'))

    city = list(city.split(", "))
    gender = list(gender.split(", "))
    age = list(age.split(", "))
    income = list(income.split(", "))
    education = list(education.split(", "))
    marital_status = list(marital_status.split(", "))
    seni = list(seni.split(", "))
    IS = list(IS.split(", "))
    citizen = list(citizen.split(", "))
    insurance = list(insurance.split(", "))


    d1 = pd.read_csv('Dataset/output.csv')
    d4 = pd.read_csv('Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    dataold = pd.read_csv('Dataset/new - Sheet1.csv')
    d7 = pd.read_csv('Dataset/general_data.csv')
    d8 = pd.read_csv('Dataset/HRDataset_v13.csv')
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

    data = dataold[['customerID','City', 'Gender', 'Age', 'Income', 'Illness', 'Education', 'MaritalStatus', 'SeniorCitizen', 'InternetService', 'Citizen', 'Insurance_claim']].copy()

    labelencoder_X = LabelEncoder()
    dataold.iloc[:, 1] = labelencoder_X.fit_transform(dataold.iloc[:, 1])

    dataold.iloc[:, 6] = labelencoder_X.fit_transform(dataold.iloc[:, 6])

    dataold.iloc[:, 10] = labelencoder_X.fit_transform(dataold.iloc[:, 10])

    dataold.iloc[:, 9] = labelencoder_X.fit_transform(dataold.iloc[:, 9])

    dataold.iloc[:, 4] = labelencoder_X.fit_transform(dataold.iloc[:, 4])

    dataold.iloc[:, 0] = labelencoder_X.fit_transform(dataold.iloc[:, 0])


    ClusteringData = dataold[['City', 'Gender', 'Age', 'Income', 'Illness', 'Education', 'MaritalStatus', 'SeniorCitizen', 'InternetService', 'Citizen', 'Insurance_claim']].copy()


    ClusteringData = ClusteringData.sample(frac=1).reset_index(drop=True)

    # d = pd.Series(ClusteringData.iloc[79])
    # ClusteringData = ClusteringData.append(d)



    #Data Cleaning

    #Scaling the data to bring all the attributes to a comparable level

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ClusteringData)


    #Normalizing the data so that the data approximately follows a Gaussian distribution

    X_normalized = normalize(X_scaled)

    X_normalized = pd.DataFrame(X_normalized)


    #PCA (Principle Component analysis)

    pca = PCA(n_components = 2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']


    #Performing Clustering (DBSCAN)

    db_default = DBSCAN(eps = 0.038, min_samples = 8).fit(X_principal)
    core_samples_mask = np.zeros_like(db_default.labels_, dtype=bool)
    core_samples_mask[db_default.core_sample_indices_] = True
    labels = db_default.labels_

    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

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


    for i in city:
        if(i == "No particular city"):
            city = ['Dallas', 'New York City', 'Austin', 'San Diego', 'Los Angeles', 'Mountain View']

    for i in gender:
        if(i == "Doesn't matter"):
            gender = ['Male', 'Female']

    for i in range(0, len(age)):
        if(age[i] == "Below 20"):
            age[i] = int("16")

        elif(age[i] == "20-40"):
            age[i] = int("30")

        elif(age[i] == "40-60"):
            age[i] = int("50")

        elif(age[i] == "60 above"):
            age[i] = int("65")

        elif(i == "Doesn't matter"):
            age = [16, 30, 50, 65]


    for i in range(0, len(income)):
        if(income[i] == "below 20,000"):
            income[i] = int("16000")

        elif(income[i] == "20,000-60,000"):
            income[i] = int("40000")

        elif(income[i] == "60,000-100,000"):
            income[i] = int("80000")

        elif(income[i] == "100,000 above"):
            income[i] = int("100000")

        elif(i == "Doesn't matter"):
            income = [16000, 40000, 80000, 100000]


    for i in range(0, len(education)):
        if(education[i] == "Below High School"):
            education[i] = int("1")

        elif(education[i] == "High School"):
            education[i] = int("2")

        elif(education[i] == "Junior College"):
            education[i] = int("3")

        elif(education[i] == "Graduation"):
            education[i] = int("4")

        elif(education[i] == "Masters"):
            education[i] = int("5")

        elif(education[i] == "Doesn't matter"):
            education = [1,2,3,4,5]


    for i in marital_status:
        if(i == "Doesn't matter"):
            marital_status = ['Single', 'Married', 'Divorced']


    for i in range(0, len(seni)):
        if(seni[i] == "Yes"):
            seni[i] = int("1")

        elif(seni[i] == "No"):
            seni[i] = int("0")

        elif(seni[i] == "Doesn't matter"):
            seni = [1, 0]


    for i in IS:
        if(i == "NO"):
            IS = ['DSL', 'Fiber optic']


    for i in citizen:
        if(i == "Non US Citizen"):
            i = "NonCitizen"

        elif(i == "Doesn't matter"):
            citizen = ["US Citizen", "NonCitizen"]


    for i in range(0, len(insurance)):
        if(insurance[i] == "Yes"):
            insurance[i] = int("1")

        elif(insurance[i] == "No"):
            insurance[i] = int("0")

        elif(insurance[i] == "Doesn't matter"):
            insurance = [1, 0]


    finallist = [city, gender, age, income, education, marital_status, seni, IS, citizen, insurance ]

    lists = list(product(*finallist))
    # lists = list(ClusteringData.columns)
    # length = len(lists)

    prediction = pd.DataFrame(lists, columns=['City','Gender','Age','Income','Education','MaritalStatus','SeniorCitizen','InternetService','Citizen','Insurance_claim'])

    ClassificationData = data[['City', 'Gender', 'Age', 'Income', 'Education', 'MaritalStatus', 'SeniorCitizen', 'InternetService', 'Citizen', 'Insurance_claim']].copy()

    ClassificationData['class'] = labels

    labelencoder_p = LabelEncoder()
    prediction.iloc[:, 0] = labelencoder_p.fit_transform(prediction.iloc[:, 0])
    prediction.iloc[:, 1] = labelencoder_p.fit_transform(prediction.iloc[:, 1])
    prediction.iloc[:, 5] = labelencoder_p.fit_transform(prediction.iloc[:, 5])
    prediction.iloc[:, 7] = labelencoder_p.fit_transform(prediction.iloc[:, 7])
    prediction.iloc[:, 8] = labelencoder_p.fit_transform(prediction.iloc[:, 8])

    labelencoder_C = LabelEncoder()
    ClassificationData.iloc[:, 0] = labelencoder_C.fit_transform(ClassificationData.iloc[:, 0])
    ClassificationData.iloc[:, 1] = labelencoder_C.fit_transform(ClassificationData.iloc[:, 1])
    ClassificationData.iloc[:, 5] = labelencoder_C.fit_transform(ClassificationData.iloc[:, 5])
    ClassificationData.iloc[:, 7] = labelencoder_C.fit_transform(ClassificationData.iloc[:, 7])
    ClassificationData.iloc[:, 8] = labelencoder_C.fit_transform(ClassificationData.iloc[:, 8])


    ClassificationData = ClassificationData.sample(frac=1).reset_index(drop=True)

    X = pd.DataFrame(ClassificationData.iloc[:,:-1])
    y = pd.DataFrame(ClassificationData.iloc[:,-1])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    scalerC = StandardScaler()
    C_scaled = scalerC.fit_transform(X)
    C_scaledT = scalerC.fit_transform(prediction)

    c_normalized = normalize(C_scaled)
    c_normalizedT = normalize(C_scaledT)
    C_normalized = pd.DataFrame(c_normalized)
    C_normalizedT = pd.DataFrame(c_normalizedT)

    logmodel = LogisticRegression()
    logmodel.fit(C_normalized,y.values.ravel())

    y_pred = logmodel.predict(C_normalizedT)
    output = list(set(y_pred))

    # clusterValue_NewEntry = labels[-1]
    #  clusterValue_NewEntry



    a = []
    out = []

    for i in range(0, len(labels)):
        if(labels[i] == output): #change the comparison
            a.append(i)

    # a.pop(-1)

    for i in a:
        # print("I is this", i)
        out.append(data.loc[i])
        # print(data.loc[i])
        # print('*****************')
    #
    out = pd.DataFrame(out)
    # o = len(out)
    result = out['customerID']
    l = len(out)
    result.to_csv(r'output/targetOutput.csv', index=False)


    # c = city, g = gender, a = age, i = income, edu = education, ms = marital_status, s = seni, internet = IS, citizen = citizen, insurance = insurance, lll = l

    return render_template('temp.html')


@app.route("/getPlotCSV")
def getPlotCSV():
    # with open("outputs/Adjacency.csv") as fp:
    #     csv = fp.read()
    # csv = pd.read_csv('E:/FilesandFolders/collegeProject/output/targetOutput.csv')
    return send_file('output/targetOutput.csv',
                     mimetype='text/csv',
                     attachment_filename='targetOutput.csv',
                     as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
