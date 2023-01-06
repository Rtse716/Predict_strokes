from statistics import mean
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
import seaborn as sns
from matplotlib import pyplot as plt
from mlxtend.evaluate import accuracy_score, bias_variance_decomp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn import preprocessing, svm
from imblearn.over_sampling import RandomOverSampler

stroke_df = pd.read_excel(r"C:\Users\rinat\OneDrive\Desktop\SCHOOL STUFF\Machine Learning\stroke_data.xlsx")
stroke_df = stroke_df.drop(['id'], axis=1)

def whitespace_remover(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].map(str.strip)
        else:
            pass


# applying whitespace_remover function on dataframe
whitespace_remover(stroke_df)

# Apply One Hot Encoder
stroke_df = pd.get_dummies(stroke_df)

# Impute NaN values
stroke_df = stroke_df.fillna(1000000000)
stroke_df = stroke_df.replace(100000000, np.nan)
imputer = KNNImputer(n_neighbors=5)
stroke_df = pd.DataFrame(imputer.fit_transform(stroke_df), columns=stroke_df.columns)

# Produce Excel for visualization purposes
stroke_df.to_excel('test.xlsx')


'''
#Visualize y class
stroke_df['stroke'].value_counts().plot.bar()
plt.title('Stroke Counts', fontsize=18)
plt.xlabel('Stroke Occurrence', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()
'''

# Drop target column
features_stroke = pd.DataFrame(stroke_df).drop(['stroke'], axis=1)
stroke_all_features = features_stroke.columns.values.tolist()

# Set X and Y
X = features_stroke
X_normalized = preprocessing.normalize(X, norm='l2')
stroke_target_cols = ['stroke']
y = (stroke_df[stroke_target_cols].astype(int).values.ravel())

# Apply Oversampling
ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(X_normalized, y)

'''
target = pd.DataFrame(y_ros)
target.value_counts().plot.bar()
plt.title('Oversampled_Stroke Count', fontsize=18)
plt.xlabel('Stroke Occurrence', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()
'''
# Create Test/Train split
X_train, X_test, y_train, y_test = train_test_split(x_ros, y_ros, test_size=0.2,
                                                    random_state=42)  # 80% training and 20% test
# Apply svm classifier
clf = svm.SVC(decision_function_shape='ovo')

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Create RFE pipeline
rfe = RFECV(estimator=DecisionTreeClassifier(random_state=42))
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s', rfe), ('m', model)])

# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=5)
n_scores = cross_val_score(pipeline, x_ros, y_ros, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print((mean(n_scores)))

# summarize all features
rfe.fit(X, y)
feature_cols = []
for i, j in zip(range(X.shape[1]), X.columns):
    if rfe.support_[i] == True:
        feature_cols.append(j)
        print(i, rfe.support_[i], rfe.ranking_[i])
print(feature_cols)

# Train Model to Predict Stroke

# Set X and Y
X = (stroke_df[feature_cols].astype(int))
X_normalized = preprocessing.normalize(X, norm='l2')
target_cols = ['stroke']
y = (stroke_df[target_cols].astype(int).values.ravel())

# Set training and testing split
X_train, X_test, y_train, y_test = train_test_split(x_ros, y_ros, test_size=0.26, random_state=1)

# kNN
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=10)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(x_ros, y_ros)

'''
#Other models to try:
clf = svm.SVC(decision_function_shape='ovr')
lr = LogisticRegression()
gnb = GaussianNB()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
'''

# Predict the response for test dataset
y_pred = nca_pipe.predict(X_test)

# Produce Confusion Matrix
y_test_list = (list(y_test))
y_pred_list = (list(y_pred))
PP = []
PN = []
NN = []
NP = []
for i, j in zip(y_test_list, y_pred_list):
    if i == 0 and j == 0:
        NN.append(1)
    elif i == 1 and j == 0:
        PN.append(1)
    elif i == 0 and j == 1:
        NP.append(1)
    elif i == 1 and j == 1:
        PP.append(1)

print(len(PP), len(PN))
print(len(NP), len(NN))

# Get Accuracy Score
print(accuracy_score(y_test, y_pred))

# Estimate bias and variance
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(nca_pipe, X_train, y_train, X_test, y_test, loss='mse',
                                                            num_rounds=50, random_seed=20)

# Summary of the results
print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
