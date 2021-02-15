

# # 1. Data preparation


# import libraries that we may need (will be added)
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
import xgboost
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# let's see datasets what we got:


# information about contracts
contract = pd.read_csv('/Users/Ilia/PycharmProjects/customer_churn_forecast/datasets_customer_churn_forecast/contract.csv')
# personal information about clients
personal = pd.read_csv('/Users/Ilia/PycharmProjects/customer_churn_forecast/datasets_customer_churn_forecast/personal.csv')
# information about internet-services
internet = pd.read_csv('/Users/Ilia/PycharmProjects/customer_churn_forecast/datasets_customer_churn_forecast/internet.csv')
# information about phone services
phone = pd.read_csv('/Users/Ilia/PycharmProjects/customer_churn_forecast/datasets_customer_churn_forecast/phone.csv')

# each table is assigned a variable in accordance with the content


# will increase the amount of displayed information when displaying
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


# let's see the first 5 rows of each of the tables and the basic information on the data
print(contract.head(5))
contract.info()
print()
print(personal.head(5))
personal.info()
print()
print(internet.head(5))
internet.info()
print()
print(phone.head(5))
phone.info()
print()

# Target attribute "EndDate" is presented in the "contract" table.
# A quick inspection reveals the presence of the customerID column.
# It will be possible to merge tables using it.
# There are columns with incorrect data format or answers (yes / no).
# A little later, we will bring everything to a single format.

# In order to see general information on the attributes with the "object" format, we will write a small function.
def desc(data):
    cat_col = []
    for i in data.columns:
        if data[i].dtype.name == 'object':
            cat_col.append(i)
    print(data[cat_col].describe())


# First, check the table with contract data
print(contract['PaymentMethod'].unique())


# A few observations:
# 1. You can notice a small number of unique values by the start date of the contract.
# I'll assume that these were package or share connections (home / office / district connection ...).
# 2. Only 5 unique expiration dates of the contract and the prevailing number of contracts that are still in force (5174 vs 1869).
# Clients leave exactly on the first hour of each month, starting from 12/01/19,
# presumably with the end of the contract with a monthly payment.
# Perhaps it was from this date that the competitor began to be active.
## 3. The most popular payment option is the monthly payment for the company's services,
# and these contracts account for more than half of the customers.
# 4. Most people prefer to pay with electronic checks, but not a small number of payments fall on paper billing (2872 users).

# Analysis of the table by personal data of clients
print(desc(personal))


# # There were no external anomalies in the personal data of clients.


# Analysis of the table by internet-services
print(desc(internet))


# You may notice that most people prefer Fiber optic and also refuse additional services.
# Except for streaming movies and TV (approximately 50/50).



# Analysis of the table by phone-services
print(desc(phone))

print(phone.shape[0] / contract.shape[0])
print(internet.shape[0] / contract.shape[0])


# Most of the clients refuse the service of connecting a telephone set to several lines at the same time.
# Having analyzed all four tables in aggregate, we can conclude that ~ 90% of customers use telephony services, ~ 78% use the Internet,
# which indicates that most of the customers use a package of services (Internet + telephony), and not separately.


# Check all four tables for nan's
print(contract.isna().sum())
print(personal.isna().sum())
print(internet.isna().sum())
print(phone.isna().sum())


# Select the target feature in a separate column and name it "Gone".
contract['Gone'] = 0
for i in range(len(contract)):
    if contract['EndDate'][i] == 'No':
          contract['Gone'][i] = 0
    else:
          contract['Gone'][i] = 1

print(contract.head())
print()
# It is now easier to track who closed contracts


# Transform a piece of data in the contract table:
# Transform data in columns with the beginning and end of the contract in the date format
contract['BeginDate'] = pd.to_datetime(contract['BeginDate'], format = '%Y-%m-%d')
contract['EndDate'] = contract['EndDate'].replace('No','2020-02-01')
contract['EndDate'] = pd.to_datetime(contract['EndDate'], errors='coerce')

# Add a column in which we will count the number of days of validity of the contract for those who left
contract['Difference'] = (contract['EndDate'] - contract['BeginDate']).dt.days


# Remove the columns with dates so that in the future they will not interfere with our prediction
contract = contract.drop(columns = 'BeginDate')
contract = contract.drop(columns = 'EndDate')


# Convert the binary values of the columns to 0 and 1 in tables
# There is only 1 column in the "contract" and "phone" tables.
contract['PaperlessBilling'] = pd.get_dummies(contract['PaperlessBilling'], drop_first = True)
phone['MultipleLines'] = pd.get_dummies(phone['MultipleLines'], drop_first = True)

# Create a list of binary columns for "personal" and "internet" of the tables
personal_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

internet_col = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


# Loop through the lists of binary columns in both tables
for i in internet_col:
    internet[i] = pd.get_dummies(internet[i], drop_first = True)

for p in personal_col:
    personal[p] = pd.get_dummies(personal[p], drop_first = True)


# Merge all four tables in the one.

data = pd.merge(contract,personal, on='customerID', how='outer')
data = pd.merge(data,internet, on='customerID', how='outer')
data = pd.merge(data,phone, on='customerID', how='outer')

# Convert string data types to numeric
data = data[data['TotalCharges'] != " "]
data['TotalCharges'] = data['TotalCharges'].astype('float64')

# 11 empty strings were found when I trying to convert the values of the "TotalCharges" column
# I decided to remove them, because their number was not relative to the entire dataset.

data['MonthlyCharges'].isna().sum()


# # 2. Exploratory data analysis

# Let's research monthly user payments in the resulting table

print(data['MonthlyCharges'].describe())

#
# data.boxplot('MonthlyCharges')
# plt.show()
# data['MonthlyCharges'].hist(bins = 20)
# plt.xlabel('monthly charges')
# plt.ylabel('amount of users')
# plt.show()
#
# # # Let's look at the distribution of monthly charges for those customers who closed the contract with the company
# # and those who stayed.
# print('Statistics on monthly customer charges that remained',
#       data.query('Gone == 0')['MonthlyCharges'].describe())
# print()
# print('Statistics on monthly customer charges that have gone',
#       data.query('Gone == 1')['MonthlyCharges'].describe())
#
#
# # At first glance, you can see an approximately equal value for the minimum and maximum values of monthly charges.
#
# data.query('Gone == 0').boxplot('MonthlyCharges')
# plt.title("Stay", fontsize=16)
# plt.show()
#
# data.query('Gone == 1').boxplot('MonthlyCharges')
# plt.title("Lost", fontsize=16)
# plt.show()
#
# data.query('Gone == 0')['MonthlyCharges'].hist(bins = 20, label = 'Stay')
# data.query('Gone == 1')['MonthlyCharges'].hist(bins = 20, label = 'Lost')
#
# plt.xlabel('monthly charges')
# plt.ylabel('amount of users')
# plt.legend()
#
# plt.show()


# For both boxplot's, you can see that the distribution of the size of the monthly charges in the groups of left
# and remaining customers have visible differences:
# The size of the average monthly charges for those who stayed is strongly biased towards the minimum payment values (18.25)
# relative to those who left.
# This can be clearly seen at the border of the 25% quartile (for "Lost" this indicator is almost 2 times higher).
# The median monthly payment is also higher for those who left.
# It can be assumed that customers leave because of too high spending on company services.

# The percentage of users who used Internet services among those who left
data.query('Gone == 1 & InternetService >= 0.0').shape[0] / data.query('Gone == 1').shape[0]

# The percentage of users who used Internet services among those who stay
data.query('Gone == 0 & InternetService >= 0.0').shape[0] / data.query('Gone == 0').shape[0]


# Build a pie plot with
# percentage of Internet users among all remaining users.
# labels = ['use the service','dont use service']
# values = [data.query('Gone == 0 & InternetService >= 0.0').shape[0] / data.query('Gone == 0').shape[0], 1 - (data.query('Gone == 0 & InternetService >= 0.0').shape[0] / data.query('Gone == 0').shape[0])]
# colors = ['green','red']
# explode = [0,0.1]
# plt.title('The percentage of users who used Internet services among those who stay', fontsize = 14)
# plt.pie(values,labels=labels,colors=colors,explode=explode,shadow=True,autopct='%1.1f%%',startangle=180)
# plt.show()

# Build a pie plot with
# percentage of Internet users among all users who gone.
# labels = ['use the service','dont use service']
# values = [data.query('Gone == 1 & InternetService >= 0.0').shape[0] / data.query('Gone == 1').shape[0], 1 - (data.query('Gone == 1 & InternetService >= 0.0').shape[0] / data.query('Gone == 1').shape[0])]
# colors = ['yellow','grey']
# explode = [0,0.3]
# plt.title('The percentage of users who used Internet services among those who gone', fontsize = 14)
# plt.pie(values,labels=labels,colors=colors,explode=explode,shadow=True,autopct='%1.1f%%',startangle=180)
# plt.show()


# Diagrams clearly show that almost all the users who decided to leave were users of the Internet service
# And also that the percentage of Internet users among those who stayed below by about 20%



# Build a pie plot with percentage of phone communication among those who stay
# labels = ['use the service','dont use service']
# values = [data.query('Gone == 0 & MultipleLines >= 0.0').shape[0] / data.query('Gone == 0').shape[0], 1 - (data.query('Gone == 0 & MultipleLines >= 0.0').shape[0] / data.query('Gone == 0').shape[0])]
# colors = ['green','red']
# explode = [0,0.3]
# plt.title('The percentage of users who used phone communications among those who stay', fontsize = 14)
# plt.pie(values,labels=labels,colors=colors,explode=explode,shadow=True,autopct='%1.1f%%',startangle=180)
# plt.show()


# Build a pie plot with percentage of phone communications among those who gone
# labels = ['use the service','dont use service']
# values = [data.query('Gone == 1 & MultipleLines >= 0.0').shape[0] / data.query('Gone == 1').shape[0], 1 - (data.query('Gone == 1 & MultipleLines >= 0.0').shape[0] / data.query('Gone == 1').shape[0])]
# colors = ['yellow','grey']
# explode = [0,0.3]
# plt.title('The percentage of users who used phone communications among those who gone', fontsize = 14)
# plt.pie(values,labels=labels,colors=colors,explode=explode,shadow=True,autopct='%1.1f%%',startangle=180)
# plt.show()

# The percentage of telephone users in both groups are approximately equal.
# It can be assumed that people leave because of dissatisfaction with the quality of Internet services.

data.isna().sum()



# Normalize the features
data['MonthlyCharges']= (data['MonthlyCharges'] - data['MonthlyCharges'].mean()) / data['MonthlyCharges'].std()
data['TotalCharges']= (data['TotalCharges'] - data['TotalCharges'].mean()) / data['TotalCharges'].std()

# in the column with the duration of the contracts, replace Nan with 0
data['Difference'] = data['Difference'].fillna(0)
# Normalize feature in column "Difference"
data['Difference']= (data['Difference'] - data['Difference'].mean()) / data['Difference'].std()

# Replace Nan in columns with binary  meanings with 0

col = ['InternetService',
'OnlineSecurity',
'OnlineBackup',
'DeviceProtection',
'TechSupport',
'StreamingTV',
'StreamingMovies',
'MultipleLines']

data[col] = data[col].fillna(0)

# we transform some of the features using Ordinal Encoder
encoder = OrdinalEncoder()
columns = ['Type','PaymentMethod']
data[columns] = pd.DataFrame(encoder.fit_transform(data[columns]),
                            columns=data[columns].columns)

data['PaymentMethod'] = data['PaymentMethod'].astype('float64')

data = data.dropna()

# # 3. Selection optimal model

# Separetion general dataset at train/valid/test sets
# Let's see several models with stock hyperparameters to identify one (or two)
# that will show the best result according to the target metrics

# First of all let's appoint column with id as new indexes
data = data.set_index('customerID')

# Pick out target feature
features = data.drop(['Gone'], axis = 1)

target = data['Gone']
# display(data.corr())

features.describe()
cols = ['Type','PaymentMethod','MonthlyCharges','TotalCharges','Difference']
features[cols].corr()
features = features.drop(['TotalCharges'], axis = 1)



# Let's check value ratio of the target feature
class_frequency = data['Gone'].value_counts(normalize = (0,1))
print(class_frequency)



# we have a clear imbalance of the target feature
# write a function to increase the sample, which corresponds to class 1

def mix(features, target, num):
    features_1 = features[target == 1]
    features_0 = features[target == 0]
    target_1 = target[target == 1]
    target_0 = target[target == 0]

    features_new = pd.concat([features_0] + [features_1] * num)
    target_new = pd.concat([target_0] + [target_1] * num)
    features_new, target_new = shuffle(features_new, target_new, random_state=12345)
    return features_new, target_new

# Separate samples with features and target feature to train/valid/test
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size = 0.25, random_state = 12345)

features_train, features_test, target_train, target_test = train_test_split(features_train, target_train, test_size = 0.25, random_state = 12345)


# scale numerical features (column "MonthlyCharges")
numeric = ['MonthlyCharges']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])

features_test[numeric] = scaler.transform(features_test[numeric])


features_train, target_train = mix(features_train, target_train, 3)



print(target_train.value_counts(normalize = (0 ,1)))
print(target_valid.value_counts(normalize = (0 ,1)))
print(target_test.value_counts(normalize = (0 ,1)))


# Let's create a constant model that will assume that none of the clients left
model_constant = pd.Series([0 for x in range(len(target))])

print('Constant model accuracy: ', accuracy_score(target, model_constant))
print('Constant model precision: ', precision_score(target, model_constant))
print('Constant model recall: ', recall_score(target, model_constant))




# # The values of accuracy of the constant model will be taken as the baseline
# and future calculations of the models will be guided by it until we get a higher result.


# Let's start testing models with logistic regression

model_lr = LogisticRegression(random_state=12345, solver='liblinear', class_weight = 'balanced')
model_lr.fit(features_train, target_train)
predicted_valid = model_lr.predict(features_valid)

print('LogisticRegression model accuracy: ', accuracy_score(target_valid, predicted_valid))
probabilities_valid = model_lr.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('Validation set auc_roc', auc_roc)

print('Logistic Regression model precision on valid set: ', precision_score(target_valid, predicted_valid))
print('Logistic Regression model recall on valid set: ', recall_score(target_valid, predicted_valid))


# The accuracy score is already higher than that of the constant model.
# I'll also add roc_auc score on the validation set to each model to select the most suitable model later.


# Decision tree classifier model
model_tree = DecisionTreeClassifier(random_state = 12345, class_weight = 'balanced')
model_tree.fit(features_train, target_train)
predicted_valid = model_tree.predict(features_valid)

probabilities_valid = model_tree.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('DecisionTreeClassifier model accuracy: ', accuracy_score(target_valid, predicted_valid))
print('validation set auc_roc', auc_roc)


print('DecisionTreeClassifier model precision: ', precision_score(target_valid, predicted_valid))
print('DecisionTreeClassifier model recall: ', recall_score(target_valid, predicted_valid))


# The accuracy score shows the result not much higher than the random model with 50% prediction of the correct prediction.

# Random forest classifier model
model_rf = RandomForestClassifier(random_state = 12345)
model_rf.fit(features_train, target_train)
predicted_validr = model_rf.predict(features_valid)

probabilities_validr = model_rf.predict_proba(features_valid)
probabilities_one_validr = probabilities_validr[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_validr)
print('RandomForestClassifier model accuracy: ', accuracy_score(target_valid, predicted_validr))
print('validation set auc_roc', auc_roc)

print('RandomForestClassifier model precision: ', precision_score(target_valid, predicted_validr))
print('RandomForestClassifier model recall: ', recall_score(target_valid, predicted_validr))


# The random forest model has a higher accuracy score, but lower for the main quality metric compared to linear regression.


# Cat boost classifier model
model_cbr = CatBoostClassifier(iterations=5, random_state = 12345)
model_cbr.fit(features_train, target_train)
predict_cat = model_cbr.predict(features_valid)

probabilities_validc = model_cbr.predict_proba(features_valid)
probabilities_one_validc = probabilities_validc[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_validc)
print('CatBoostClassifier model accuracy: ', accuracy_score(target_valid, predict_cat))
print('validation set auc_roc', auc_roc)

print('CatBoostClassifier model precision: ', precision_score(target_valid, predict_cat))
print('CatBoostClassifier model recall: ', recall_score(target_valid, predict_cat))


# Catboost model performs well even with stock hyperparameters


# Extra trees classifier model
model_extra = ExtraTreesClassifier(random_state = 12345)
model_extra.fit(features_train, target_train)
pred_ex = model_extra.predict(features_valid)

probabilities_valide = model_extra.predict_proba(features_valid)
probabilities_one_valide = probabilities_valide[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valide)
print('ExtraTreesClassifier model accuracy : ', accuracy_score(target_valid, pred_ex))
print('validation set auc_roc', auc_roc)

print('ExtraTreesClassifier model precision: ', precision_score(target_valid, pred_ex))
print('ExtraTreesClassifier model recall: ', recall_score(target_valid, pred_ex))

# Not very good results, but I didn't really hope

# GradientBoostingClassifier
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=12345)
gbt.fit(features_train, target_train)

pred_gbt = gbt.predict(features_valid)

probabilities_validg = gbt.predict_proba(features_valid)
probabilities_one_validg = probabilities_validg[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_validg)
print('GradientBoostingClassifier model accuracy: ', accuracy_score(target_valid, pred_gbt))
print('Validation set auc_roc', auc_roc)

print('GradientBoostingClassifier model precision: ', precision_score(target_valid, pred_gbt))
print('GradientBoostingClassifier model recall: ', recall_score(target_valid, pred_gbt))

# XGBClassifier
model_x = xgboost.XGBClassifier(random_state = 12345)
model_x.fit(features_train, target_train)

pred_x = model_x.predict(features_valid)

probabilities_validx = model_x.predict_proba(features_valid)
probabilities_one_validx = probabilities_validx[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_validx)
print('XGBClassifier model accuracy: ', accuracy_score(target_valid, pred_x))
print('Validation set auc_roc', auc_roc)

print('XGBClassifier model precision: ', precision_score(target_valid, pred_x))
print('XGBClassifier model recall: ', recall_score(target_valid, pred_x))
# Both XGBClassifier and GradientBoostingClassifier showed the same accuracy score, but roc_auc was higher for the former.


# LGBMClassifier
clf = LGBMClassifier(random_state = 12345, class_weight = 'balanced')
clf.fit(features_train, target_train)
pred_clf = clf.predict(features_valid)

probabilities_valid1 = clf.predict_proba(features_valid)
probabilities_one_validl = probabilities_valid1[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_validl)
print('LGBMClassifier model accuracy: ', accuracy_score(target_valid, pred_clf))
print('Validation set auc_roc', auc_roc)

print('LGBMClassifier model precision: ', precision_score(target_valid, pred_clf))
print('LGBMClassifier model recall: ', recall_score(target_valid, pred_clf))


# # 4. Model training

# Selection of the optimal hyperparameters for the model and checking the target metric on the test set

# 1. GradientBoostingClassifier
gbt2 = ensemble.GradientBoostingClassifier(random_state = 12345)
gbt2.fit(features_train, target_train)

# Let's build a plot to select the list of optimal features for the model
importances = gbt2.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = features.columns

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))

d_first = 16
plt.figure(figsize=(10, 10))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first]);
# plt.show()

# You may notice that the "Difference" column has the greatest influence on model prediction.
# This was noticeable even at the time of looking at the carrelation of features.
# I would also like to note the features "InternetService" and "MonthlyCharges", which are also in the top 3.
# The rest of the features have much less influence on the model's prediction.

# Check the prediction of the model with the features that we selected earlier
best_features = indices[:10]
best_features_names = feature_names[best_features]
gbt2 = ensemble.GradientBoostingClassifier(random_state = 12345)
gbt2.fit(features_train[best_features_names], target_train)
pred_gbt2 = gbt2.predict(features_valid[best_features_names])
probabilities_valid = gbt2.predict_proba(features_valid[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('GradientBoostingClassifier model accuracy: ', accuracy_score(target_valid, pred_gbt2))
print('Validation set auc_roc ', auc_roc)

pred_gbt2 = gbt2.predict(features_test[best_features_names])
probabilities_valid = gbt2.predict_proba(features_test[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_test, probabilities_one_valid)
print('GradientBoostingClassifier model accuracy on the test set: ', accuracy_score(target_test, pred_gbt2))
print('Test set auc_roc ', auc_roc)

print('GradientBoostingClassifier model precision on test set: ', precision_score(target_test, pred_gbt2))
print('GradientBoostingClassifier model recall on test set: ', recall_score(target_test, pred_gbt2))


# Let's try to improve the numbers with a small selection of hiperparameters
params = {"max_depth" :[5 ,15 ,20 ,30], "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1]}
gbt2 = ensemble.GradientBoostingClassifier(random_state = 12345)

grid = GridSearchCV(gbt2, params, cv=3)
grid.fit(features_train[best_features_names], target_train)
grid.best_params_


# Let's use values witch we got from gridsearch in our model
gbt2 = ensemble.GradientBoostingClassifier(learning_rate= 0.1 ,max_depth= 15 , random_state = 12345)
gbt2.fit(features_train[best_features_names], target_train)
pred_gbt2 = gbt2.predict(features_valid[best_features_names])
probabilities_valid = gbt2.predict_proba(features_valid[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('GradientBoostingClassifier model accuracy: ', accuracy_score(target_valid, pred_gbt2))
print('Validation set auc_roc ', auc_roc)

pred_gbt2 = gbt2.predict(features_test[best_features_names])
probabilities_valid = gbt2.predict_proba(features_test[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_test, probabilities_one_valid)
print('GradientBoostingClassifier model accuracy on the test set: ', accuracy_score(target_test, pred_gbt2))
print('Test set auc_roc', auc_roc)

print('GradientBoostingClassifier model precision on test set: ', precision_score(target_test, pred_gbt2))
print('GradientBoostingClassifier model recall on test set: ', recall_score(target_test, pred_gbt2))

# XGBClassifier

# Also try the XGBClassifier model.
# First, let's select the features, and then the optimal values of some hyperparameters.
# Based on the results of comparing the two models, we'll choose the best one.

model_x = xgboost.XGBClassifier(random_state = 12345)
model_x.fit(features_train, target_train)
pred_model_x = model_x.predict(features_valid)
probabilities_valid = model_x.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('XGBClassifier model accuracy: ', accuracy_score(target_valid, pred_model_x))
print('Validation set auc_roc', auc_roc)

pred_model_x = model_x.predict(features_test)
probabilities_valid = model_x.predict_proba(features_test)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_test, probabilities_one_valid)
print('XGBClassifier model accuracy on the test set: ', accuracy_score(target_test, pred_model_x))
print('Test set auc_roc', auc_roc)

print('XGBClassifier model precision on the test set: ', precision_score(target_test, pred_model_x))
print('XGBClassifier model recall on the test set: ', recall_score(target_test, pred_model_x))

# Let's build a plot to select the list of optimal features for the model

importances = model_x.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = features.columns

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))

d_first = 16
plt.figure(figsize=(10, 10))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first]);


best_features = indices[:6]
best_features_names = feature_names[best_features]
model_x = xgboost.XGBClassifier(random_state = 12345)
model_x.fit(features_train[best_features_names], target_train)
pred_model_x = model_x.predict(features_valid[best_features_names])
probabilities_valid = model_x.predict_proba(features_valid[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('XGBClassifier model accuracy: ', accuracy_score(target_valid, pred_model_x))
print('Validation set auc_roc ', auc_roc)

pred_model_x = model_x.predict(features_test[best_features_names])
probabilities_valid = model_x.predict_proba(features_test[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_test, probabilities_one_valid)
print('XGBClassifier model accuracy on the test set: ', accuracy_score(target_test, pred_model_x))
print('Test set auc_roc ', auc_roc)

print('XGBClassifier model precision on the test set: ', precision_score(target_test, pred_model_x))
print('XGBClassifier model recall on the test set: ', recall_score(target_test, pred_model_x))



model_x = xgboost.XGBClassifier(random_state = 12345)

params = {
    'subsample': [0.6 ,0.7 ,0.8, 0.9 ,1.0],
    'n_estimators': [80 ,90 ,100 ,110 ,120]
}

grid = GridSearchCV(model_x, params, cv=3)
grid.fit(features_train[best_features_names], target_train)
grid.best_params_


model_x = xgboost.XGBClassifier(n_estimators = 100 ,subsample = 0.9 ,random_state = 12345)

model_x.fit(features_train[best_features_names], target_train)
pred_model_x = model_x.predict(features_valid[best_features_names])
probabilities_valid = model_x.predict_proba(features_valid[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('XGBClassifier model accuracy: ', accuracy_score(target_valid, pred_model_x))
print('Validation set auc_roc ', auc_roc)

pred_model_x = model_x.predict(features_test[best_features_names])
probabilities_valid = model_x.predict_proba(features_test[best_features_names])
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_test, probabilities_one_valid)
print('XGBClassifier model accuracy on test set: ', accuracy_score(target_test, pred_model_x))
print('Test set auc_roc ', auc_roc)

print('XGBClassifier model precision on the test set: ', precision_score(target_test, pred_model_x))
print('XGBClassifier model recall on the test set: ', recall_score(target_test, pred_model_x))



# The main metric (roc_auc) reached 92%.
# Secondary metric (accuracy) at 86%
# From the test sample, we were able to classify customers who leave us with a 71 percent probability.