import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix




#-----Load dataset-----

df = pd.read_csv("train.csv")
test_df = pd.read_csv("sample.csv")
tst_id = test_df['id']

X_test_feature = test_df.drop('id',axis=1)
# -----Basic EDA-----

print(df.head())



print(df.isnull().sum())


print(df['Heart Disease'].value_counts(normalize=True)*100)

# df['Heart Disease'].value_counts().plot(kind='bar')
# plt.xlabel("class")
# plt.ylabel("count")
# plt.title("Target distribution")
# plt.show()
# print(df.sample(5))

num_cols = df.select_dtypes(include=['int64','float64']).columns

# skew_values = df[num_cols].skew()

# print("skew values -------",skew_values)

# df['Cholesterol'].hist()
# plt.title("Cholesterol distribution")
# plt.show()
# print("sexing",df['Sex'].value_counts(normalize=True)*100)
# df['Sex'].value_counts().plot(kind="bar")
# plt.xlabel("SEX")
# plt.ylabel("Count")
# plt.title("Sex distribution")
# plt.show()
# plt.hist(df['BP'],bins=20)
# plt.xlabel("BP")
# plt.ylabel("Frequency")
# plt.title("BP distribution")
# # plt.show()

# print("Before skew",df['BP'].skew())


# plt.boxplot(df['BP'])
# plt.show()


# plt.figure(figsize=(8,6))
# sns.heatmap(df[num_cols].corr(),annot=True)
# plt.show()

# sns.histplot(data=df,x='BP',hue='Heart Disease', kde=True)
# plt.show()

# Q1 = np.percentile(df,25)
# Q3 = np.percentile(df,75)

# IQR = Q3-Q1

# lower_bound = Q1-1.5*IQR
# upper_bound = Q3+1.5*IQR
# outliers = df[(df<lower_bound)|(df>upper_bound)]

# print("outliers deceted",outliers)

# # ---------------- Feature & Target ----------------

X = df.drop("Heart Disease",axis=1)
y = df['Heart Disease']


cols = ['BP','Max HR','Cholesterol']
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
# print("numerical columns",num_cols)
# print("catergorial columns",cat_cols)

#------Data Handling------

#------Skew Handling------

log_transformer = FunctionTransformer(np.log1p,validate=False)

#------No missing values------

# print(df.isnull().sum())

#------Categorical Encoding------

cols = ['BP','Max HR','Cholesterol']

cat_cols = ['Sex','FBS over 120','Thallium','Slope of ST']

num_col = ['BP','Max HR','Cholesterol']



mapping = {'Presence':1,'Absence':0}

y_train = y_train.map(mapping)
y_val = y_val.map(mapping)


X_train = X_train.drop("id",axis=1)
X_val = X_val.drop("id",axis=1)


preprocessor = ColumnTransformer(transformers=[('num',Pipeline([('log',log_transformer),('scaler',StandardScaler())]),num_col),('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols)])


#--------Model Creation--------

model = Pipeline(steps=[('preprocess',preprocessor),
                        ('classifier',RandomForestClassifier(n_estimators=200,max_depth=10,n_jobs=-1,random_state=42))])

model.fit(X_train,y_train)
y_pre = model.predict(X_val)

print("Accuracy", accuracy_score(y_val,y_pre))
# print("Confusion martirx for check", confusion_matrix(y_val,y_pre))
# print("classification report", classification_report(y_val,y_pre))

submission = pd.DataFrame({
    'id': tst_id,
    'Heart disease': model.predict(X_test_feature)})
print(submission.head())

submission.to_csv("submission.csv",index=False)