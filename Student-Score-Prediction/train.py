import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv('StudentsPerformance.csv')

df = df.drop(['parental level of education', 'lunch'], axis=1)

x = df.drop('math score', axis=1)
y = df['math score']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

trf = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first'), ['gender', 'race/ethnicity', 'test preparation course']),
    ('ss', StandardScaler(), ['reading score', 'writing score'])
])

pipe = Pipeline([
    ('transform', trf),
    ('model', LinearRegression())
])

pipe.fit(x_train, y_train)
y_pre = pipe.predict(x_test)

print(r2_score(y_test, y_pre))

#save model
print('before saving')

with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Model Saved Successfully")