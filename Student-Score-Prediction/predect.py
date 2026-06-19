import pickle
import pandas as pd

with open('pipe.pkl', 'rb') as f:
    model = pickle.load(f)

student = pd.DataFrame({
    'gender': ['male'],
    'race/ethnicity': ['group A'],
    'test preparation course': ['completed'],
    'reading score': [71],
    'writing score': [65]
})

pre = model.predict(student)

print(pre[0])