import pandas as pd
try:
    import sklearn
    st.write("scikit-learn is installed!")
except ImportError:
    st.write("scikit-learn is not installed.")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import pickle

df=pd.read_excel("D://newmoli.xlsx")
# print(df.columns)

label_encoder = LabelEncoder()

df['job_title_encoded'] = label_encoder.fit_transform(df['job_title'])


df=df[(df['rate']>58) &(df['rate']<=80)]

X=df[['entity_id','certification','job_title_encoded']]
y=df['rate']

# print(df['rate'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['entity_id', 'certification','job_title_encoded'])
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=300, random_state=43,learning_rate=0.1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'The r2 score is {r2}')
print(f'{mse}')
print(f'{mae}')

print(df)



with open('pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

with open('pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

st.title('Rate Prediction model')


languages = ['Select a language'] + df['job_title'].unique().tolist()
selected_language = st.selectbox('Select language:',languages )
cities = ['Select a city'] + df['entity_name'].unique().tolist()
selected_entity = st.selectbox('Select city:', cities)
certifications = ['Select a certification'] + df['certification_name'].unique().tolist()
selected_certification = st.selectbox('Select certification:', certifications)


if st.button('Calculate Average Age'):
    if selected_certification!="Select a certification" and selected_entity!="Select certification" and selected_language!="Select a language":
        
        def find_entity_id(selected_entity):
            return int(df[df['entity_name'] == selected_entity]['entity_id'].unique().item())  
        def find_certification_id(selected_certification):
            return  int(df[df['certification_name'] == selected_certification]['certification'].unique().item())
        def find_job_title_encoded(selected_entity):
            return  int(df[df['job_title'] == selected_entity]['job_title_encoded'].unique().item())
        
        

        input_data = pd.DataFrame({
        'entity_id': find_entity_id(selected_entity),
        'certification': find_certification_id(selected_certification),
        'job_title_encoded': find_job_title_encoded(selected_language)
    },index=[0])
        
        predictions =loaded_pipeline.predict(input_data)
    
    # Display predictions
        st.write(f"{predictions}")
    
    else:
        st.write("Please select values from the dropdown.")
        
        
        




