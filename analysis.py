import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [4, 4]
st.title('Data Analysis project')



df_accident = pd.read_csv("./exams.csv")




gender = {'male': 1,'female': 2}
df_accident.gender = [gender[item] for item in df_accident.gender]

race= {'group A': 1,'group B': 2,'group C': 3,'group D': 4, 'group E': 5}
df_accident['race/ethnicity']= [race[item] for item in df_accident['race/ethnicity']]

parent_education= {'some high school': 1,'high school':2,'some college': 3,"associate's degree": 4,"bachelor's degree": 5, "master's degree": 6}
df_accident['parental level of education']= [parent_education[item] for item in df_accident['parental level of education']]

lunch= {'standard': 0,'free/reduced': 1}
df_accident['lunch']= [lunch[item] for item in df_accident['lunch']]

test_preperation= {'none': 0,'completed': 1}
df_accident['test preparation course']= [test_preperation[item] for item in df_accident['test preparation course']]

st.caption('The values gender is replaced by male:1, female: 2')
st.caption('race/ethnicity=> group A:1, group b:2, group c:3, group D:4,')
st.caption('parental level of education=> some high school: 1,high school:2,some college: 3,associate"s degree: 4,bachelor"s degree: 5, master"s degree: 6')

df_accident.dropna()
add_selectbox = st.sidebar.selectbox(
    "Dataset Overview",
    ("head", "tail", "Describe","Columns")
)
if 'head' in add_selectbox: # If user selects Email  do 👇
        head_of_df=df_accident.head()
        st.write(head_of_df)
elif 'tail' in add_selectbox: # If user selects Email  do 👇
        head_of_df=df_accident.tail()
        st.write(head_of_df)

elif 'Describe' in add_selectbox: # If user selects Email  do 👇
        head_of_df=df_accident.describe()
        st.write(head_of_df)
elif 'Columns' in add_selectbox: # If user selects Email  do 👇
        st.write(df_accident.columns)

agree = st.checkbox('Show correlation matrix')

if agree:
     fig, ax = plt.subplots()
     sns.heatmap(df_accident.corr(), ax=ax)
     st.write(fig)    

distribution_graph = st.checkbox('Show ditstribution graph')

if distribution_graph:
     
  
 distr_graph_data = ["gender","race/ethnicity","parental level of education","lunch","test preparation course","math score","reading score","writing score"]
 add_selectbox = st.selectbox(
    "Select Column",
    (distr_graph_data)
 )  
 st.write(add_selectbox)
 arr =  add_selectbox
 fig=plt.figure(figsize=(7, 3) )
 fig, ax = plt.subplots()
 ax.hist(df_accident[add_selectbox], bins=100)
 
 st.pyplot(fig)

     


