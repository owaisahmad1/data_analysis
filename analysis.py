import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.rcParams['figure.figsize'] = [4, 4]
st.title('Data Analysis project')



df_exam = pd.read_csv("./exams.csv")




gender = {'male': 1,'female': 2}
df_exam.gender = [gender[item] for item in df_exam.gender]

race= {'group A': 1,'group B': 2,'group C': 3,'group D': 4, 'group E': 5}
df_exam['race/ethnicity']= [race[item] for item in df_exam['race/ethnicity']]

parent_education= {'some high school': 1,'high school':2,'some college': 3,"associate's degree": 4,"bachelor's degree": 5, "master's degree": 6}
df_exam['parental level of education']= [parent_education[item] for item in df_exam['parental level of education']]

lunch= {'standard': 0,'free/reduced': 1}
df_exam['lunch']= [lunch[item] for item in df_exam['lunch']]

test_preperation= {'none': 0,'completed': 1}
df_exam['test preparation course']= [test_preperation[item] for item in df_exam['test preparation course']]
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Gender ')
    st.write("male:1")
    st.write(" female: 2")

with col2:
    st.subheader('race/ethnicity')
    st.write("group A: 1")
    st.write(" group b: 2")
    st.write(" group c: 3")
    st.write(" group D: 4")

with col3:
    st.subheader("parental level of education")
    st.write("some high school: 1")
    st.write(" high school:2")
    st.write("some college: 3")
    st.write(" associate's degree: 4")
    st.write(" bachelor's degree: 5")
    st.write(" master's degree: 6")


df_exam.dropna()
add_selectbox = st.sidebar.selectbox(
    "Dataset Overview",
    ("head", "tail", "Describe","Columns")
)
if 'head' in add_selectbox: # If user selects Email  do 👇
        head_of_df=df_exam.head()
        st.write(head_of_df)
elif 'tail' in add_selectbox: # If user selects Email  do 👇
        head_of_df=df_exam.tail()
        st.write(head_of_df)

elif 'Describe' in add_selectbox: # If user selects Email  do 👇
        head_of_df=df_exam.describe()
        st.write(head_of_df)
elif 'Columns' in add_selectbox: # If user selects Email  do 👇
        st.write(df_exam.columns)

agree = st.checkbox('Show correlation matrix')

if agree:
     fig, ax = plt.subplots()
     sns.heatmap(df_exam.corr(), ax=ax)
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
 ax.hist(df_exam[add_selectbox], bins=100)
 
 st.pyplot(fig)

     
grading_count = st.checkbox('show grading count')

if grading_count:
     
  
 distr_graph_data = ["math score","reading score","writing score"]
 add_selectbox = st.selectbox(
    "Overall Score distribution",
    (distr_graph_data)
 )  
 st.write(add_selectbox)
 arr =  df_exam[add_selectbox]
 def grading(arr):
    if arr >= 70:
        return "High"
    elif arr<= 70 and arr >= 50:
        return "Medium"
    else:
        return "Low"
    

 df_exam["grades"] = df_exam[add_selectbox].apply(grading)
 grades_count=np.unique(df_exam["grades"],return_counts=True)

 fig=plt.figure(figsize=(3, 3) )
 
 fig, ax = plt.subplots()
 ax.bar(grades_count[0],grades_count[1])

 
 st.pyplot(fig)
 
 st.subheader('Impact on Score')
 col_1_data= ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
 col_2_data= ["math score","reading score","writing score"]
 col_1= st.selectbox(
    "Select Column",
    (col_1_data)
 ) 
 
 col_2 = st.selectbox(
         "Select",
         (col_2_data)
)

 score=df_exam[col_2]
 def grade(score):
    if score >= 70:
        return "High"
    elif score<= 70 and score>= 50:
        return "Medium"
    else:
         return "Low"
 df_exam["grade"] = df_exam[col_2].apply(grade)

 df_exam_grouped = df_exam["grade"].groupby([df_exam["grade"],df_exam[col_1]]).count().unstack()


 df_exam_grouped
 df_exam_grouped.plot.bar()
 st.pyplot()

st.title("Mobile price prediction model using random forest")
df = pd.read_csv("./train.csv")
st.write(df.head())

correlation = df.corr()

figure_heatmap, ax = plt.subplots()
sns.heatmap(correlation, ax=ax)
st.write(figure_heatmap)
selected_columns = ["battery_power","fc","int_memory","mobile_wt","px_height","px_width","ram","sc_h","sc_w","talk_time","four_g"]


def battery_power(x):
    if x >= 1401:
        return 0
    elif x<=1400 and x >= 1001:
        return 1
    elif x<=237750 and x >= 202880:
        return 2
    else:
        return 3
    

df["battery_range"] = df["battery_power"].apply(battery_power)

df["px_dp"]=df["px_height"]* df["px_width"]

def bin_pxDP(x):
    if x <  263200:
        return 0
    elif x < 601359 and x > 263200:
        return 1
    elif x < 1359027 and x > 601359:
        return 2
    elif x > 601359:
        return 3

df["bin_pxDP"] = df.px_dp.apply(bin_pxDP)
train_col = ["battery_range","fc","int_memory","mobile_wt","bin_pxDP","ram","sc_h","sc_w","talk_time","four_g"]

from sklearn.model_selection import train_test_split

train_x,test_x, train_y, test_y = train_test_split(df[train_col], df["price_range"], test_size=0.33, random_state=42)

callasifier_performance = {}

from sklearn.ensemble import RandomForestClassifier


    
random_clf = RandomForestClassifier(144)
random_model = random_clf.fit(train_x, train_y)
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix



predictions = random_model.predict(test_x)


plot_confusion_matrix(random_clf,test_x, test_y)
st.pyplot()


acc_s = accuracy_score(predictions,test_y)
callasifier_performance["Random Forest"] = acc_s
st.write("prediction with random forest")
st.write("Accuracy: ",acc_s,"\nLoss: ",1-acc_s)


st.subheader(" Input your Mobile Specifications ")

battry_range = st.slider('battery range', 0, 3)
fc = st.slider('Front Camera', 0, 19)
int_memory = st.slider('Memory', 2, 64)
mobile_wt = st.slider('Mobile weight', 80, 200)
bin_pxDP =st.slider('Screen pixel', 0, 3)
ram =st.slider('Ram', 256, 3998)
sc_h =st.slider('screen height', 5, 19)
sc_w =st.slider('screen width', 0, 18)
talk_time = st.slider('Talk time', 2, 18)
four_g= st.slider('4 G', 0, 1)
train_col = ["battery_range","fc","int_memory","mobile_wt","bin_pxDP","ram","sc_h","sc_w","talk_time","four_g"]
predict_data = [[battry_range,fc,int_memory,mobile_wt,bin_pxDP,ram,sc_h,sc_w,talk_time,four_g]]

prediction_of_data= random_model.predict(predict_data)
st.write(prediction_of_data)


