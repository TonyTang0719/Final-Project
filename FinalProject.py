import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from streamlit.state.session_state import SessionState
st.title("Final_Project")
st.markdown("JIACHEN TANG ID:83049912")
st.markdown("Here (https://www.kaggle.com/tmdb/tmdb-movie-metadata) is the link to the dataset")

df = pd.read_csv("TMDBMovies.csv", na_values = " ")
df
st.write("Based on the dataset, My project is focusing on two questions:")
st.write("1. As time goes by, what is the variation trend for the runtime of the movies?")
st.write("2. What is the most popular genre of movies? ")

#First let's clean the data and eliminate the abnormal values and unrelated columns
df.info()
df.describe()
#check the statistical information of the dataset
#Refrence: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html?highlight=describe#pandas.DataFrame.describe
df=df[df['budget']>0]
df=df[df['revenue']>0]
df=df[df['runtime']>0]
#Delete the meaningless data(when the budget, revenue,runtime is zero)
df.describe()
#Based on the topic, delete unrelated columns in this dataset
df1=df.drop(["id","homepage","keywords","overview","production_companies","production_countries","tagline"],axis=1)
#Add a new column to show the released year of each movie
df1["release_year"]= pd.to_datetime(df1["release_date"])
df1["release_year"] = df1["release_year"].apply(lambda s: str(str(s)[0:4]))
df1["release_year"] = pd.to_numeric(df1['release_year']) #change the "released year" from "string" to "numeric", which is important to data analysis and make charts
df1
st.write("Above is the processed version of dataset. Specific approaches and explanation is shown on the code.")

### QUESTION 1
st.header("QUESTION 1")
st.write("For Question1, we need to calculate the mean runtime of all movies in each year")
df1_runtime=df1.groupby(['release_year'],as_index=False)["runtime"].mean() 
#Refrence: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
df1_runtime.sort_values('runtime',ascending=True)
df1_runtime

#Make chart
chart1=alt.Chart(df1_runtime).mark_circle().encode(
    x = alt.X('release_year', scale=alt.Scale(zero=False)),
    y = alt.Y('runtime', scale=alt.Scale(zero=False)),
    tooltip=["release_year","runtime"],
   
).properties(
    width = 800,
    height = 800
)

st.altair_chart(chart1)
st.write("From the chart we can see the mean runtime of the movie before 1990 is unstable and in a big fluctuation wave. That might because the number of movies before 1990 is not enough, there exists a contingency between the data.")
df1["release_year"].value_counts()
df1["release_year"] = pd.to_numeric(df1['release_year'])
df2=df1[df1['release_year']>1990]
df2["release_year"].value_counts()
df2_runtime=df2.groupby(['release_year'],as_index=False)["runtime"].mean() 
df2_runtime.sort_values('runtime',ascending=True)
#This time we only select the movies released after 1990 and repeated the same process

chart2= alt.Chart(df2_runtime).mark_circle().encode(
    x = alt.X('release_year', scale=alt.Scale(zero=False)),
    y = alt.Y('runtime', scale=alt.Scale(zero=False)),
    tooltip=["release_year","runtime"],
   
).properties(
    width = 800,
    height = 800
)

st.altair_chart(chart2)
st.write("This time we only select the movies released after 1990 and make the new plot.")

###QUESTION 2
st.header("QUESTION 2")
st.write("For question 2, the most two obvious ways to reflect the most popular genres of the movies are: the total number of movies in a genre and the mean popularity of movies in a genre")
# Since most movies have multiple genres at the same time, we need to clean the data first
import ast
#Refrence：https://www.geeksforgeeks.org/python-convert-string-dictionary-to-dictionary/
my_list=[]
mv_genres=[]
for i in df2["genres"]:
    my_list=[]
    for j in ast.literal_eval(i):
        my_list.append(j["name"])
    mv_genres.append(my_list)
st.write("List the tpye of genres in each movie:")
mv_genres
#Calculate the the total number of movies in different genres
num_of_genres = {}
for i in mv_genres:
    for a in i:
        if a in num_of_genres:
            num_of_genres[a] +=1
        else:
            num_of_genres[a] = 1
st.write("below is the total number of movies in each genre")
st.write(num_of_genres)

#And we make the pie chart of it.
from matplotlib import pyplot as plt
lables = 'Action', 'Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War', 'Music', 'Documentary', 'Foreign'
data=[918, 661, 342,  431,  521,  1441,  935, 188,  365,  57, 1110,  574,  332,  265,  145,  120,  111,  38,  5]
size=[data[i]/8559 for i in range(19)]
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(size, labels=lables, autopct='%1.1f%%',
        startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
st.write("We conclude from the pie chart that 'drama' is the most common genres of movies, which contains 16.8% of the movies; then it comes to 'comedy'(13.0%）, 'thriller'(10.9%) and 'action'(10.7%)" )
#Refrences:https://discuss.streamlit.io/t/how-to-draw-pie-chart-with-matplotlib-pyplot/13967/2


#Find the mean popularity of each genres
df3 = df2.copy()
df3["genres"] = mv_genres
dic = {}
genres=['Action', 'Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War', 'Music', 'Documentary', 'Foreign']
for j in genres:
    total = 0
    count = 0
    for index, value in enumerate(df3["genres"]):
        if j in value:
            total += df3.iloc[index]['popularity']
            count += 1
    dic[j] = total/count
st.write(dic)
st.write("Above is the mean popularity of each movies, we can see 'Adventure' has the highest popularity; Then it comes to 'Science Fiction', 'Animation' and 'Fantasy'")

##Conclusion
st.header("Conclusion")
st.write("For question 1, we conclude the runtime of movies before 1990s did not have an obvious trending because the number of movies in total is small; From 1990 to 2010, the runtime of the movies gradually decrease and increase after 2010 ")
st.write("For question 2, we conclude the top 3 common genres in movies are 'drama','comedy' and 'thriller', but 'Adventure','Science Friction' and 'Animation' movies have the highest popularity")

#Check to see if the data is overfitting or not
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
df = pd.read_csv("TMDBmovies.csv", na_values = " ")
df = df[df.notna().all(axis=1)].copy()
df = df.drop(["id"],axis=1)
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
scaler = StandardScaler()
scaler.fit(df[numeric_cols])
df[numeric_cols] = scaler.transform(df[numeric_cols])
X_train = df[numeric_cols].drop("popularity", axis=1)
y_train = df["popularity"]
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (5,)),
        keras.layers.Dense(5, activation="sigmoid"),
        keras.layers.Dense(5, activation="sigmoid"),
        keras.layers.Dense(1,activation="linear")
    ]
)

model.compile(
    loss="mean_squared_error", 
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

history = model.fit(X_train,y_train,epochs=100, validation_split = 0.2, verbose=False)
fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
st.pyplot(fig)
st.write("It's not overfitting, since the loss of train line and validation line decrease and gradually close to a constant with the increase of the epoch")

