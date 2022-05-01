import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as cv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as cv
import datetime
from datetime import date
import time
raw_data = pd.read_csv("https://raw.githubusercontent.com/adityajoe/KSEB_load_demand/main/raw_data.csv")
demand_data = pd.read_csv("https://raw.githubusercontent.com/adityajoe/KSEB_load_demand/main/tvm%20experiment%20data.csv")
rainfall_missing = demand_data["Demand in MW"][pd.isna(demand_data[ "Rainfall in mm"])].count()
humidity_missing = demand_data["Demand in MW"][pd.isna(demand_data[ "Relative Humidity in %"])].count()
temperature_missing = demand_data["Demand in MW"][pd.isna(demand_data[ "Temperature in Degree C"])].count()
radiation_missing = demand_data["Demand in MW"][pd.isna(demand_data[ "Radiation value in MJ/m^2"])].count()
final = demand_data.drop(columns = ["Relative Humidity in %", "Temperature in Degree C", "Radiation value in MJ/m^2" ])
date = pd.to_datetime(final["Date_Time"]).dt.date
time = pd.to_datetime(final["Date_Time"]).dt.time
final.insert(1,"Time", time, True)
final.insert(0,"Date",date,True)
final = final.drop(columns = ["Date_Time"])
final["Time"] = final["Time"].map(str)
final["Date"] = final["Date"].map(str)
time_array = np.array(final["Time"])
for i in range(len(time_array)):
  time_array[i] = time_array[i][0:2]
final["Time"] = time_array
st.title("KSEB Load Demand Prediction using Linear Regression")
st.write("""• My main motivation to do this project was to reduce the spot purchase of power in KSEB. 
         Spot purchase of power is much more expensive than purchasing power for a later date. 
         So if we can predict the demand a day before, we might be able to reduce the spot purchase in power.""") 
st.write("""• In this project I used historical demand dat
a collected from Kerala State Electricity Board and weather data collected from IMD and ANERT to create a Machine Learning Model
which could predict the load demand for the next day/hour given the required data.""")
st.write("""•	After collating the data from various sources, I performed simple feature engineering and exploratory data analysis.
         I found the demand to follow different patterns on weekdays and weekends as well as on holidays.
         Demand also varied according to the time of the day. 
         I also added new features like Hourly average of previous weeks, 
         Day of the week etc.These features made the model more sensible to less prone to outliers. I also performed standardization.""")
st.write("""•	After the initial data preprocessing, I decided to use Linear regression to create the model since it offers
high interpretability and good results for linearly distributed data""" )
st.subheader("Raw Data Collected")
st.dataframe(raw_data)
st.write("1. Missing values in rainfall - ", rainfall_missing)
st.write("2. Missing values in Relative Humidity - ", humidity_missing)
st.write("3. Missing values in Temperature - ", temperature_missing)
st.write("4. Missing values in Radiation value - ", radiation_missing)
st.write("* From this data, I removed the features that had a lot of missing values and added new features that I thought would bring value to the model. Since Temperature and Relative Humidity have a large proportion of missing values, I dropped them so that my model is not trained on garbage data.")
st.subheader("Processed and Cleaned Data")
st.dataframe(final)
st.header("Exploratory Data Analysis")
st.subheader("Variation of Demand on holidays and non holidays")
newplot = sns.FacetGrid(final, hue="Kerala Holidays", size=10);
newplot.map(sns.distplot, "Demand in MW");
newplot.add_legend();
st.pyplot(newplot)
st.write("* The average demand on a holiday is", final["Demand in MW"][final["Kerala Holidays"] == True].mean())
st.write("* The average demand on a non -holiday is", final["Demand in MW"][final["Kerala Holidays"] == False].mean())
st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader(" Variation of Demand at different times of the day")
ax = sns.boxplot(x='Time',y='Demand in MW', data=final)
st.pyplot()
st.caption("* We can see here that time of the day has a big significance on demand during that time.From 7:00 pm to 12:00 pm, we can see an increased demand when compared to other times of the day.")
st.subheader("Variation of Demand on different days of the week")
ax = sns.violinplot(x='Day of the week',y='Demand in MW', data=final)
st.pyplot()
st.caption("* We can see here that on Sundays, we generally see a lower demand than other days.")
X = final.drop(["Date", "Demand in MW"], axis = 1)
Y = final["Demand in MW"]
imp = IterativeImputer(max_iter=100, random_state=0,min_value=0.0)
X = imp.fit_transform(X)
st.subheader("Splitting into training and testing data...")
X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size = 0.33, random_state = 5)
st.write("* Shape of X_train is ", X_train.shape)
st.write("* Shape of X_test is ", X_test.shape)
st.write("* Shape of Y_train is ", Y_train.shape)
st.write("* Shape of Y_test is ", Y_test.shape)
lm = LinearRegression()
lm.fit(X_train, Y_train)
st.header("Predicting Demand using Linear Regression")
coefficients  = lm.coef_
features = ["Time", "Rainfall in mm", "Kerala Holidays", "Day of the week", "Yesterday's Demand in the same hour", "Hourly Average of the previous week"]
Y_pred = lm.predict(X_test)
ax = sns.scatterplot(x = Y_test, y = Y_pred)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
st.pyplot()
st.write("* We can observe an almost 45 degree line (with some outliers), so we can conclude that our model is almost predicting the demand accurately.")
st.caption("Machine Learning is never perfect, we can't predict the exact future demand for each data point we may receive but we can have 85-90 percent accuracy")
st.subheader(" Accuracy Measurement ---> CDF of Errors")
delta_y = abs(Y_test - Y_pred);
sns.set_style('whitegrid')
ax = sns.ecdfplot(data = delta_y)
st.pyplot()
st.caption("This is the Cumulative Density Function curve, In our curve, CDF value at any point tells us the percentage of data points that have predicted errors below the corresponding demand value")
st.subheader("Conclusions")
st.write("1. We can see that 93 percent of the error deviation is lower than 300 MW")
e_value = mean_squared_error(Y_test, Y_pred, squared = False)
print(e_value)
st.write("2. The RMSE value is ", e_value)
st.subheader("The coefficients of different features are as follows")
for i in range(len(coefficients)):
    st.write("* ", features[i],"--->", coefficients[i])
st.write("The intercept term is  --->", lm.intercept_)
st.caption("""PS. This brings us to the conclusions that on the days we have rainfall or on holidays, the demand will be lower than usual. 
We can also observe that the parameters "Hourly Average of the previous week" and "Yesterday's
 Demand" affect our output the most.""")
st.subheader("Deployment of the Model")
st.write("""* Finally, after all the data analysis, visualization and validation
         of my model on test dataset,I have deployed it.
         However, since I only had the data from 2017-2019, 
         predictions can be made on that time span only. 
         Kerala State Electricity Board stores the demand data hour wise for each day,
         so this model can definitely be used if trained with recent data.""")
with st.form("input values"):
    st.write("Enter your input")
    Date = st.date_input("Enter a date between 1/1/2017 to 1/1/2020 ",
                         datetime.date(2019, 7, 6), max_value= datetime.date(2020, 1, 1), min_value= datetime.date(2017,1,1))
    st.text("")
    st.text("")
    Holiday = st.checkbox("Tick the check box if it is a Holiday")
    if Holiday:
        Holiday = 1
    else:
        Holiday = 0
    st.text("")
    st.text("")
    rainfall = st.number_input("Enter the amount of rainfall in mm on the present day", step= 1)
    st.text("")
    st.text("")
    hour = st.slider("Enter the hour for which you want to predict the demand", 0,23)
    if len(str(hour)) == 1:
        hour = "0" + str(hour)
    else:
        hour = str(hour)
    curr_date = str(Date)
    previous_Date = Date - datetime.timedelta(days=1)
    previous_Date = str(previous_Date)
    st.text("")
    st.text("")
    day_week = st.number_input("Enter the Day of the Week- 1---> Sunday", step= 1, min_value= 1, max_value= 7)
    submitted = st.form_submit_button("Predict Demand")
yesterday = final["Demand in MW"][(final["Date"] == previous_Date) & (final["Time"] == hour)].sum()
hourly_prev_week = 0
count = 0
for i in range(1,8,1):
    count = count + 1
    previous_Date = Date - datetime.timedelta(days=i)
    previous_Date = str(previous_Date)
    hourly_prev_week = hourly_prev_week + final["Demand in MW"][(final["Date"] == previous_Date) & (final["Time"] == hour)].sum()
hourly_prev_week = hourly_prev_week/7
hour = int(hour)
input_array = np.array([hour, rainfall, Holiday, day_week, yesterday, hourly_prev_week])
output = lm.intercept_ + np.array(coefficients).dot(input_array)
st.text("")
st.text("")
if submitted:
    st.write("The predicted demand for the date {} and hour {} is {}".format(Date, hour, output))
st.caption("You can contact me at adityajoethomas@gmail.com")

