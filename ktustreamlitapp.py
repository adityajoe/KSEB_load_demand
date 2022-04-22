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
demand_data = pd.read_csv("https://raw.githubusercontent.com/adityajoe/KSEB_load_demand/main/tvm%20experiment%20data.csv")
final = demand_data.drop(columns = ["Relative Humidity in %", "Temperature in Degree C", "Radiation value in MJ/m^2" ])
date = pd.to_datetime(final["Date_Time"]).dt.date
time = pd.to_datetime(final["Date_Time"]).dt.time
final.insert(1,"Time", time, True)
final.insert(0,"Date",date,True)
final = final.drop(columns = ["Date_Time"])
final["Time"] = final["Time"].map(str)
time_array = np.array(final["Time"])
for i in range(len(time_array)):
  time_array[i] = time_array[i][0:2]
final["Time"] = time_array
st.title("KSEB Load Demand Prediction using Linear Regression")
st.write("• In this project I used historical demand data collected from Kerala State Electricity Board and weather data collected from IMD and ANERT. My goal was to create a Machine Learning Model which could predict the load demand for the next day/hour given the required data.")
st.write("•	After collating the data from various sources, I performed simple feature engineering and exploratory data analysis. I found the demand to follow different patterns on weekdays and weekends as well as on holidays.Demand also varied according to the time of the day. I also added new features like Hourly average of previous weeks.These features made our model more sensible to less prone to outliers. I also performed standardization.")
st.write("•	After the initial data preprocessing, I used Linear regression to create the model. My model gave an accuracy of 95%")
st.dataframe(final)
st.header("Variation of Demand on holidays and non holidays")
newplot = sns.FacetGrid(final, hue="Kerala Holidays", size=10);
newplot.map(sns.distplot, "Demand in MW");
newplot.add_legend();
st.pyplot(newplot)
st.write("The average demand on a holiday is", final["Demand in MW"][final["Kerala Holidays"] == True].mean())
st.write("The average demand on a non -holiday is", final["Demand in MW"][final["Kerala Holidays"] == False].mean())
#figure1 = plt.figure(figsize=(20,8))
st.set_option('deprecation.showPyplotGlobalUse', False)
#plt.subplots(figsize=(20,8))
st.header(" Variation of Demand at different times of the day")
ax = sns.boxplot(x='Time',y='Demand in MW', data=final)
st.pyplot()
st.caption("We can see here that time of the day has a big significance on demand during that time")
st.header("Variation of Demand on different days of the week")
ax = sns.boxplot(x='Day of the week',y='Demand in MW', data=final)
st.pyplot()
st.caption("We can see here that on Sundays(day1) we generally see lower demand than other days.")
X = final.drop(["Date", "Demand in MW"], axis = 1)
Y = final["Demand in MW"]
imp = IterativeImputer(max_iter=100, random_state=0,min_value=0.0)
X = imp.fit_transform(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
new_demand_data = scaler.fit_transform(X, Y)
st.header("Splitting into training and testing data...")
X_train, X_test, Y_train, Y_test = cv.train_test_split(new_demand_data, Y, test_size = 0.33, random_state = 5)
st.write("Shape of X_train is ", X_train.shape)
st.write("Shape of X_test is ", X_test.shape)
st.write("Shape of Y_train is ", Y_train.shape)
st.write("Shape of Y_test is ", Y_test.shape)

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
st.write("We can observe an almost 45 degree line (with some outliers), so we can conclude that our model is almost predicting the demand accurately.")
st.caption("Machine Learning is never perfect, we can't predict accurately for each data point we have")
st.subheader(" Accuracy Measurement ---> CDF of Errors")
delta_y = abs(Y_test - Y_pred);
sns.set_style('whitegrid')
ax = sns.ecdfplot(data = delta_y)
st.pyplot()
st.write("1. We can see that 93 percent of the error deviation is lower than 300 MW")
e_value = mean_squared_error(Y_test, Y_pred, squared = False)
print(e_value)
st.write("2. The RMSE value is ", e_value)
st.subheader("The coefficients of different features are as follows")
for i in range(len(coefficients)):
    st.write(features[i],"--->", coefficients[i])
st.write("The intercept term is  --->", lm.intercept_)
st.caption("""PS. This brings us to the conclusions that on the days we have rainfall or are holidays, the demand will be lower. 
We can also observe that the parameters "Hourly Average of the previous week" and "Yesterday's Demand" affect our output the most.""")
