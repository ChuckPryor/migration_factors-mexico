import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from datetime import date
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import numpy as np

lr = LinearRegression()
st.markdown("<h1 style='text-align: center; color: black;'>Estimating Migration to the United States from Mexico</h1>", unsafe_allow_html=True)
st.markdown('__*A prediction algorithm application predicting the number of immigrants seeking refuge in the United States given several variables.*__')
st.image("mximm.png", use_column_width=True)

st.write("Each year Mexico sees thousands of migrants fleeing the country. The migrants mainly attempt to immigrate to the United States however South America, the West Indies, and Europe are starting to see significant numbers of Central American Migrants.  Factors that could influence migration are food insecurity, violent crime, politics, domestic violence, jobs, access to medical care, and poverty.  This migration causes a massive immediate influx to countries that take them.  The United States seeks to understand what exactly drives migration and how best to serve the immigrants as well as help the countries they are coming from retain their population.  We have unpacked reasons why the United States remains the optimal destination for Mexican immigrants")
st.markdown("<h1 style='text-align: center; color: black;'>…</h1>", unsafe_allow_html=True)
df = pd.read_csv('df_full_mx_immigration.csv')
df1=df[['Date', 'Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change', 'Mx_crimes_per_100k',
       'mx_annual_per_change_crimes', 'mx_%pov_Under_US_$5.50_Per_Day',
       'yearly_change_in_mx_pov_rate', 'MEXICO_TOTAL_KILOTONS_Co2',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE',
       'Number_Migrant_origin_Mexico_Destination_US',
       'Number_Migrant_origin_World_Destination_US',
       'Mexico Murder_Per 100K Population', 'Mexico Murder_Annual % Change', 'us_border_spending_in_millions', 'kt of CO2 equivalent',
       'Annual % Change', 'UNAUTHORIZED_POP_ENTERED_US']]

#st.write('In this chart, you will see a linear relationship between the USGDP, the USGDP change rate, and the amount of immigrants coming from Mexico')

#custom_chart = alt.Chart(df1).mark_line().encode(x='US_GDP_Growth %', y='UNAUTHORIZED_POP_ENTERED_US').properties(width=900,height=500)

#st.altair_chart(custom_chart)

#sns.set_style("darkgrid")

st.sidebar.title("Operations on the Dataset")

#st.subheader("Checkbox")
w1 = st.sidebar.checkbox("show table", False)
plot= st.sidebar.checkbox("show plots", False)
plothist= st.sidebar.checkbox("show hist plots", False)
trainmodel= st.sidebar.checkbox("Train model", False)
dokfold= st.sidebar.checkbox("DO KFold", False)
distView=st.sidebar.checkbox("Dist View", False)
_3dplot=st.sidebar.checkbox("3D plots", False)
linechart=st.sidebar.checkbox("Linechart",False)
#st.write(w1)
st.sidebar.markdown("<h1 style='text-align: center; color: black;'>…</h1>", unsafe_allow_html=True)
st.sidebar.write('This project was conceptualized by [Helen Haile](https://www.linkedin.com/in/helenhaile/), [Rahel Mehreteab](https://www.linkedin.com/in/rahel-mehreteab-043802aa/), [Charles Pryor](https://www.linkedin.com/in/charlespryorjr/), [Natalie Scavuzzo](https://www.linkedin.com/in/nataliescavuzzo/), and [Fiona Suliman](https://www.linkedin.com/in/fionasuliman/).  You can find all source code, datasets, and our full use case on [github](https://github.com/ChuckPryor/migration_factors-mexico).')


@st.cache
def read_data():
    return pd.read_csv("df_full_mx_immigration.csv")[['Date','Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE', 'UNAUTHORIZED_POP_ENTERED_US']]

st.write("We narrowed down the reasons for the factors of immigration to the U.S. to such variables as Crime, Economics, and Environment. We gathered, cleaned, then merged over 20 datasets to get the following dataset to perform our regression.  We used the most recent 30 years to train our model")

df=read_data()

#st.write(df)
if w1:
    st.dataframe(df,width=2000,height=500)
if linechart:
	st.subheader("Line charts showing the change in feature values over time")
	options = ('Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE', 'UNAUTHORIZED_POP_ENTERED_US')
	sel_cols = st.selectbox("Select a column", options,1)
	st.write(sel_cols)
	line_fig = go.Scatter(y=df[sel_cols], x=df['Date'])

	st.plotly_chart([line_fig])
if plothist:
    st.subheader("Distributions of each column")
    options = ('Date','Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE', 'UNAUTHORIZED_POP_ENTERED_US')
    sel_cols = st.selectbox("select columns", options,1)
    st.write(sel_cols)
    #f=plt.figure()
    fig = go.Histogram(x=df[sel_cols],nbinsx=50)
    st.plotly_chart([fig])
    

#    plt.hist(df[sel_cols])
#    plt.xlabel(sel_cols)
#    plt.ylabel("sales")
#    plt.title(f"{sel_cols} vs Sales")
    #plt.show()	
#    st.plotly_chart(f)

if plot:
    st.subheader("correlation between UNAUTHORIZED_POP_ENTERED_US and all features")
    options = ('Date','Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE', 'UNAUTHORIZED_POP_ENTERED_US')
    w7 = st.selectbox("UNAUTHORIZED_POP_ENTERED_US", options,1)
    st.write(w7)
    f=plt.figure()
    plt.scatter(df[w7],df["UNAUTHORIZED_POP_ENTERED_US"])
    plt.xlabel(w7)
    plt.ylabel("UNAUTHORIZED_POP_ENTERED_US")
    plt.title(f"{w7} vs UNAUTHORIZED_POP_ENTERED_US")
    #plt.show()	
    st.plotly_chart(f)


if distView:
	st.subheader("Combined distribution viewer")
	st.write("See how the distributions overlay each other")
	# Add histogram data

	# Group data together
	hist_data = [df['Mex_GDP_Growth %'].values,df['Mex_Annual_GDP_Change'].values,df['US_GDP_Growth %'].values,df['US_Annual_GDP_Change'].values,df['MEXICO_MT_Co2_PER_CAPITA'].values,df['MEXICO_INFLATION_RATE_PERCENTAGE'].values,df['MEXICO_INFLATION_ANNUAL_CHANGE'].values]

	group_labels = ['Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE']

	# Create distplot with custom bin_size
	fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.5, 0.5, 0.5, .5,.5,.5,.5])

	# Plot!
	st.plotly_chart(fig)

if _3dplot:
	options = st.multiselect(
     'Enter columns to plot',('Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE'))
	st.write('You selected:', options)
	st.subheader("Features vs UNAUTHORIZED_POP_ENTERED_US")
	hist_data = [df['Mex_GDP_Growth %'].values,df['Mex_Annual_GDP_Change'].values,df['US_GDP_Growth %'].values,df['US_Annual_GDP_Change'].values,df['MEXICO_MT_Co2_PER_CAPITA'].values,df['MEXICO_INFLATION_RATE_PERCENTAGE'].values,df['MEXICO_INFLATION_ANNUAL_CHANGE'].values]

	#x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()
	trace1 = go.Scatter3d(
		x=hist_data[0],
		y=hist_data[1],
		z=df["UNAUTHORIZED_POP_ENTERED_US"].values,
		mode="markers",
		marker=dict(
			size=8,
			#color=df['sales'],  # set color to an array/list of desired values
			colorscale="Viridis",  # choose a colorscale
	#        opacity=0.,
		),
	)

	data = [trace1]
	layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
	fig = go.Figure(data=data, layout=layout)
	st.write(fig)



# trainmodel= st.checkbox("Train model", False)

if trainmodel:
	st.header("Modeling")
	y=df.UNAUTHORIZED_POP_ENTERED_US
	X=df[['Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE']].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	lr = LinearRegression()
	lr.fit(X_train,y_train)
	pred = lr.predict(X_test)
	mse = mean_squared_error(y_test,pred)
	rmse = sqrt(mse)

	st.markdown(f"""
	Linear Regression model trained :
		- MSE:{mse}
		- RMSE:{rmse}
	""")
	st.success('Model trained successfully')


if dokfold:
	st.subheader("KFOLD Random sampling Evalution")
	st.empty()
	my_bar = st.progress(0)

	from sklearn.model_selection import KFold

	X=df.values[:,-1].reshape(-1,1)
	y=df.UNAUTHORIZED_POP_ENTERED_US
	#st.progress()
	kf=KFold(n_splits=5)
	#X=X.reshape(-1,1)
	mse_list=[]
	rmse_list=[]
	r2_list=[]
	idx=1
	fig=plt.figure()
	i=0
	for train_index, test_index in kf.split(X):
	#	st.progress()
		my_bar.progress(idx*10)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		lr = LinearRegression()
		lr.fit(X_train,y_train)
		pred = lr.predict(X_test)
		
		mse = mean_squared_error(y_test,pred)
		rmse = sqrt(mse)
		r2=r2_score(y_test,pred)
		mse_list.append(mse)
		rmse_list.append(rmse)
		r2_list.append(r2)
		plt.plot(pred,label=f"dataset-{idx}")
		idx+=1
	plt.legend()
	plt.xlabel("Data points")
	plt.ylabel("Predictions")
	plt.show()
	st.plotly_chart(fig)

	res=pd.DataFrame(columns=["MSE","RMSE","r2_SCORE"])
	res["MSE"]=mse_list
	res["RMSE"]=rmse_list
	res["r2_SCORE"]=r2_list

	#st.write(res)
	#st.balloons()
#st.subheader("results of KFOLD")

#f=res.plot(kind='box',subplots=True)
#st.plotly_chart([f])
X= df[['Mex_GDP_Growth %', 'Mex_Annual_GDP_Change',
       'US_GDP_Growth %', 'US_Annual_GDP_Change',
       'MEXICO_MT_Co2_PER_CAPITA', 'MEXICO_INFLATION_RATE_PERCENTAGE',
       'MEXICO_INFLATION_ANNUAL_CHANGE']].values
y=df['UNAUTHORIZED_POP_ENTERED_US'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
#Saving the Model
classifier = lr

st.header('Immigrant Prediction')
select = st.selectbox('Select Form', ['Form 1'], key='1')
if not st.checkbox("Hide", True, key='2'):
    st.title('Immigration from Mexico Prediction')
    Mex_GDP_Growth_Percentage = st.number_input("Mex_GDP_Growth %:")
    Mex_Annual_GDP_Change = st.number_input("Mex_Annual_GDP_Change:")
    US_GDP_Growth_Percentage = st.number_input("US_GDP_Growth % :")
    US_Annual_GDP_Change =  st.number_input("US_Annual_GDP_Change:")
    MEXICO_MT_Co2_PER_CAPITA = st.number_input("MEXICO_MT_Co2_PER_CAPITA:")
    MEXICO_INFLATION_RATE_PERCENTAGE = st.number_input("MEXICO_INFLATION_RATE_PERCENTAGE:")
    MEXICO_INFLATION_ANNUAL_CHANGE = st.number_input("MEXICO_INFLATION_ANNUAL_CHANGE:")

st.write("Try to make your own prediciton as to how many people will come to America by assigning values to each variable")
submit = st.button('Predict')

#st.subheader("results of KFOLD")

#f=res.plot(kind='box',subplots=True)
#st.plotly_chart([f])
if submit:
        prediction = classifier.predict([[Mex_GDP_Growth_Percentage, Mex_Annual_GDP_Change, US_Annual_GDP_Change, US_GDP_Growth_Percentage, MEXICO_MT_Co2_PER_CAPITA, MEXICO_INFLATION_RATE_PERCENTAGE, MEXICO_INFLATION_ANNUAL_CHANGE]])
        st.write(round(np.mean(y_pred)))
		
        