# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sc
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(6, 6))

def csvreader(payth):
     df=pd.read_csv(payth,header=2)

     df.replace('..',0,inplace=True)
     df=df.fillna(0)
     #df.head()
     dftemp=df
     dftemp = dftemp.set_index('Country Name')

      # Transpose the DataFrame
     df_transposed = dftemp.transpose()
     return df,df_transposed

df,tfd=csvreader('API_EN.ATM.CO2E.PP.GD_DS2_en_csv_v2_4904492.csv')

df.head()

df_transposed=tfd
df_transposed.head()

df.describe()

df["Country Name"].unique()

df2,tr=csvreader('API_EN.ATM.CO2E.SF.KT_DS2_en_csv_v2_4904494.csv')
df2.head()

df2.describe()

df_stay=df.drop(["Country Name"	,"Country Code",'Indicator Name','Indicator Code'], axis=1)
df_stat1 = df_stay.astype(float)
df_stat1.head()

df_stat1.describe()

df_sta=df2.drop([	"Country Name"	,"Country Code",'Indicator Name','Indicator Code'], axis=1)
df_stat2 = df_sta.astype(float)
df_stat2.head()

df_stat2.describe()

fig, ax = plt.subplots()

n_rows=df_stat1.shape[0]

k=np.arange(n_rows)


ax.plot( np.arange(n_rows)[50:100], df_stat1['1990'].values[50:100], label='1990')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2000'].values[50:100], label='2000')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2012'].values[50:100], label='2012')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2013'].values[50:100], label='2013')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2014'].values[50:100], label='2014')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2015'].values[50:100], label='2015')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2016'].values[50:100], label='2016')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2017'].values[50:100], label='2017')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2018'].values[50:100], label='2018')
ax.plot( np.arange(n_rows)[50:100], df_stat1['2019'].values[50:100], label='2019')



ax.set_xlabel('countries')
ax.set_ylabel("CO2 emissions (kg per PPP $ of GDP)")
ax.set_title('CO2 emissions (kg per PPP $ of GDP) vs countries')

# add a legend
ax.legend()

# display the plot
plt.show()

figg, axx = plt.subplots()

n_row=df_stat2.shape[0]

k=np.arange(n_row)
axx.plot(  np.arange(n_rows)[50:100], df_stat2['1990'].values[50:100], label='1990')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2000'].values[50:100], label='2000')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2012'].values[50:100], label='2012')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2013'].values[50:100], label='2013')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2014'].values[50:100], label='2014')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2015'].values[50:100], label='2015')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2016'].values[50:100], label='2016')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2017'].values[50:100], label='2017')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2018'].values[50:100], label='2018')
axx.plot( np.arange(n_rows)[50:100], df_stat2['2019'].values[50:100], label='2019')

axx.set_xlabel('countries')
axx.set_ylabel("CO2 emissions from solid fuel consumption (kt)")
axx.set_title("CO2 emissions from solid fuel consumption (kt) vs countries")

# add a legend
axx.legend()

# display the plot
plt.show()

#x=df1.index
x = df["Country Name"].values[90:100]
y = df_stat1['2016'].values[90:100]

# Create bar chart
plt.bar(x, y)

# Add title and labels
plt.title("CO2 emissions (kg per PPP $ of GDP) 2016")
plt.xlabel("Country ")
plt.ylabel("2016")
plt.xticks(rotation=90)
plt.subplots_adjust(wspace=0.9)
# Show the chart
plt.show()

#x=df2.index
x = df["Country Name"].values[90:100]
y = df_stat2['2016'].values[90:100]

# Create bar chart
plt.bar(x, y)

# Add title and labels
plt.title("CO2 emissions from solid fuel consumption (kt)")
plt.xlabel("Country ")
plt.ylabel("2016")
plt.xticks(rotation=90)
# Show the chart
plt.show()

df4,dfe=csvreader('API_AG.LND.AGRI.K2_DS2_en_csv_v2_5327464.csv')
#plt.hist(df4["2016 [YR2016]"].astype(float).values[0:10] , bins=60, density=True, alpha=0.5, color='green')

df7=df4.drop([	"Country Name"	,"Country Code",'Indicator Name','Indicator Code'], axis=1)
data =df7.iloc[1:11,:10]

# create a heatmap using the 'hot' colormap
heatmap = plt.imshow(data, cmap='hot')


plt.colorbar(heatmap)

plt.yticks(np.arange(0.5, 10.5), df4["Country Name"].values[0:10])
plt.xticks(np.arange(0.5, 10.5), range(1961, 1971),rotation=90)

plt.xlabel('years')
plt.ylabel('countries')
plt.title('agricultural land (sq km)')


plt.show()

x=df4['Country Name'].values[30:50]
y1=df4['1980'].values[30:50].astype(float)
y2=df4['1990'].values[30:50].astype(float)
y3=df4['2000'].values[30:50].astype(float)
y4=df4['2010'].values[30:50].astype(float)
plt.bar(x, y1, width=0.2, align='center', label='1980')
plt.bar([i + 0.2 for i in range(len(x))], y2, width=0.2, align='center', label='1990')
plt.bar([i + 0.4 for i in range(len(x))], y3, width=0.2, align='center', label='2000')
plt.bar([i + 0.6 for i in range(len(x))], y4, width=0.2, align='center', label='2010')

# Add labels and legend
plt.xlabel('countries')
plt.ylabel('Years 1980 1990 2000 2010')
plt.title('agricultural land (sq km)')
plt.xticks(rotation=90)
plt.legend()

# Show the plot
plt.show()

df5,dfb=csvreader('API_AG.LND.AGRI.K2_DS2_en_csv_v2_5327464.csv')
#plt.hist(df5["2016 [YR2016]"].astype(float).values[0:10], bins=60, density=True, alpha=0.5, color='blue')
#print(df5["2016 [YR2016]"].head())
# Add title and axis labels
#plt.boxplot(df5["2016 [YR2016]"].astype(float).values)
plt.scatter(np.arange(266), df5["2016"].astype(float).values)
plt.title('Forest area (sq. km)')
plt.xlabel('2016 [YR2016]')
plt.ylabel('Frequency')

# Show plot
plt.show()



#scipy ttest_ind vetween CO2 emissions (kg per PPP $ of GDP) 2016 and CO2 emissions from solid fuel consumption (kt)
t_statistic, p_value = ttest_ind(df_stat1["2016"].values, df_stat2["2016"].values)

# Print results
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Calculate covariance
covariance = np.cov(df_stat1["2016"].values, df_stat2["2016"].values)

# Print covariance
print("Covariance:\n", covariance)