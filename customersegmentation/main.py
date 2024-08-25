import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

data=pd.read_csv("C:/Users/Ece/Desktop/Veri.csv")
veri=data.copy()
veri=veri.dropna()

veri["Total"]=veri["Quantity"]*veri["Price"]

veri=veri.drop(veri[veri["Total"]<=0].index)

Q1=veri["Total"].quantile(0.25)
Q3=veri["Total"].quantile(0.75)

IQR=Q3-Q1

altsınır=Q1-1.5*IQR
ustsınır=Q3+1.5*IQR

veri=veri[~((veri["Total"]>ustsınır) | (veri["Total"]<altsınır))]
veri=veri.reset_index(drop=True)

veri["Customer ID"]=veri["Customer ID"].astype("int")
veri["InvoiceDate"]=pd.to_datetime(veri["InvoiceDate"])

bugün=veri["InvoiceDate"].max()
bugün=dt.datetime(2010,12,9,20,1,0)

r=(bugün-veri.groupby("Customer ID").agg({"InvoiceDate":"max"})).apply(lambda x:x.dt.days)

f=veri.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
f=f.groupby("Customer ID").agg({"Invoice":"count"})

m=veri.groupby("Customer ID").agg({"Total":"sum"})

RFM=r.merge(f,on="Customer ID").merge(m,on="Customer ID")
RFM=RFM.reset_index()
RFM=RFM.rename(columns={"Customer ID":"Customer","InvoiceDate":"Recency","Invoice":"Frequency","Total":"Monetary"})


df=RFM.iloc[:,1:]
print(df)

sc=MinMaxScaler()
dfnorm=sc.fit_transform(df)
dfnorm=pd.DataFrame(dfnorm,columns=df.columns)

kmodel=KMeans(random_state=0,n_clusters=4,init="k-means++")
kfit=kmodel.fit(dfnorm)
labels=kfit.labels_


RFM["Labels"]=labels

print(RFM.groupby("Labels").mean().iloc[:,1:])
