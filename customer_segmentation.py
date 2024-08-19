import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_excel('data.xlsx')
df_not_nan = df[df['CustomerID'].notna()]
df_modified = df_not_nan.loc[df_not_nan.Quantity >= 0]
df_fixed = df_modified.sample(10000,random_state = 42)

q25_1, q75_1 = np.percentile(df_fixed['Quantity'], 25), np.percentile(df_fixed['Quantity'], 75)
iqr_1 = q75_1 - q25_1

limit_iqr_1 = 1.5*iqr_1
lower_iqr_1, upper_iqr_1 = q25_1 - limit_iqr_1, q75_1 + limit_iqr_1

q25_2, q75_2 = np.percentile(df_fixed['UnitPrice'], 25), np.percentile(df_fixed['UnitPrice'], 75)
iqr_2 = q75_2 - q25_2

limit_iqr_2 = 1.5*iqr_2
lower_iqr_2, upper_iqr_2 = q25_2 - limit_iqr_2, q75_2 + limit_iqr_2

df_completed = df_fixed.loc[(df_fixed.Quantity >= lower_iqr_1) & (df_fixed.Quantity <= upper_iqr_1) & (df_fixed.UnitPrice >= lower_iqr_2) & (df_fixed.UnitPrice <= upper_iqr_2)]

sns.countplot(data = df_completed, x = 'Quantity')
plt.show()

df_completed['InvoiceDate'] = pd.to_datetime(df_completed['InvoiceDate'],format = '%Y-%m-%d %H:%M:%S')

current_date = max(df_completed['InvoiceDate']) + datetime.timedelta(days = 1)

df_completed['TotalPay'] = df_completed['Quantity'] * df_completed['UnitPrice']

df_customers = df_completed.groupby(['CustomerID']).agg(
   {'InvoiceDate' : lambda x: (current_date - x.max()).days,
    'InvoiceNo' : 'count',
    'TotalPay' : 'sum'
    }
)

df_customers.rename(columns = {'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalPay':'MonetaryValue'},inplace = True)

fig , ax = plt.subplots(1,3 ,figsize = (16,5))
sns.distplot(df_customers['Recency'],ax = ax[0])
sns.distplot(df_customers['Frequency'],ax = ax[1])
sns.distplot(df_customers['MonetaryValue'],ax = ax[2])
plt.show()

def analyze_skewness(x):
    fig , ax = plt.subplots(2,2, figsize = (5,5))
    sns.distplot(df_customers[x],ax = ax[0,0])
    sns.distplot(np.log(df_customers[x]),ax = ax[0,1])
    sns.distplot(np.sqrt(df_customers[x]),ax = ax[1,0])
    sns.distplot(stats.boxcox(df_customers[x])[0],ax = ax[1,1])
    plt.tight_layout()
    plt.show()

    print('Original Value:', df_customers[x].skew().round(2))
    print('Value of Log Skewness:', np.log(df_customers[x]).skew().round(2))
    print('Value of Sqrt Skewness:', np.sqrt(df_customers[x]).skew().round(2))
    print('Value of Boxcox Skewness:', pd.Series(stats.boxcox(df_customers[x])[0]).skew().round(2))

analyze_skewness('Recency')

df_customers_t = pd.DataFrame()
df_customers_t['Recency'] = stats.boxcox(df_customers['Recency'])[0]
df_customers_t['Frequency'] = stats.boxcox(df_customers['Frequency'])[0]
df_customers_t['MonetaryValue'] = pd.Series(np.cbrt((df_customers['MonetaryValue']))).values

scaler = StandardScaler()
scaler.fit(df_customers_t)
df_customers_t = scaler.transform(df_customers_t)

sse ={}
for k in range (1,11):
  kmeans = KMeans(n_clusters = k, random_state = 42)
  kmeans.fit(df_customers_t)
  sse[k] = kmeans.inertia_

plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x = list(sse.keys()),y = list(sse.values()))
plt.show()

for n_clusters in range(3,10):
    check_model = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    check_model.fit(df_customers_t)
    clusters = check_model.predict(df_customers_t)
    silhouette_avg = silhouette_score(df_customers_t,check_model.labels_)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

model = KMeans(n_clusters = 5, random_state = 42)
model.fit(df_customers_t)

df_customers['Cluster'] = model.labels_
df_customers.groupby('Cluster').agg(
    {
        'Recency' : 'mean',
        'Frequency' : 'mean',
        'MonetaryValue' : 'mean'
    }
).round(2)

df_cluster_0 = df_customers.loc[df_customers.Cluster == 0]
fig , ax = plt.subplots(1,3 ,figsize = (18,5))
sns.distplot(df_cluster_0['Recency'],ax = ax[0])
sns.distplot(df_cluster_0['Frequency'],ax = ax[1])
sns.distplot(df_cluster_0['MonetaryValue'],ax = ax[2])
plt.show()

df_cluster_1 = df_customers.loc[df_customers.Cluster == 1]
fig , ax = plt.subplots(1,3 ,figsize = (18,5))
sns.distplot(df_cluster_1['Recency'],ax = ax[0])
sns.distplot(df_cluster_1['Frequency'],ax = ax[1])
sns.distplot(df_cluster_1['MonetaryValue'],ax = ax[2])
plt.show()

df_cluster_2 = df_customers.loc[df_customers.Cluster == 2]
fig , ax = plt.subplots(1,3 ,figsize = (18,5))
sns.distplot(df_cluster_2['Recency'],ax = ax[0])
sns.distplot(df_cluster_2['Frequency'],ax = ax[1])
sns.distplot(df_cluster_2['MonetaryValue'],ax = ax[2])
plt.show()

df_cluster_3 = df_customers.loc[df_customers.Cluster == 3]
fig , ax = plt.subplots(1,3 ,figsize = (18,5))
sns.distplot(df_cluster_3['Recency'],ax = ax[0])
sns.distplot(df_cluster_3['Frequency'],ax = ax[1])
sns.distplot(df_cluster_3['MonetaryValue'],ax = ax[2])
plt.show()