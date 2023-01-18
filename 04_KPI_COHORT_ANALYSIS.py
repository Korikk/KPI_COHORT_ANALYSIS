############################################
# KPI & COHORT ANALIZI: RETENTION RATE
############################################

# 3 ADIMDA RETENTION RATE KPI'NIN COHORT ANALIZINE SOKULMASI

# 1. Veri ön işleme
# 2. Retention matrisinin oluşturulması
#    1. Her bir müşteri için eşsiz sipariş sayısının hesaplanması
#    2. Tüm veri setinde bir kereden fazla sipariş veren müşteri oranı.
#    3. Sipariş aylarının yakalanması.
#    4. Cohort değişkeninin oluşturulması.
#    5. Aylık müşteri sayılarını çıkarılması.
#    6. Periyod numarasının çıkarılması
#    7. Cohort_pivot'un oluşturulması
#    8. Retention_matrix'in oluşturulması
# 3. Retention matrisinin ısı haritası ile görselleştirilmesi


####################################
# 1. Veri ön işleme
####################################



import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from operator import attrgetter
import matplotlib.colors as mcolors

df_ = pd.read_excel('Week_03/datasets/online_retail.xlsx',
                   dtype={'CustomerID': str,
                          'InvoiceID': str},
                   parse_dates=['InvoiceDate'])
df = df_.copy()
df.head()
df.shape
df.info()

df.dropna(subset=['CustomerID'], inplace=True)
df = df[['CustomerID', 'InvoiceNo', 'InvoiceDate']].drop_duplicates()

df.shape






####################################
# 2. Retention matrisinin oluşturulması
####################################

# SONUC
# n_orders = df.groupby(['CustomerID'])['InvoiceNo'].nunique()
# orders_perc = np.sum(n_orders > 1) / df['CustomerID'].nunique()
# df['order_month'] = df['InvoiceDate'].dt.to_period('M')
# df['cohort'] = df.groupby('CustomerID')['InvoiceDate'] \
#     .transform('min') \
#     .dt.to_period('M')
# df_cohort = df.groupby(['cohort', 'order_month']) \
#     .agg(n_customers=('CustomerID', 'nunique')) \
#     .reset_index(drop=False)
# df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(attrgetter('n'))
# cohort_pivot = df_cohort.pivot_table(index='cohort',
#                                      columns='period_number',
#                                      values='n_customers')
#
# cohort_size = cohort_pivot.iloc[:, 0]
# retention_matrix = cohort_pivot.divide(cohort_size, axis=0)


# 1. her bir müşteri için eşsiz sipariş sayısının hesaplanması
n_orders = df.groupby(['CustomerID'])['InvoiceNo'].nunique()

# 2. tüm veri setinde bir kereden fazla sipariş veren müşteri oranı.
orders_perc = np.sum(n_orders > 1) / df['CustomerID'].nunique()

100*orders_perc

# 3. sipariş aylarının yakalanması.
df['order_month'] = df['InvoiceDate'].dt.to_period('M')

# 4. cohort değişkeninin oluşturulması.
df['cohort'] = df.groupby('CustomerID')['InvoiceDate'] \
    .transform('min') \
    .dt.to_period('M')

# 5. aylık müşteri sayılarını çıkarılması.
df_cohort = df.groupby(['cohort', 'order_month']) \
    .agg(n_customers=('CustomerID', 'nunique')) \
    .reset_index(drop=False)


# 6. periyod numarasının çıkarılması
(df_cohort.order_month - df_cohort.cohort).head()

df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(attrgetter('n'))

# 7. cohort_pivot'un oluşturulması
cohort_pivot = df_cohort.pivot_table(index='cohort',
                                     columns='period_number',
                                     values='n_customers')


cohort_size = cohort_pivot.iloc[:, 0]


# 8. retention_matrix'in oluşturulması
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
retention_matrix



def create_retention_matrix(dataframe):
    n_orders = dataframe.groupby(['CustomerID'])['InvoiceNo'].nunique()
    dataframe['order_month'] = dataframe['InvoiceDate'].dt.to_period('M')
    dataframe['cohort'] = dataframe.groupby('CustomerID')['InvoiceDate'] \
        .transform('min') \
        .dt.to_period('M')
    df_cohort = dataframe.groupby(['cohort', 'order_month']) \
        .agg(n_customers=('CustomerID', 'nunique')) \
        .reset_index(drop=False)
    df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(attrgetter('n'))
    cohort_pivot = df_cohort.pivot_table(index='cohort',
                                         columns='period_number',
                                         values='n_customers')

    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    return retention_matrix


create_retention_matrix(df)





####################################
# 3. Retention matrisinin ısı haritası ile görselleştirilmesi
####################################


sns.axes_style("white")
fig, ax = plt.subplots(1, 2,
                       figsize=(12, 8),
                       sharey=True,  # y eksenini paylas
                       gridspec_kw={'width_ratios': [1, 11]}
                       # to create the grid the subplots are placed on
                       )

# retention matrix
sns.heatmap(retention_matrix,
            annot=True,
            fmt='.0%',  # grafikteki ifadelerin yüzdelik gösterimi
            cmap='RdYlGn',  # colormap
            ax=ax[1])  # subplot'taki grafikleri seçmek

ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
ax[1].set(xlabel='# of periods', ylabel='')


# cohort size
cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
white_cmap = mcolors.ListedColormap(['white'])
sns.heatmap(cohort_size_df,
            annot=True,
            cbar=False,  # ikinci grafik için cbar istemiyoruz (sağ taraftaki renkli ölçeklendirme)
            fmt='g',
            cmap=white_cmap,
            ax=ax[0])
fig.tight_layout()
plt.show()
































# RECAP
# Virtual env.
# Dependency man.
# Python (functions, if-else, loops, list comp.)
# EDA

# PROJE'den elde edilmesi gereken çıktılar nelerdi?
# - Herhangi ds, ml tekniği bilmeyen kişi nasıl segmentasyon yapar bunu görmek.
# - Dinamik label ve bin oluşturmak
# - Elimizde herhangi verisi olmadığı halde olası müşteri potansiyelini belirlemek (sınıflandırmak)

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
users = pd.read_csv('datasets/users.csv')
purchases = pd.read_csv('datasets/purchases.csv')
df = purchases.merge(users, how='inner', on='uid')

agg_df = df.groupby(by=["country", 'device', "gender", "age"]).agg({"price": "sum"}).sort_values("price", ascending=False)
agg_df = agg_df.reset_index()
bins = [0, 19, 24, 31, 41, agg_df["age"].max()]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["age"].max())]
agg_df["age_cat"] = pd.cut(agg_df["age"], bins, labels=mylabels)


agg_df["customers_level_based"] = [row[0] + "_" + row[1].upper() + "_" + row[2] + "_" + row[5] for row in agg_df.values]
agg_df = agg_df[["customers_level_based", "price"]]
agg_df = agg_df.groupby("customers_level_based").agg({"price": "mean"})
agg_df = agg_df.reset_index()

agg_df["segment"] = pd.qcut(agg_df["price"], 4, labels=["D", "C", "B", "A"])

new_user = "TUR_IOS_F_41_75"
new_user = "USA_AND_F_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

