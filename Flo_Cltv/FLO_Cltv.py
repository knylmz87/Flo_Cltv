### BG-NBD ve Gamma-Gamma ile CLTV Tahmini ###

## İş Problemi ##
##Bir ayakkabi magazasi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan  müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin
# tahmin edilmesi gerekmektedir.##

## Veri Seti Hikayesi ##
# Veri seti Flo’dan son alışverişlerini 2020 -2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id : Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı(Android, ios, Desktop, Mobile)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

import datetime as dt

# Ilk olarak kutuphanelerimizi import edelim
!pip install lifetimes
import datetime as dt
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

# butun sutunlar gozuksun
pd.set_option('display.max_columns' , None)
#pd.set_option('display.max_rows', None)

# virgulden sonra kac basamak gelecek
pd.set_option('display.float_format' , lambda x: '%.2f' % x )

# dosyayi okutalim
df_ = pd.read_csv('C:/Users/pc/PycharmProjects/pythonProject2/3.Hafta/flo_data_20k.csv')

# dosyanin bir yedegini alalim
df = df_.copy()

# degiskenlere ait ilk 5 gozleme bakalim
df.head(5)

# kac degisken ve kac gozlem var
df.shape

#degiskenlerin isimleri nelerdir
df.columns

#betimsel istatistik degerleri

df.describe().T

#hangi degiskende ne kadar eksik deger var
df.isnull().sum()

df['master_id'].nunique

#Aykiri degerleri baskilamak icin fonksiyon yaziyoruz

def outlier_trashholds(dataframe , variable):
    quantile1 = dataframe[variable].quantile(0.01)
    quantile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit , up_limit

def replace_with_tresholds(dataframe , variable):
    low_limit , up_limit = outlier_trashholds(dataframe , variable)
    dataframe.loc[(dataframe[variable] < low_limit) , variable] = round(low_limit , 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit , 0)


# "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerlerini baskıliyoruz.

num_values_col = ['order_num_total_ever_online' , 'order_num_total_ever_offline' ,
                  'customer_value_total_ever_offline' , 'customer_value_total_ever_online']


for col in num_values_col:
    replace_with_tresholds(df , col)


#aykiri degerler baskilandiktan sonraki degisime bakalim

df.describe().T

#Offline ve online musterilerin toplam alisveris sayisi ve harcamasi

df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']
df.head()

# degisken tiplerine bakalim , tarih ifade eden degiskenlerin tipini date ye cevirelim


df.dtypes

df['first_order_date'] = df['first_order_date'].astype('datetime64[ns]')
df['last_order_date'] = df['last_order_date'].astype('datetime64[ns]')
df['last_order_date_online'] = df['last_order_date_online'].astype('datetime64[ns]')
df['last_order_date_offline'] = df['last_order_date_offline'].astype('datetime64[ns]')

df.dtypes

# Analiz yapilan tarihi son alisverisin yapildigi tarihten 2 gun sonrasini yapiyoruz
# Bunun icin ilk once son alisveris tarihini buluyoruz.

df['last_order_date'].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)
today_date
df.dtypes

#cltv_df adinda yeni bir df olusturup, bu df icine rfm degiskenlerini yerlestirelim
cltv_df = pd.DataFrame()
cltv_df['customer_id'] = df['master_id']
cltv_df['recency_cltv_weekly'] = ((df['last_order_date'] -df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['T_weekly'] = ((today_date - df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['frequency'] = df['order_num_total']
cltv_df['monetary_cltv_avg'] = df['customer_value_total'] / df['order_num_total']
cltv_df.head()

# BG-NBD Model nesnesini yaziyoruz

bgf = BetaGeoFitter(penalizer_coef=0.001)

# Modelimizi fit ediyoruz

bgf.fit(cltv_df['frequency'] ,
        cltv_df['recency_cltv_weekly'] ,
        cltv_df ['T_weekly'])

# 3 ay içerisinde müşterilerden beklenen satınalmaları tahmin edelim ve exp_sales_3_month olarak cltv df'sine ekleyelim

cltv_df['exp_sales_3_month'] = bgf.conditional_expected_number_of_purchases_up_to_time(3*4,
                                                                                       cltv_df['frequency'],
                                                                                       cltv_df['recency_cltv_weekly'],
                                                                                       cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satınalmaları tahmin edelim ve exp_sales_6_month olarak cltv df'sine ekleyelim

cltv_df['exp_sales_6_month'] = bgf.conditional_expected_number_of_purchases_up_to_time(6*4,
                                                                                       cltv_df['frequency'],
                                                                                       cltv_df['recency_cltv_weekly'],
                                                                                       cltv_df['T_weekly'])

cltv_df.head(10)

#plot_period_transactions(bgf)
#plt.show()

# GAMMA - GAMMA Model nesnemizi yaziyoruz

ggf = GammaGammaFitter(penalizer_coef=0.001)

#Modelimizi fit ediyoruz

ggf.fit(cltv_df['frequency'] , cltv_df['monetary_cltv_avg'])

#Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv_df'e ekleyelim

cltv_df['exp_average_value'] = ggf.conditional_expected_average_profit(cltv_df['frequency'] ,
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df['exp_average_value'].sort_values(ascending=False).head(10)


cltv_df.head()

# 6 aylık CLTV hesaplayalim ve cltv ismiyle dataframe'e ekleyelim.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq='W',
                                   discount_rate=0.01)
#cltv'yi cltf_df'e ekliyoruz
cltv_df['cltv'] = cltv

cltv_df.head()

#clvt degeri en yuksek 20 kisiyi gozlemliyoruz
cltv_df.sort_values('cltv' , ascending=False).head(20)

#6 aylık CLTV'ye göre tüm müşterilerimizi 4 gruba (segmente) ayırıp ve grup isimlerini veri setine ekliyoruz.

cltv_df['segment'] = pd.qcut(cltv_df['cltv'] , 4 , labels=['D','C','B','A'])

cltv_df.head()

cltv_df.describe().T.head(10)

cltv_df[(cltv_df['segment']=='A')].sort_values(by = 'T_weekly' , ascending=True).head(10)

cltv_df[(cltv_df['segment']=='D')].sort_values('cltv' , ascending=False).head(10)


