#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libraries
import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


import json 


get_ipython().system('pip install geocoder')
get_ipython().system('pip install geopy')
from geopy.geocoders import Nominatim 
import geocoder
import geopy

import requests  

from pandas.io.json import json_normalize 

from bs4 import BeautifulSoup

import matplotlib.cm as cm
import matplotlib.colors as colors


get_ipython().system('pip install sklearn')
from sklearn.cluster import KMeans

get_ipython().system('pip install folium')
import folium 

print('Libraries imported.')


# In[27]:


#For getting neighbourhood current population,I used "https://www.talent-berlin.de/en/living/districts-neighborhoods"

from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv

url=requests.get("https://www.talent-berlin.de/en/living/districts-neighborhoods").text
soup=BeautifulSoup(url,"lxml")

table=soup.find_all("tr")
headers = ["Borough","Area in ha","Population"]
     
rows = []
for row in table:
    td = row.find_all('td')
    row = [row.text for row in td]
    rows.append(row)

with open('neighbourhood.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(row for row in rows if row)

df1=pd.read_csv("neighbourhood.csv",encoding= 'unicode_escape')


# In[28]:


#Current Berlin population by borough
df1


# In[29]:


# Geographical Coordinates of Neighborhoods


def get_latlng(borough):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, Berlin, Germany'.format(borough))
        lat_lng_coords = g.latlng
    return lat_lng_coords

coords = [ get_latlng(borough) for borough in df1["Borough"].tolist() ]

df_coords = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])

# merge the coordinates into the original dataframe
df1['Latitude'] = df_coords['Latitude']
df1['Longitude'] = df_coords['Longitude']


# check the neighborhoods and the coordinates
print(df1.shape)
df1.head()


# In[30]:


address = 'Berlin, Germany'

geolocator = Nominatim(user_agent = "myapp1")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Berlin is {}, {}.'.format(latitude, longitude))


# In[31]:


# Create map of Berlin using latitude and longitude values
map_berlin = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough in zip(df1['Latitude'], df1['Longitude'], df1['Borough']):
    label = '{}'.format(borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_berlin)  
    
map_berlin


# ## Using Foursquare API

# In[32]:


#Now we will try to get venues by each borough of Berlin


# In[33]:


CLIENT_ID = 'YP03FATPA3WT3GI4K5D5EC2XOTUE3YKFYULHSR1DBJSIKK5M' # your Foursquare ID
CLIENT_SECRET = 'FQTXNXDDJYMZVYAUHCKY5TVVF1WZPJKLDMI411XXAGYOHV2P' # your Foursquare Secret
ACCESS_TOKEN = '42GH2CXALXEM5R1VEXYBCUVJETZ31E135YWUNA1DSPDPFMTT' # your FourSquare Access Token
VERSION = '20180604'
LIMIT = 50
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[34]:


radius = 1500
LIMIT = 50

venues = []
for lat, long, neighborhood in zip(df1['Latitude'], df1['Longitude'], df1['Borough']):
    
    # create the API request URL
    url = "https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat,
        long,
        radius, 
        LIMIT)
    
    # make the GET request
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
    # return only relevant information for each nearby venue
    for venue in results:
        venues.append((
            neighborhood,
            lat, 
            long, 
            venue['venue']['name'], 
            venue['venue']['location']['lat'], 
            venue['venue']['location']['lng'],  
            venue['venue']['categories'][0]['name']))


# In[90]:


# Convert the venues list into a new DataFrame
venues_df = pd.DataFrame(venues)

# define the column names
venues_df.columns = ['Borough', 'Latitude', 'Longitude', 'VenueName', 'VenueLatitude', 'VenueLongitude', 'VenueCategory']

print(venues_df.shape)
venues_df.head()


# In[91]:


venues_df.groupby(["Borough"]).count()
print('There are {} uniques categories.'.format(len(venues_df['VenueCategory'].unique())))


# 160 different  venues categories are available in Berlin.We are going to focus on Italian Restaurants.

# In[92]:


#### Analyse each neighborhood
# one hot encoding
onehot = pd.get_dummies(venues_df[['VenueCategory']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
onehot['Boroughs'] = venues_df['Borough'] 

# move neighborhood column to the first column
fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
onehot = onehot[fixed_columns]

print(onehot.shape)

grouped = onehot.groupby(["Boroughs"]).mean().reset_index()

print(grouped.shape)
grouped


# We decreased  520 observers to 12 by groupby.

# In[93]:


len(grouped[grouped["Italian Restaurant"] > 0])
berlin_rest = grouped[["Boroughs","Italian Restaurant","Bar"]]
berlin_rest


# ## Clustering

# In[94]:


#Now it is time to divide Berlin in clusters
kclusters = 4

berlin_clustering = berlin_rest.drop(["Boroughs"], 1)

# run k-means clustering
kmeans = KMeans(init="k-means++", n_clusters=kclusters, n_init=12).fit(berlin_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[95]:


merged = berlin_rest.copy()

# add clustering labels
merged["Category"] = kmeans.labels_
merged.rename(columns={"Boroughs": "Borough"}, inplace=True)
merged.head()


# In[96]:


dfmerged = merged.merge(df1)
dfmerged.head()

#Sort
dfmerged.sort_values(["Italian Restaurant"], inplace=True, ascending=True)
dfmerged


# In[97]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dfmerged['Latitude'], dfmerged['Longitude'], dfmerged['Borough'], dfmerged['Category']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[98]:


merged.loc[merged['Category'] == 0]


# In[99]:


merged.loc[merged['Category'] == 1]


# In[100]:


merged.loc[merged['Category'] == 2]


# In[101]:


merged.loc[merged['Category'] == 3]


# ### According to our Italian Restaurant number in the clusters ,the best category will be category 0. I will explain it in report.However,in here i want to take a second paremeter, beside Italian numbers, that includes which boroughs have much Italian immigrants. 

# In[102]:


import csv

website_url = requests.get('https://en.wikipedia.org/wiki/Demographics_of_Berlin').text
soup = BeautifulSoup(website_url,'lxml')
table = soup.find('table',{'class':'wikitable sortable'})
#print(soup.prettify())

headers = [header.text for header in table.find_all('th')]

table_rows = table.find_all('tr')        
rows = []
for row in table_rows:
    td = row.find_all('td')
    row = [row.text for row in td]
    rows.append(row)


with open('wiki.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(row for row in rows if row)
    
df2=pd.read_csv('wiki.csv',encoding= 'unicode_escape')
df2


# In[103]:


df2 = df2[df2['Largest Non-German ethnic groups\r\n'].str.contains('Italian',regex=False)]
df2.columns


# In[104]:


df2.drop(['Population 2010', 'Area in km²'],axis=1, inplace=True)
df2


# It can be seen, of course according to Wikipedia, 2 Boroughs have Italian living groups.

# ### I merged the tables, one of them shows Italian immigrants boroughs and the second one shows Italian restuarants number

# In[105]:


dfmerged = dfmerged.merge(df2)
dfmerged


# In[106]:


dfmerged.rename(columns = {'Largest Non-German ethnic groups\r\n': 'Largest Non-German ethnic groups'}, inplace = True)


# In[107]:


dfmerged


# In[108]:


dfmerged.replace(to_replace ="Turks, Poles, Serbs, Croats, Arabs, Italians\r\n",
                 value ="Turks, Poles, Serbs, Croats, Arabs, Italians",inplace=True)
dfmerged


# #### After merging the table includes Italian restuarant numbers and  the table includes more  Italian immigrants living boroughs, the results say us there are 2 options.I am going to pick Pankow up and the explanation will be in the report.

# In[ ]:




