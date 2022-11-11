# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 07:26:42 2022

@author: mritchey
"""
#streamlit run "C:\Users\mritchey\.spyder-py3\Python Scripts\streamlit projects\streamlit_nn.py"
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy import spatial
from vincenty import vincenty

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=0).encode('utf-8')

@st.cache
def alt_vertices():
    return pd.read_parquet('https://github.com/mattritchey/DistanceCoast/raw/main/Atlantic%20Shore%20Vertices_short.parquet')

@st.cache
def gulf_vertices():
    return pd.read_parquet('https://github.com/mattritchey/DistanceCoast/raw/main/Gulf%20Shore%20Vertices2.parquet')


def distance(x):
    left_coords = (x[0], x[1])
    right_coords = (x[2], x[3])
    return vincenty(left_coords, right_coords, miles=True)	

def nearest_neighbor(df_input,df_points):
    df_input=df_input.reset_index(drop=1)
    df_points=df_points.reset_index(drop=1)
    df_input2=df_input[['Lon','Lat']]
    df_points=df_points.reset_index()
    df_kd_fires=spatial.KDTree(df_points[['Lon','Lat']])
    ktree=df_kd_fires.query(df_input2,k=1, workers=-1)
    output=pd.DataFrame(ktree[0],ktree[1]).reset_index()
    results=pd.merge(output,df_points, how='left', on='index')
    results2=df_input.join(results,lsuffix='_input',rsuffix='_attached')
    results2['HubDist']=pd.DataFrame([distance(i) for i in results2[['Lat_input','Lon_input','Lat_attached','Lon_attached']].to_numpy()])
    results2['HubDist']=results2['HubDist'].astype(float).round(2)
    return results2.drop(columns=['index',0])

def distance_coast(df):
    Atl_vertices= alt_vertices()#Attach to
    Gulf_vertices= gulf_vertices()#Attach to
    df_atl=nearest_neighbor(df,Atl_vertices[['Lat','Lon']]).rename(columns={"HubDist":"Distance to Atlantic",
                                                                            "Lat_attached":"Lat_atl","Lon_attached":"Lon_atl" })
    df_gulf=nearest_neighbor(df,Gulf_vertices[['Lat','Lon']]).rename(columns={"HubDist":"Distance to Gulf",
                                                                              "Lat_attached":"Lat_gulf","Lon_attached":"Lon_gulf" })	
    df_gulf=df_gulf.drop(columns=['Lat_input','Lon_input'])                                                                   
    final=pd.concat([df_atl,df_gulf],axis=1)
    return final

st.set_page_config(layout="wide")
col1, col2 = st.columns((2))

address_file = st.sidebar.radio('Address or File:', ('Address', 'File'))
address = st.sidebar.text_input(
    "Address", "123 Main Street, Loveland, OH 43215")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if address_file=='File':
    df=pd.read_csv(uploaded_file)
    
else:
    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(address)
    lat, lon = location.latitude, location.longitude
    df=pd.DataFrame({'Lat':lat,'Lon':lon},index=[0])

results=distance_coast(df)[['Lat_input','Lon_input','Distance to Gulf','Distance to Atlantic']]

lat_lons=results.values

m = folium.Map(location=[lat_lons[0][0], lat_lons[0][1]],  zoom_start=4)

for coord in lat_lons:
    gulf_dis,alt_dis=coord[2], coord[3]
    folium.Marker( location=[ coord[0], coord[1] ], 
                  fill_color='#43d9de', 
                  popup=f"""Gulf: {gulf_dis}; Altantic: {alt_dis} miles""",

                  radius=8 ).add_to( m )
with col1:
    st.title('Distance to Coast')
    st_folium(m, height=500,width=500)

with col2:
    st.title('Results')
    st.dataframe(results)
    csv = convert_df(results)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Results.csv',
        mime='text/csv')

