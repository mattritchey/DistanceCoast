# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 07:26:42 2022

@author: mritchey
"""
# streamlit run "C:\Users\mritchey\.spyder-py3\Python Scripts\streamlit projects\Distance Coast\streamlit_nn.py"
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy import spatial
from vincenty import vincenty
from threading import Thread

@st.cache
def alt_vertices():
    return pd.read_parquet('https://github.com/mattritchey/DistanceCoast/raw/main/Atlantic%20Shore%20Vertices_short.parquet')


@st.cache
def gulf_vertices():
    return pd.read_parquet('https://github.com/mattritchey/DistanceCoast/raw/main/Gulf%20Shore%20Vertices2.parquet')

@st.cache
def convert_df(df):
    return df.to_csv(index=0).encode('utf-8')

@st.cache
def geocode(address):
    try:
        address2 = address.replace(' ', '+').replace(',', '%2C')
        df = pd.read_json(
            f'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?address={address2}&benchmark=2020&format=json')
        results = df.iloc[:1, 0][0][0]['coordinates']
        lat, lon = results['y'], results['x']
    except:
        geolocator = Nominatim(user_agent="GTA Lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(address)
        lat, lon = location.latitude, location.longitude
    return lat, lon


@st.cache
def thread_census_geocode_single(func_input, df):
    cols = df.columns.to_list()[:4]
    df['Address_q'] = df[cols[0]]+' %2C '+df[cols[1]] + \
        ' %2C '+df[cols[2]]+' '+df[cols[3]].str[:5]
    df['Address_q'] = df['Address_q'].str.replace(' ', '+')
    address_values = df['Address_q'].values
    results_thread = []

    def thread_func(value_input):
        response = func_input(value_input)
        results_thread.append(response)
        return True

    threads = []
    for i in range(len(address_values)):
        process = Thread(target=thread_func, args=[address_values[i]])
        process.start()
        threads.append(process)

    for process in threads:
        process.join()

    results = pd.merge(df, pd.concat(results_thread), on='Address_q')
    return results


def map_results(results):
    for index, row in results.iterrows():
        gulf_dis, alt_dis = results.loc[index,
                                        'Distance to Gulf'], results.loc[index, 'Distance to Atlantic']
        html = f"""<p style="arial"><p style="font-size:14px"> 
                Gulf: {gulf_dis.round(1)} miles 
                <br> Altantic: {alt_dis.round(1)} miles"""

        iframe = folium.IFrame(html)
        popup = folium.Popup(iframe,
                             min_width=140,
                             max_width=140)

        folium.Marker(location=[results.loc[index, 'Lat_input'],
                                results.loc[index, 'Lon_input']],
                      fill_color='#43d9de',
                      popup=popup,
                      radius=8).add_to(m)
    return folium


def distance(x):
    left_coords = (x[0], x[1])
    right_coords = (x[2], x[3])
    return vincenty(left_coords, right_coords, miles=True)


def nearest_neighbor(df_input, df_points):
    df_input = df_input.reset_index(drop=1)
    df_points = df_points.reset_index(drop=1)
    df_input2 = df_input[['Lon', 'Lat']]
    df_points = df_points.reset_index()
    df_kd_fires = spatial.KDTree(df_points[['Lon', 'Lat']])
    ktree = df_kd_fires.query(df_input2, k=1, workers=-1)
    output = pd.DataFrame(ktree[0], ktree[1]).reset_index()
    results = pd.merge(output, df_points, how='left', on='index')
    results2 = df_input.join(results, lsuffix='_input', rsuffix='_attached')
    results2['HubDist'] = pd.DataFrame([distance(i) for i in results2[[
                                       'Lat_input', 'Lon_input', 'Lat_attached', 'Lon_attached']].to_numpy()])
    results2['HubDist'] = results2['HubDist'].astype(float).round(2)
    return results2.drop(columns=['index', 0])


def distance_coast(df):
    Atl_vertices = alt_vertices()  # Attach to
    Gulf_vertices = gulf_vertices()  # Attach to
    df_atl = nearest_neighbor(df, Atl_vertices[['Lat', 'Lon']]).rename(columns={"HubDist": "Distance to Atlantic",
                                                                                "Lat_attached": "Lat_atl", "Lon_attached": "Lon_atl"})
    df_gulf = nearest_neighbor(df, Gulf_vertices[['Lat', 'Lon']]).rename(columns={"HubDist": "Distance to Gulf",
                                                                                  "Lat_attached": "Lat_gulf", "Lon_attached": "Lon_gulf"})
    df_gulf = df_gulf.drop(columns=['Lat_input', 'Lon_input'])
    final = pd.concat([df_atl, df_gulf], axis=1)
    final = final.loc[:, ~final.columns.duplicated()]
    return final


def census_geocode_single_address(address):
    try:
        df = pd.read_json(
            f'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?address={address}&benchmark=2020&format=json')
        results = df.iloc[:1, 0][0][0]['coordinates']
        x, y = results['x'], results['y']
    except:
        x, y = np.nan, np.nan
    return pd.DataFrame({'Address_q': address, 'Lat': y, 'Lon': x}, index=[0])


st.set_page_config(layout="wide")
col1, col2 = st.columns((2))

address = st.sidebar.text_input(
    "Address", "123 Main Street, Columbus, OH 43215")
uploaded_file = st.sidebar.file_uploader("Choose a file")
address_file = st.sidebar.radio('Choose',
                                ('Single Address', 'Addresses (Geocode: Will take a bit)', 'Lat Lons'))

if address_file == 'Lat Lons' and not None:
    try:
        df = pd.read_csv(uploaded_file)[['Lat', 'Lon']]
    except:
        st.header('Make Sure there is a Lat and Lon Field')

elif address_file == 'Addresses (Geocode: Will take a bit)' and not None:
    try:
        df = pd.read_csv(uploaded_file)
        cols = df.columns.to_list()[:4]
        with st.spinner("Geocoding: Hang On..."):
           df3 = thread_census_geocode_single(
               census_geocode_single_address, df)
        df = df3.query("Lat==Lat")
        errors = df3.query("Lat!=Lat")[cols].reset_index(drop=1)
    except:
        st.header('Make Sure File is Loaded First and then hit "Addresses"')

else:
    lat, lon = geocode(address)
    df = pd.DataFrame({'Lat': lat, 'Lon': lon}, index=[0])

results = distance_coast(df)

if address_file == 'Addresses':
    results = results[cols+['Lat_input', 'Lon_input',
                            'Distance to Gulf', 'Distance to Atlantic']]
else:
    results = results[['Lat_input', 'Lon_input',
                       'Distance to Gulf', 'Distance to Atlantic']]

m = folium.Map(location=[39.50, -98.35],  zoom_start=3)

with col1:
    st.title('Distance to Coast')
    map_results(results)
    st_folium(m, height=500, width=500)

with col2:
    st.title('Results')
    results.index = np.arange(1, len(results) + 1)
    st.dataframe(results)
    csv = convert_df(results)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Results.csv',
        mime='text/csv')
    try:
        if errors.shape[0] > 0:

            st.header('Errors')
            errors.index = np.arange(1, len(errors) + 1)
            st.dataframe(errors)
            # st.table(errors.assign(hack='').set_index('hack'))
            csv2 = convert_df(errors)
            st.download_button(
                label="Download Errors as CSV",
                data=csv2,
                file_name='Errors.csv',
                mime='text/csv')
    except:
        pass

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
