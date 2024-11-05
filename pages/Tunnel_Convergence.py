import streamlit as st
import os
import pandas as pd
import altair as alt
from datetime import datetime
import numpy as np
import pickle


def app():
    pass

dist_df = pd.DataFrame(data=[],columns=['From','To','Distance','timestamp','period'])

dashboard = st.container()
with st.sidebar:
    contract = st.selectbox(
        'Select Contract: C2, C3', ('C2', 'C3')
    )

    Direction = st.selectbox(
        'Select Direction: North-Up, North-Down, South-Up, South-Down',('N-UP', 'N-DN','S-UP','S-DN')
    )

datapath = 'data/'
files = os.listdir(datapath)
for file in files:
    if (file[:7]) == contract+"-"+Direction:
        loaded_file = open(datapath + file, 'rb')
        # dump information to that file
        df = pickle.load(loaded_file)
        #df['x'] = -1*df['x']
        df['y'] = -1*df['y']
        #df['z'] = -1*df['z']

with dashboard:
    st.header(f"Tunnel Convergence Data Visualization")

    try:
        # Plot Graph
        st.subheader(f"Denchai Chieng Rai - Chieng Khong Project: {contract}-{Direction}")

        # Sidebar selection
        st.sidebar.header("Filter Options")
        selected_timestamps = st.sidebar.multiselect("Select Timestamp", df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').unique())

        # Filter DataFrame based on selected timestamps
        filtered_df = df[df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').isin(selected_timestamps)]


        # Plot with Altair
        if not filtered_df.empty:
            points = alt.Chart(filtered_df).mark_circle(size=75).encode(
                x='x:Q',
                y='y:Q',
                color='period:N',
                opacity=alt.value(0.5),
                tooltip=['Node', 'x', 'y', 'z', 'timestamp', 'period']
            )
            # Add text labels for each point (Node names)

            text = points.mark_text(
                align='left',
                dx=5,  # distance from the point
                dy=-5
            ).encode(
                text='Node:N'
            )
            chart = (points+text).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available for the selected timestamps.")

        # Display filtered DataFrame
        st.write("Coordinate Datat of Tunnel-" + contract + "-" + Direction)
        st.write(filtered_df)

    except: st.write('Data Not Found!')

