import streamlit as st
import os
import pandas as pd
import altair as alt
from datetime import datetime
import numpy as np
import pickle


def app():
    pass

def calculate_distance(row):
    initial_position = initial_positions.loc[row['Node']]
    return np.sqrt(
        (row['x'] - initial_position['x']) ** 2 +
        (row['y'] - initial_position['y']) ** 2 +
        (row['z'] - initial_position['z']) ** 2
    )

dist_df = pd.DataFrame(data=[],columns=['From','To','Distance','timestamp','period'])

datapath = 'data/'

dashboard = st.container()

with st.sidebar:
    contract = st.selectbox(
        'Select Contract: C2, C3', ('C2', 'C3')
    )
    dataFile = st.selectbox(
        'Select DataFile',os.listdir(datapath)
    )

with dashboard:
    st.header(f"Tunnel Convergence Data Visualization")

    try:
        loaded_file = open(datapath + dataFile, 'rb')
        # dump information to that file
        df = pickle.load(loaded_file)
        # df['x'] = -1*df['x']
        df['y'] = -1 * df['y']
        # df['z'] = -1*df['z']

        # Plot Graph
        st.subheader(f"Denchai Chieng Rai - Chieng Khong Project: {contract}")

        # Sidebar selection
        st.sidebar.header("Filter Options")
        selected_timestamps = st.sidebar.multiselect("Select Timestamp", df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').unique())

        # Filter DataFrame based on selected timestamps
        filtered_df = df[df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').isin(selected_timestamps)]

        group_by_node_period = df.groupby(['Node', 'period']).mean().reset_index()
        group_by_node_period = group_by_node_period.sort_values(by='period')
        periods = list(set(group_by_node_period['period'].tolist()))
        periods.sort()
        # Find initial position for each Node at period[0]
        initial_positions = group_by_node_period[group_by_node_period['period'] == periods[0]].set_index('Node')[['x', 'y', 'z']]

        # Calculate distances and add as a new column
        group_by_node_period['displacement'] = group_by_node_period.apply(calculate_distance, axis=1)

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

            # Create the Altair line plot
            timeline = alt.Chart(group_by_node_period).mark_line().encode(
                x=alt.X('period:O', title='Period'),
                y=alt.Y('displacement:Q', title='Distance from Initial Position'),
                color='Node:N'
            ).properties(
                title="Distance from Initial Position Across Periods",
                width=600,
                height=400
            )

            st.altair_chart(timeline, use_container_width=True)

        else:
            st.write("No data available for the selected timestamps.")

    except: st.write('Data Not Found!')

