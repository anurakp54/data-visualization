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
        df['x'] = -1 * df['x']
        df['y'] = -1 * df['y']
        # df['z'] = -1*df['z']

        # Plot Graph
        st.subheader(f"Denchai Chieng Rai - Chieng Khong Project: {dataFile[:-4]}")

        # Sidebar selection
        st.sidebar.header("Filter Options")
        selected_timestamps = st.sidebar.multiselect("Select Timestamp",
                                                     df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').unique())

        # Filter DataFrame based on selected timestamps
        filtered_df = df[df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').isin(selected_timestamps)]

        # Plot Shape of Tunnel
        plot_timestamps = filtered_df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').unique()

        reference_shape = df[df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S') == plot_timestamps[0]][['x', 'y']]

        centroid_ref_x = reference_shape['x'].mean()
        centroid_ref_y = reference_shape['y'].mean()


        def calculate_angle(row):
            return math.atan2(row['y'] - centroid_ref_y, row['x'] - centroid_ref_x)


        reference_shape['angle'] = reference_shape.apply(calculate_angle, axis=1)
        reference_shape.sort_values(by='angle', ascending=False, inplace=True)

        latest_shape = filtered_df[filtered_df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S') == plot_timestamps[-1]][
            ['x', 'y']]

        centroid_x = latest_shape['x'].mean()
        centroid_y = latest_shape['y'].mean()

        latest_shape['angle'] = latest_shape.apply(calculate_angle, axis=1)
        latest_shape.sort_values(by='angle', ascending=False, inplace=True)

        reference_shape_array = reference_shape[['x', 'y']].values
        latest_shape_array = latest_shape[['x', 'y']].values

        ref_x, ref_y = close_loop(reference_shape_array)
        _x, _y = close_loop(latest_shape_array)

        ref_xy_df = pd.DataFrame({'ref_x': ref_x, 'ref_y': ref_y})
        ref_xy_df['order'] = range(len(ref_xy_df))
        _xy_df = pd.DataFrame({'x': _x, 'y': _y})
        _xy_df['order'] = range(len(_xy_df))

        group_by_node_period = df.groupby(['Node', 'period']).mean().reset_index()
        group_by_node_period = group_by_node_period.sort_values(by='period')
        periods = list(set(group_by_node_period['period'].tolist()))
        periods.sort()
        # Find initial position for each Node at period[0]
        initial_positions = group_by_node_period[group_by_node_period['period'] == periods[0]].set_index('Node')[
            ['x', 'y', 'z']]

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

            ref_shape_plot = alt.Chart(ref_xy_df).mark_line().encode(
                x='ref_x',
                y='ref_y',
                order='order'
            )
            latest_shape_plot = alt.Chart(_xy_df).mark_line(strokeDash=[5, 5], color='red').encode(
                x='x',
                y='y',
                order='order',
            )

            chart = (points + text + ref_shape_plot + latest_shape_plot).interactive()

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

    except:
        st.write('Data Not Found!')
