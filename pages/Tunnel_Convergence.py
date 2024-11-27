import streamlit as st
import os
import pandas as pd
import altair as alt
from datetime import datetime
import numpy as np
import pickle
import math
import joblib


def app():
    pass

def calculate_distance(row):
    initial_position = initial_positions.loc[row['Node']]
    return np.sqrt(
        (row['x'] - initial_position['x']) ** 2 +
        (row['y'] - initial_position['y']) ** 2 +
        (row['z'] - initial_position['z']) ** 2
    )

def calculate_diff_distance(row):

    return np.sqrt(row['x_diff'] ** 2 +row['y_diff'] ** 2 +row['z_diff'] ** 2)

def close_loop(points):
    # Ensure the loop is closed by repeating the first point
    points = np.vstack([points, points[0]])

    # Separate x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    ## Parameterize based on a simple linear parameter for each point
    #t = np.linspace(0, 1, len(points))

    # Create periodic cubic splines with boundary conditions
    #cs_x = CubicSpline(t, x, bc_type='periodic')
    #cs_y = CubicSpline(t, y, bc_type='periodic')

    # Generate fine points for a smooth closed loop
    #t_fine = np.linspace(0, 1,10)
    #x_fine = cs_x(t_fine)
    #y_fine = cs_y(t_fine)

    return x, y


dist_df = pd.DataFrame(data=[],columns=['From','To','Distance','timestamp','period'])
datapath = 'data/'
# Filter files in the directory containing "cha" in their names
filtered_files = [file for file in os.listdir(datapath) if "aligned" in file]

dashboard = st.container()
with st.sidebar:
    st.subheader("Developer: Anurak Puengrostham (Copyright)")
    contract = st.selectbox(
        'Select Contract: C2, C3', ('C2', 'C3')
    )
    dataFile = st.selectbox(
        'Select DataFile',filtered_files
    )

with dashboard:
    st.header(f"Tunnel Convergence Data Visualization")

    try:
        loaded_file = open(datapath + dataFile, 'rb')
        # dump information to that file
        #df = pickle.load(loaded_file)
        df = joblib.load(datapath+dataFile)
        df['x'] =  1 * df['x']
        df['y'] =  -1 * df['y']
        # df['z'] = -1*df['z']
        df['norm'] = df.apply(lambda row: np.linalg.norm([row['x'], row['y'], row['z']]), axis=1)

        # Plot Graph
        st.subheader(f"Denchai Chieng Rai - Chieng Khong Project: {dataFile[:-4]}")

        # Sidebar selection
        st.sidebar.header("Filter Options")

        selected_timestamps = st.sidebar.multiselect("Select Timestamp",
                                                     df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').unique())
        crown_node = st.sidebar.selectbox("Select Crown Node", df['Node'].unique())

        # Filter DataFrame based on selected timestamps
        filtered_df = df[df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').isin(selected_timestamps)]

        # Plot Shape of Tunnel
        plot_timestamps = filtered_df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S').unique()


        reference_shape = df[df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S') == plot_timestamps[0]][['x','y']]

        centroid_ref_x = reference_shape['x'].mean()
        centroid_ref_y = reference_shape['y'].mean()

        def calculate_angle(row):
            return math.atan2(row['y'] - centroid_ref_y, row['x'] - centroid_ref_x)

        reference_shape['angle'] = reference_shape.apply(calculate_angle, axis=1)
        reference_shape.sort_values(by='angle', ascending=False, inplace=True)


        latest_shape = filtered_df[filtered_df['timestamp'].dt.strftime('%Y-%m-%d_%H-%M-%S')==plot_timestamps[-1]][['x','y']]

        centroid_x = latest_shape['x'].mean()
        centroid_y = latest_shape['y'].mean()

        latest_shape['angle'] = latest_shape.apply(calculate_angle,axis =1)
        latest_shape.sort_values(by='angle',ascending=False, inplace =True)

        reference_shape_array = reference_shape[['x','y']].values
        latest_shape_array = latest_shape[['x','y']].values

        ref_x,ref_y = close_loop(reference_shape_array)
        _x, _y = close_loop(latest_shape_array)

        ref_xy_df = pd.DataFrame({'ref_x':ref_x, 'ref_y':ref_y})
        ref_xy_df['order'] = range(len(ref_xy_df))
        _xy_df = pd.DataFrame({'x':_x, 'y':_y})
        _xy_df['order'] = range(len(_xy_df))

        group_by_node_period = df.groupby(['Node', 'period']).mean().reset_index()
        group_by_node_period = group_by_node_period.sort_values(by=['Node','period'])


        # Compute row differences for numeric columns, grouped by `Node`
        numeric_cols = ['x', 'y', 'z']
        df_differences = group_by_node_period.groupby('Node')[numeric_cols].diff()
        df_differences = df_differences.fillna(0)
        group_by_node_period = pd.concat([group_by_node_period, df_differences.add_suffix('_diff')],axis=1)

        periods = sorted(list(set(group_by_node_period['period'].tolist())))
        select_period = st.sidebar.multiselect("Select Period", periods)
        select_period = pd.to_datetime(select_period)

        periods.sort()
        initial_positions = group_by_node_period[group_by_node_period['period'] == periods[0]].set_index('Node')[['x', 'y', 'z']]
        # Calculate distances and add as a new column
        group_by_node_period['displacement'] = group_by_node_period.apply(calculate_distance, axis=1) # diff distance compare to initial reading time[0]
        group_by_node_period['diff_displacement'] = group_by_node_period.apply(calculate_diff_distance, axis=1) # diff distance compare to previous timestep

        def apply_signed_norm(row, previous_norm):
            if previous_norm is None:  # First row, no previous comparison
                return row['diff_displacement']
            elif row['norm'] < previous_norm:  # Current norm is less than the previous row's norm
                return -row['diff_displacement']
            else:
                return row['diff_displacement']


        # Compute the signed norm
        signed_norm = []
        for i in range(len(group_by_node_period)):
            if i == 0:
                signed_norm.append(apply_signed_norm(group_by_node_period.loc[i], None))
            else:
                signed_norm.append(apply_signed_norm(group_by_node_period.loc[i], group_by_node_period.loc[i - 1, 'norm']))

        group_by_node_period['signed_diff_disp'] = signed_norm
        group_by_node_period['4d-Mavg'] = group_by_node_period['signed_diff_disp'].rolling(window=2).sum()
        group_by_node_period['period'] = pd.to_datetime(group_by_node_period['period'])
        group_by_node_period_filter = group_by_node_period[group_by_node_period['period'].isin(select_period.tolist())]
        print(group_by_node_period_filter)

        reference_shape_period = group_by_node_period_filter[group_by_node_period_filter['period'] == select_period[0]][['x', 'y']]

        centroid_ref_x_period = reference_shape_period['x'].mean()
        centroid_ref_y_period = reference_shape_period['y'].mean()

        def calculate_angle_period(row):
            return math.atan2(row['y'] - centroid_ref_y_period, row['x'] - centroid_ref_x_period)

        reference_shape_period['angle'] = reference_shape_period.apply(calculate_angle_period, axis=1)
        reference_shape_period.sort_values(by='angle', ascending=False, inplace=True)

        latest_shape_period = group_by_node_period_filter[group_by_node_period_filter['period'] == select_period[-1]][
            ['x', 'y']]

        centroid_x_period = latest_shape_period['x'].mean()
        centroid_y_period = latest_shape_period['y'].mean()

        latest_shape_period['angle'] = latest_shape_period.apply(calculate_angle, axis=1)
        latest_shape_period.sort_values(by='angle', ascending=False, inplace=True)

        reference_shape_period_array = reference_shape_period[['x', 'y']].values
        latest_shape_period_array = latest_shape_period[['x', 'y']].values

        ref_x_period, ref_y_period = close_loop(reference_shape_period_array)
        _x_period, _y_period = close_loop(latest_shape_period_array)

        ref_xy_period_df = pd.DataFrame({'ref_x': ref_x_period, 'ref_y': ref_y_period})
        ref_xy_period_df['order'] = range(len(ref_xy_period_df))
        _xy_period_df = pd.DataFrame({'x': _x_period, 'y': _y_period})
        _xy_period_df['order'] = range(len(_xy_period_df))


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

            ref_shape_period_plot = alt.Chart(ref_xy_df).mark_line().encode(
                x = 'ref_x',
                y = 'ref_y',
                order = 'order'
            )
            latest_shape_period_plot = alt.Chart(_xy_df).mark_line(strokeDash=[5,5],color='red').encode(
                x='x',
                y='y',
                order='order',
            )

            chart = (points + text + ref_shape_period_plot + latest_shape_period_plot).interactive()

            st.altair_chart(chart, use_container_width=True)

            st.subheader("Relative Movement Plot - Period")
            points_period = alt.Chart(group_by_node_period).mark_circle(size=75).encode(
                x='x:Q',
                y='y:Q',
                color=alt.Color('period:T', scale = alt.Scale(scheme='dark2')),
                opacity=alt.value(0.5),
                tooltip=['Node', 'x', 'y', 'z', 'period']
            )

            text_period = points_period.mark_text(
                align='left',
                dx=5,  # distance from the point
                dy=-5
            ).encode(
                text='Node:N'
            )

            ref_shape_period_plot = alt.Chart(ref_xy_period_df).mark_line().encode(
                x='ref_x',
                y='ref_y',
                order='order'
            )
            latest_shape_period_plot = alt.Chart(_xy_period_df).mark_line(strokeDash=[5, 5], color='red').encode(
                x='x',
                y='y',
                order='order',
            )

            chart_period = (points_period + text_period + ref_shape_period_plot + latest_shape_period_plot).interactive()
            st.altair_chart(chart_period, use_container_width=True)

            st.subheader("Data Analysis Plot")
            # Create the Altair total displacement line plot
            timeline = alt.Chart(group_by_node_period).mark_line().encode(
                x=alt.X('period:T', title='Period'),
                y=alt.Y('displacement:Q', title='Distance from Initial Position'),
                color='Node:N'
            ).properties(
                title="Distance from Initial Position Across Periods",
                width=600,
                height=400
            )

            st.altair_chart(timeline, use_container_width=True)

            # Create Histogram plot on crown node
            # Filter data for the specified Node
            specified_node = crown_node  # Change this to the desired Node
            node_data = group_by_node_period[group_by_node_period['Node'] == specified_node]

            # Melt the DataFrame to make it suitable for Altair
            melted_data = node_data.melt(
                id_vars=['Node'],
                value_vars=['diff_displacement'],
                var_name='Metric',
                value_name='Value'
            )

            # Create histogram
            histogram_chart = alt.Chart(melted_data).mark_bar(opacity=0.7, binSpacing=0).encode(
                alt.X('Value:Q', bin=alt.Bin(maxbins=10), title='Value'),
                alt.Y('count()', title='Frequency'),
                alt.Color('Metric:N', legend=alt.Legend(title='Metric')),
                tooltip=['Metric:N', 'count()']
            ).properties(
                title=f'Histograms for Node {specified_node}',
                width=600,
                height=300
            ).facet(
                facet=alt.Facet('Metric:N', title='Histogram Plot'),
                columns=3
            )

            st.altair_chart(histogram_chart, use_container_width=True)

            # Create the Altair line diff_displacement
            # Melt the DataFrame to make it suitable for Altair
            node_data_2 = group_by_node_period[group_by_node_period['Node'] == specified_node]

            melted_data_2 = node_data_2.melt(
                id_vars=['Node','period'],
                value_vars=['signed_diff_disp'],
                var_name='Metric',
                value_name='Value'
            )

            line_chart = alt.Chart(melted_data_2).mark_line(point=True).encode(
                x=alt.X('period:T', title='Period'),
                y=alt.Y('Value:Q', title='Signed Displacement'),
                tooltip=['period', 'Value:Q']
            ).properties(
                title='Daily Differential Movement',
                width=600,
                height=400
            )

            st.altair_chart(line_chart, use_container_width=True)

            # Moving Average Plot
            line_chart_Mavg = alt.Chart(group_by_node_period[group_by_node_period['Node']==specified_node]).mark_line(point=True).encode(
                x=alt.X('period:T', title='Period'),
                y=alt.Y('displacement:Q', title='Displacement'),
                tooltip=['period', '4d-Mavg:Q']
            ).properties(
                title='Accumulate Displacement Over Period',
                width=600,
                height=400
            )

            st.altair_chart(line_chart_Mavg, use_container_width=True)

            # Histogram
            histogram_diff_disp = alt.Chart(melted_data_2).mark_bar().encode(
                x=alt.X('Value:Q', bin=True, title='Signed Displacement'),
                y=alt.Y('count():Q', title='Count'),
                tooltip=['count():Q']
            ).properties(
                title='Histogram of Signed Displacement',
                width=600,
                height=400
            )

            st.altair_chart(histogram_diff_disp, use_container_width=True)

        else:
            st.write("No data available for the selected timestamps.")

    except:
        st.write('Data Not Found!')
