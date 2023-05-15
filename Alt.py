import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import itertools
import calendar

def create_3d_plot_3(data):  # Changed df to data
    # Convert the DateTime column to a datetime object
    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y %H:%M')

    # Group by year and month
    data['Year'] = data['DateTime'].dt.year
    data['Month'] = data['DateTime'].dt.month
    grouped = data.groupby(['Year', 'Month']).size().reset_index(name='Count')

    # Create a 3D line plot using Plotly
    data = []
    years = grouped['Year'].unique()

    for year in years:
        year_data = grouped[grouped['Year'] == year]
        trace = go.Scatter3d(
            x=year_data['Month'],
            y=year_data['Year'].astype(str),  # Use the year as the Y-axis
            z=year_data['Count'],
            mode='lines',
            name=str(year),
            line=dict(width=2),
            hovertemplate=(
                "Month: %{x}<br>"
                "Year: %{y}<br>"
                "Listens: %{z}<extra></extra>"
            )
        )
        data.append(trace)

    layout = go.Layout(
        title='Listens Over Time (3D Line Plot)',
        autosize=True,
        margin=dict(l=10, r=10, t=100, b=10),
        scene=dict(
            xaxis_title='Month',
            yaxis_title='Year',
            zaxis_title='Number of Listens',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                 'Sep', 'Oct', 'Nov', 'Dec']
            ),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-2, y=-2, z=1)
            ),
        ),
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def filter_data(data, year, month):  # Changed df to data
    return data['Year_Month'] == pd.Period(year=year, month=month, freq='M')

def get_top_artists(group, n=10):
    return group.sort_values('Running_Total', ascending=False).head(n)

def update_radial_chart(day, year, month):
    # Filter the data for the selected day
    selected_day_data = data[(data['DateTime'].dt.year == selected_year) &
                             (data['DateTime'].dt.month == selected_month) &
                             (data['DateTime'].dt.day == day)]

    # Group by hour and count the number of listens
    hourly_counts = selected_day_data.groupby(selected_day_data['DateTime'].dt.hour).size()

    # Clear the current radial chart
    radial_chart.data = []

    # Add the new data to the radial chart
    radial_chart.add_trace(go.Barpolar(
        r=hourly_counts.values,
        theta=hourly_counts.index * 360 / 24,
        name='Listens',
        marker_color='blue',  # Or any color you prefer
        hovertemplate="Hour: %{theta}<br>Listens: %{r}<extra></extra>"
    ))

@st.cache_data
def load_data():
    """Load and preprocess the data. We cache this function so that Streamlit
    doesn't reload the data every time it reruns the script."""
    data = pd.read_csv('all_tracks_utf8_done.csv')
    data['DateTime'] = pd.to_datetime(data['DateTime'], format="%d/%m/%Y %H:%M")
    return data

# Load the data
data = load_data()

def add_bg_from_url(markdown_text):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1587731556938-38755b4803a6?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2678&q=80");
            background-attachment: fixed;
            background-size: cover
        }}
        .element-container {{
            background-color: rgba(0, 23, 43, 0.95);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            width: 105%;
        }}
        </style>
        {markdown_text}
        """,
        unsafe_allow_html=True
    )

# Create a tab layout
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the page:", ["About","Top Artists, Albums, Tracks", "Scrobbles Over Time", "3D Line Plot", "Heatmap and Radial Bar", "WordCloud", "Bar Chart Race", "Scatter Plot"])
if app_mode == "About":
    add_bg_from_url(
        "\n# An Aural History - Cameron's Life in Music"
        "\nWelcome to my personal exploration of music through the lens of data. This"
        " project is a testament to my passion for music and data analysis, combining"
        " my personal music listening data from Last.fm with Python-based data analysis"
        " and visualizations. You can find my music profile here: https://www.last.fm/user/cgomez08."

        "\n### Project Overview"
        "\nSpanning 15 years of listening history from May 26, 2008 to May 3, 2023, this project"
        " dives into my listening habits, tastes, and how they've evolved (or stayed the same!) over time."
        " Each page presents interactive charts and data visualizations, offering"
        " different perspectives and insights into my musical journey. Whether it's"
        " exploring my top artists, albums, or how my music preferences have changed over"
        " the years, this project invites you on a data-driven exploration of my life in music."
        " It's a unique, personal view into the soundtrack of my life, expressed through the prism of data analysis."
    )

if app_mode == "Top Artists, Albums, Tracks":
    add_bg_from_url(
    f"""
    # Top Artists, Albums, and Tracks
    ### Listening Stats - At A Glance:
    - Unique Artists: {data['Artist'].nunique()}
    - Unique Albums: {data['Album'].nunique()}
    - Unique Tracks: {data['Track'].nunique()}
    - Listening Date Range: {data['DateTime'].min()} - {data['DateTime'].max()}
    """
    )
    # Add Year column
    data['Year'] = data['DateTime'].dt.year

    # Create a selectbox
    exclude_artist = st.selectbox('Exclude Artist from Top Artists, Albums, and Tracks', options=['None', 'The Strokes'])

    # Create a selectbox for Year selection
    years = sorted(data['Year'].unique())
    selected_year = st.selectbox('Select Year', options=['All'] + list(years))

    if exclude_artist == 'The Strokes':
        # Filter out "The Strokes"
        filtered_data = data.loc[data['Artist'] != 'The Strokes']
    else:
        filtered_data = data

    if selected_year != 'All':
        # Filter data for the selected year
        filtered_data = filtered_data[filtered_data['Year'] == selected_year]

    top_artists = filtered_data['Artist'].value_counts().head(10).reset_index()
    top_artists.columns = ['Artist', 'Track Count']
    fig = px.bar(top_artists, y='Track Count', x='Artist',
             labels={'Artist':'Artist', 'Track Count':'Track Count'},
             hover_data=['Artist', 'Track Count'])
    fig.update_traces(hovertemplate='Artist: %{x}<br>Track Count: %{y}<extra></extra>')
    fig.update_layout(title_text='Top 10 Artists')
    st.plotly_chart(fig)

    # Top Albums
    top_albums = filtered_data.groupby(['Artist','Album']).size().reset_index(name='Play Count')
    top_albums = top_albums.sort_values(by='Play Count', ascending=False).head(10)
    fig = px.bar(top_albums, y='Play Count', x='Album',
        hover_name='Artist',
        labels={'Album':'Album', 'Play Count':'Play Count'},
        hover_data=['Album', 'Play Count'])
    fig.update_traces(hovertemplate='Album: %{x}<br>Play Count: %{y}<br>Artist: %{hovertext}<extra></extra>')
    fig.update_layout(title_text='Top 10 Albums')
    st.plotly_chart(fig)

    # Top Tracks
    top_tracks = filtered_data.groupby(['Artist', 'Track']).size().reset_index(name='Play Count')
    top_tracks = top_tracks.sort_values(by='Play Count', ascending=False).head(10)
    fig = px.bar(top_tracks, y='Play Count', x='Track',
             hover_name='Artist',
             labels={'Track':'Track', 'Play Count':'Play Count'},
             hover_data=['Track', 'Play Count'])
    fig.update_traces(hovertemplate='Track: %{x}<br>Play Count: %{y}<br>Artist: %{hovertext}<extra></extra>')
    fig.update_layout(title_text='Top 10 Tracks')
    st.plotly_chart(fig)

elif app_mode == "Scrobbles Over Time":
    add_bg_from_url(
    f"""
    \n# My Scrobbles Over Time:
    \n# 2010 to 2023
    \n - This chart displays my listens per month over the course of my listening history.
    """)
    scrobbles_over_time = data['DateTime'].dt.to_period('M').value_counts().sort_index()
    fig = px.line(scrobbles_over_time, x=scrobbles_over_time.index.to_timestamp(), y=scrobbles_over_time.values, labels={'x': 'Date', 'y': 'Scrobbles'})
    fig.update_traces(hovertemplate='Date: %{x}<br>Scrobbles: %{y}<extra></extra>')
    st.plotly_chart(fig)


elif app_mode == "3D Line Plot":
    add_bg_from_url(
    f"""
    \n# Listens Over Time - 3D Line Plot
    \n Each line represents a year with each point representing one month's listening total.
    """)
    # Create the 3D line plot
    plot_3d = create_3d_plot_3(data)  # Changed df to data

    # Display the plot in Streamlit
    st.plotly_chart(plot_3d)

elif app_mode == "Heatmap and Radial Bar":
    add_bg_from_url(
    f"""
    \n# When do I listen to music?
    \n## A Heatmap and Radial Bar Chart
    """)
    # Extract hour, day of the week, month, and year
    data['Hour'] = data['DateTime'].dt.hour
    data['Day'] = data['DateTime'].dt.day_name()
    data['Month'] = data['DateTime'].dt.month
    data['Year'] = data['DateTime'].dt.year

    # Create a dataframe for the heatmap and the radial bar chart
    daily_counts = data.groupby([data['DateTime'].dt.date, 'Hour']).size().reset_index(name='Count')
    daily_counts['Date'] = pd.to_datetime(daily_counts['DateTime'])
    hourly_counts = data.groupby(['Year', 'Month', 'Hour', 'Day']).size().reset_index(name='Count')

    # Filter by year and month
    years = sorted(list(data['Year'].unique()))
    months = sorted(list(data['Month'].unique()))

    selected_year = st.selectbox("Select a year:", years)
    selected_month = st.selectbox("Select a month:", months, format_func=lambda month: calendar.month_name[month])

    # Filter data for heatmap and radial bar chart
    filtered_daily_counts = daily_counts[(daily_counts['Date'].dt.year == selected_year) & (daily_counts['Date'].dt.month == selected_month)]
    filtered_hourly_counts = hourly_counts[(hourly_counts['Year'] == selected_year) & (hourly_counts['Month'] == selected_month)]

    # Create a heatmap
    heatmap = go.Figure(go.Heatmap(
        x=filtered_daily_counts['Hour'],
        y=filtered_daily_counts['Date'],
        z=filtered_daily_counts['Count'],
        colorscale='Reds',
        hovertemplate=(
            "Day: %{y|%B %d, %Y}<br>"
            "Hour: %{x}<br>"
            "Listens: %{z}<extra></extra>"
        )
    ))

    heatmap.update_layout(
        title="Listens by Day Heatmap",
        xaxis=dict(title="Hour of Day", tickmode='array', tickvals=list(range(1, 32)), showgrid=False),
        yaxis=dict(title="Day of the Month", showgrid=False),
    )

    st.plotly_chart(heatmap)

    # Create a radial bar chart
    radial_chart = go.Figure()

    colors = px.colors.qualitative.Plotly  # Use a predefined set of colors

    for i, day in enumerate(data['Day'].unique()):
        day_data = filtered_hourly_counts[filtered_hourly_counts['Day'] == day]
        radial_chart.add_trace(go.Barpolar(
            r=day_data['Count'],
            theta=day_data['Hour'] * 360 / 24,
            name=day,
            marker_color=colors[i % len(colors)],  # Cycle through the colors
            customdata=np.full(day_data['Hour'].shape, day),  # Add the day to the customdata
            hovertemplate="Day: %{customdata}<br>Hour: %{theta}:00<br>Listens: %{r}<extra></extra>"
        ))

    radial_chart.update_layout(
        title="Harmonic Rhythms: Active Listening Hours",
        polar=dict(angularaxis=dict(tickmode='array', tickvals=list(range(0, 360, 360 // 24)), ticktext=[f"{hour}:00" for hour in range(24)]))
    )

    st.plotly_chart(radial_chart)

elif app_mode == "WordCloud":
    add_bg_from_url(
    f"""
    # WordCloud
    """)

    # Group data by Year and Artist
    artist_counts = data.groupby([data['DateTime'].dt.year, 'Artist']).size().reset_index(name='Count')
    artist_counts.rename(columns={'DateTime': 'Year'}, inplace=True)

    # Create a multi-choice dropdown menu for selecting years
    years = list(range(data['DateTime'].dt.year.min(), data['DateTime'].dt.year.max() + 1))
    selected_years = st.sidebar.multiselect("Select years:", years, default=years)

    if selected_years:
        # Filter data for the selected years
        filtered_artist_counts = artist_counts[artist_counts['Year'].isin(selected_years)]

        # Group the filtered data by artist
        artist_group = filtered_artist_counts.groupby('Artist').agg({'Count': 'sum'}).reset_index()

        # Generate the word cloud
        wc = WordCloud(width=800, height=400, max_words=200, background_color='white')
        wc.generate_from_frequencies({row['Artist']: row['Count'] for _, row in artist_group.iterrows()})

        # Display the word cloud
        plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("Please select at least one year.")

elif app_mode == "Bar Chart Race":
    add_bg_from_url(
    f"""
    # Total Listens - Bar Chart Race
    """)
    # Create a selectbox
    exclude_artist = st.selectbox('Exclude Artist from Top Artists', options=['None', 'The Strokes'])

    if exclude_artist == 'The Strokes':
    # Filter out "The Strokes"
        filtered_data = data.loc[data['Artist'] != 'The Strokes']
    else:
        filtered_data = data

    top_artists = filtered_data['Artist'].value_counts().head(20).index.tolist()
    data_top_artists = filtered_data[filtered_data['Artist'].isin(top_artists)]

    data_top_artists.loc[:, 'Year_Month'] = data_top_artists['DateTime'].dt.to_period('M').astype(str)
    year_month_range = pd.period_range(start=data_top_artists['Year_Month'].min(), end=data_top_artists['Year_Month'].max(), freq='M')
    year_month_list = year_month_range.strftime('%Y-%m').tolist()

    all_combinations = pd.DataFrame(list(itertools.product(top_artists, year_month_list)), columns=['Artist', 'Year_Month'])
    year_month_counts = data_top_artists.groupby(['Artist', 'Year_Month']).size().reset_index(name='Count')
    merged_data = all_combinations.merge(year_month_counts, on=['Artist', 'Year_Month'], how='left')
    merged_data['Count'].fillna(0, inplace=True)
    merged_data['Running_Total'] = merged_data.groupby('Artist')['Count'].cumsum()
    merged_data.loc[:, 'Year_Month'] = merged_data['Year_Month'].astype(str)

    index = pd.MultiIndex.from_product([top_artists, year_month_range], names=['Artist', 'Year_Month'])
    new_df = pd.DataFrame(index=index).reset_index()
    new_df.loc[:, 'Year_Month'] = new_df['Year_Month'].astype(str)

    new_df = pd.merge(new_df, merged_data, on=['Artist', 'Year_Month'], how='left').fillna(0)
    new_df['Running_Total'] = new_df.groupby('Artist')['Count'].cumsum()

    animation_duration = 250
    frames = []
    #new_df['Year_Month'] = pd.to_datetime(new_df['Year_Month']).dt.to_period('M')
    new_df = new_df.sort_values(['Year_Month', 'Running_Total'], ascending=[True, False])
    new_df = new_df.groupby('Year_Month').head(10)

    # Define color generator
    color_gen = lambda: '#%02X%02X%02X' % (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    artist_colors = {artist: color_gen() for artist in new_df['Artist'].unique()}

    # Create an empty list to store the frames and a slider dictionary
    frames = []
    slider_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Date: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': animation_duration, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # Iterate through each period and create a frame
    for year_month in new_df['Year_Month'].unique():
        df_period = new_df[new_df['Year_Month'] == year_month]

    # Assign the color to each artist for this frame
        df_period = df_period.copy()
        df_period['color'] = df_period['Artist'].apply(lambda artist: artist_colors[artist])

    # Convert the DataFrame to a bar chart
        frame = go.Frame(
            data=[
                go.Bar(
                    x=df_period['Running_Total'],
                    y=df_period['Artist'],
                    text=df_period['Running_Total'],
                    textposition='inside',
                    insidetextanchor='start',
                    orientation='h',
                    marker={'color': df_period['color']},
                    hovertemplate='<b>%{y}</b><br>%{x} songs<extra></extra>',
                )
            ],
            layout=go.Layout(
                title_text=f'Song Count by Artist: {year_month.strftime("%B, %Y")}',
                xaxis_title="Cumulative Song Count",
                yaxis_title="Artist",
                plot_bgcolor='rgb(211, 211, 211)',  # Make chart's background grey
                showlegend=False,
                height=600,
                width=728,
                bargap=0.15,
                bargroupgap=0.1,
                yaxis={'categoryorder': 'total ascending'},
            ),
            name=year_month.strftime("%B, %Y"),
        )

        frames.append(frame)

        slider_step = {"args": [
            [year_month.strftime("%B, %Y")],
            {"frame": {"duration": animation_duration, "redraw": True},
            "mode": "immediate",
            "transition": {"duration": animation_duration}}
            ],
            "label": year_month.strftime("%B, %Y"),
            "method": "animate",
        }
        slider_dict["steps"].append(slider_step)

    # Create the layout for the bar chart
    bar_chart_layout = go.Layout(
        title_text="Bar Chart Race of Song Count by Artist Over Time",
        xaxis_title="Cumulative Song Count",
        yaxis_title="Artist",
        showlegend=False,
        height=600,
        width=728,
        bargap=0.15,
        bargroupgap=0.1,
        plot_bgcolor='rgb(211, 211, 211)',  # Make chart's background grey
        yaxis={'categoryorder': 'total ascending'},
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": animation_duration, "redraw": True}, "fromcurrent": True, "transition": {"duration": animation_duration}}],
                "label": "Play",
                "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }],
        sliders=[slider_dict],
    )

    # Create the bar chart
    bar_chart = go.Figure(
        data=frames[0]['data'],
        layout=bar_chart_layout,
        frames=frames,
    )

    # Display the bar chart in your Streamlit dashboard
    st.plotly_chart(bar_chart)

elif app_mode == "Scatter Plot":
    add_bg_from_url(
    f"""
    # Scatter Plot
    """)
    fig = px.scatter(data, x='DateTime', y=(data['DateTime'].dt.hour * 60 + data['DateTime'].dt.minute)%1440,
                     hover_data=['Artist', 'Track', data['DateTime'].dt.time])

    fig.update_traces(
        hovertemplate="Artist: %{customdata[0]}<br>Track: %{customdata[1]}<br>Date: %{x|%B %d, %Y}<br>Time: %{customdata[2]}<extra></extra>",
        marker=dict(size=2)  # Adjust the size here
    )

    fig.update_layout(
        title="Tracks Listened to Over Time",
        xaxis_title="Date",
        yaxis_title="Time of Day",
        yaxis=dict(tickvals=list(range(0, 1381, 60)) + [1440], ticktext=[f"{i}:00" for i in range(24)] + ["24:00"], autorange="reversed"),  # This will make the 24:00 at the bottom and 00:00 at the top
    )

    st.plotly_chart(fig)
