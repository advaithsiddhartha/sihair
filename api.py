import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def show_page():

    # IQAir API key and endpoint
    API_KEY = '6c0a1d0c-dbb3-4156-8488-ba25c5b2fb8b'  # Replace with your actual API key
    BASE_URL = 'http://api.airvisual.com/v2'

    # Fallback list of states and cities for India
    INDIAN_STATES_CITIES = {
        "Delhi": ["Delhi", "New Delhi"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
        "Karnataka": ["Bangalore", "Mysore"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
        "Uttar Pradesh": ["Lucknow", "Kanpur"],
        "West Bengal": ["Kolkata"],
        "Gujarat": ["Ahmedabad", "Surat"],
        "Rajasthan": ["Jaipur"],
        "Telangana": ["Hyderabad"],
        "Kerala": ["Thiruvananthapuram", "Kochi"]
    }

    # Function to get air quality data for a specific city
    def get_air_quality_data(city, state, country):
        url = f"{BASE_URL}/city?city={city}&state={state}&country={country}&key={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return data['data']
            else:
                st.error(f"Error: {data.get('data', {}).get('message', 'Unknown error')}")
        else:
            st.error("Failed to fetch data")
        return None

    # Function to simulate historical data (since API doesn't provide it)
    def simulate_historical_data(current_aqi, current_temp):
        np.random.seed(42)  # For reproducibility
        dates = [datetime.now() - timedelta(days=i) for i in range(30)][::-1]
        historical_aqi = current_aqi + np.random.normal(0, 20, size=30).astype(int)
        historical_aqi = np.clip(historical_aqi, 0, 500)  # AQI range
        historical_temp = current_temp + np.random.normal(0, 5, size=30).astype(int)
        historical_temp = np.clip(historical_temp, -10, 50)  # Temperature range
        return dates, historical_aqi, historical_temp

    # Streamlit app layout
    st.title("Air Quality Dashboard - India")
    st.info("`TEAM SIRIUS _ SMART INDIA HACKATHON`")
    st.success("R. Krishna Advaith Siddhartha , R. Bhoomika  , K.R. Nakshathra    ,  S. Ravi Teja  ,   V. Subhash ,  M. Abhinav")
    # Use hardcoded fallback for states and cities
    states = list(INDIAN_STATES_CITIES.keys())

    # Dropdown for state selection
    state = st.selectbox("Select a State", states)

    # Dropdown for city selection based on selected state
    cities = INDIAN_STATES_CITIES[state]
    city = st.selectbox("Select a City", cities)

    # Fetch data button
    if st.button("Get Data"):
        data = get_air_quality_data(city, state, "India")
        
        if data:
            # Extract necessary information
            aqi = data['current']['pollution']['aqius']
            main_pollutant = data['current']['pollution']['mainus']
            temp = data['current']['weather']['tp']
            humidity = data['current']['weather']['hu']
            pressure = data['current']['weather']['pr']
            last_update = data['current']['pollution']['ts']

                # Display basic data
            st.subheader(f"Air Quality in {city}, {state}, India")

            # CSS for custom card layout
            card_style = """
            <style>
            .card {
                padding: 20px;
                margin: 10px;
                background-color: #201E43;
                border-radius: 15px;
                box-shadow: 
                /* Existing shadow */
                0px 2px 15px rgba(255,255,255, 0.5); /* New grey shadow */
                color: #333333;
                animation: float 3s ease-in-out infinite; /* Apply floating effect */
            }

            @keyframes float {
                0%, 100% {
                    transform: translateY(0); /* Original position */
                }
                50% {
                    transform: translateY(-10px); /* Move up */
                }
            }

            .card h3 {
                font-size: 1.5em;
                margin-bottom: 10px;
                color: #FFDA76;
            }
            .card p {
                color:white;
                font-size: 1.2em;
                margin: 5px 0;
            }
            .card-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-evenly;
            }
            .card-wrapper {
                flex: 1;
                min-width: 300px;
                max-width: 300px;
                margin: 10px;
            }
            </style>
            """

            # Display cards using HTML
            card_html = f"""{card_style}
            <div class="card-container">
                <div class="card-wrapper">
                    <div class="card">
                        <h3>Last Updated</h3>
                        <p>{datetime.strptime(last_update, '%Y-%m-%dT%H:%M:%S.%fZ')}</p>
                    </div>
                </div>
                <div class="card-wrapper">
                    <div class="card">
                        <h3>Air Quality Index </h3>
                        <p>{aqi}</p>
                    </div>
                </div>
                <div class="card-wrapper">
                    <div class="card">
                        <h3>Main Pollutant</h3>
                        <p>{main_pollutant}</p>
                    </div>
                </div>
                <div class="card-wrapper">
                    <div class="card">
                        <h3>Temperature</h3>
                        <p>{temp}°C</p>
                    </div>
                </div>
                <div class="card-wrapper">
                    <div class="card">
                        <h3>Humidity</h3>
                        <p>{humidity}%</p>
                    </div>
                </div>
                <div class="card-wrapper">
                    <div class="card">
                        <h3>Pressure</h3>
                        <p>{pressure} hPa</p>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Simulate historical data
            dates, historical_aqi, historical_temp = simulate_historical_data(aqi, temp)

            # --- Visualizations ---

            st.subheader("Visualizations")

            # Layout with multiple columns
            col1, col2 = st.columns(2)

            with col1:
                # 1. Gauge Chart: AQI
                aqi_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=aqi,
                    title={'text': "AQI (US)"},
                    gauge={'axis': {'range': [0, 500]},
                        'bar': {'color': "green" if aqi < 100 else "orange" if aqi < 200 else "red"}},
                ))
                aqi_gauge.update_layout(height=300, width=400)
                st.plotly_chart(aqi_gauge, use_container_width=True)

            with col2:
                # 2. Gauge Chart: Temperature
                temp_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=temp,
                    title={'text': "Temperature (°C)"},
                    gauge={'axis': {'range': [-10, 50]},
                        'bar': {'color': "blue"}}
                ))
                temp_gauge.update_layout(height=300, width=400)
                st.plotly_chart(temp_gauge, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                # 3. Gauge Chart: Humidity
                humidity_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=humidity,
                    title={'text': "Humidity (%)"},
                    gauge={'axis': {'range': [0, 100]},
                        'bar': {'color': "purple"}}
                ))
                humidity_gauge.update_layout(height=300, width=400)
                st.plotly_chart(humidity_gauge, use_container_width=True)

            with col4:
                # 4. Gauge Chart: Pressure
                pressure_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pressure,
                    title={'text': "Pressure (hPa)"},
                    gauge={'axis': {'range': [900, 1100]},
                        'bar': {'color': "teal"}}
                ))
                pressure_gauge.update_layout(height=300, width=400)
                st.plotly_chart(pressure_gauge, use_container_width=True)

            # 5. Bar Chart: AQI and Humidity Over Time
            st.markdown("### AQI and Humidity Over the Past 30 Days")
            bar_fig = go.Figure(data=[
                go.Bar(name='AQI', x=dates, y=historical_aqi, marker_color='indianred'),
                go.Bar(name='Humidity', x=dates, y=[humidity]*30, marker_color='lightsalmon')
            ])
            bar_fig.update_layout(barmode='group', xaxis_title='Date', yaxis_title='Values', height=500)
            st.plotly_chart(bar_fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1: 
                # 6. Pie Chart: Pollutant Breakdown (Simulated)
                st.markdown("### Pollutant Breakdown")
                pollutants = ['PM2.5', 'PM10', 'CO', 'O3', 'NO2']
                pollutant_values = [40, 30, 15, 10, 5]  # Simulated values
                pie_fig = px.pie(names=pollutants, values=pollutant_values, title='Proportion of Pollutants')
                pie_fig.update_traces(textposition='inside', textinfo='percent+label')
                pie_fig.update_layout(height=500)
                st.plotly_chart(pie_fig, use_container_width=True)

            st.header("3D Scatter Plot and 3D Surface Plot")


            with col2 : 
                st.markdown("### Temperature Heat Map")
                # Simulate a 10x10 grid of temperatures
                heat_map_data = np.random.normal(loc=temp, scale=5, size=(10, 10)).astype(int)
                heat_map_df = pd.DataFrame(heat_map_data, columns=[f'Col {i+1}' for i in range(10)], index=[f'Row {i+1}' for i in range(10)])
                heatmap_fig = px.imshow(heat_map_df, title='Temperature Heat Map', color_continuous_scale='Viridis')
                heatmap_fig.update_layout(height=600)
                st.plotly_chart(heatmap_fig, use_container_width=True)
            # Create two columns
            col1, col2 = st.columns(2)

            # 3D Scatter Plot in the first column
            with col1:
                st.markdown("#### 3D Scatter Plot: Temperature, Humidity, AQI")
                x = np.random.rand(50) * 40  # Temperature range from 0°C to 40°C
                y = np.random.rand(50) * 100  # Humidity range from 0% to 100%
                z = np.random.rand(50) * 500  # AQI range from 0 to 500

                scatter_fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', 
                                                        marker=dict(size=5, color=z, colorscale='Viridis'))])

                scatter_fig.update_layout(scene=dict(xaxis_title='Temperature (°C)', 
                                                    yaxis_title='Humidity (%)', 
                                                    zaxis_title='AQI'))
                st.plotly_chart(scatter_fig)


            # 3D Surface Plot in the second column
            with col2:
                st.markdown("#### 3D Surface Plot: Simulated AQI Over Time")
                x_surface = np.linspace(0, 29, 30)  # Simulated 30-day period
                y_surface = np.linspace(1, 10, 10)  # Simulated 10 cities
                x_surface, y_surface = np.meshgrid(x_surface, y_surface)
                z_surface = np.random.rand(10, 30) * 500  # Simulated AQI values

                surface_fig = go.Figure(data=[go.Surface(z=z_surface, x=x_surface, y=y_surface, colorscale='Rainbow')])
                surface_fig.update_layout(scene=dict(xaxis_title='Days', 
                                                    yaxis_title='City Index', 
                                                    zaxis_title='AQI'))
                st.plotly_chart(surface_fig)
            col1 , col2  = st.columns(2)
            with col1:
                st.header("3D Line Plot: Daily AQI Trend")
                t = np.linspace(0, 24, 100)  # Time of day in hours
                x_line = np.linspace(0, 30, 100)  # Days
                y_line = t
                z_line = np.random.rand(100) * 500  # Simulated AQI

                line_fig = go.Figure(data=[go.Scatter3d(x=x_line, y=y_line, z=z_line, mode='lines', 
                                                        line=dict(width=4, color=z_line, colorscale='Cividis'))])
                line_fig.update_layout(scene=dict(xaxis_title='Days', 
                                                yaxis_title='Time of Day', 
                                                zaxis_title='AQI'))
                st.plotly_chart(line_fig)
            with col2 : 
                st.markdown("### Air Quality Index Distribution")
                hist_fig = px.histogram(
                x=historical_aqi,
                nbins=20,
                title='AQI Distribution Over 30 Days',
                labels={'x': 'AQI', 'y': 'Frequency'},
                color_discrete_sequence=['#ff5b00']  # Set your desired color
                )

                # Optional: Add a hover template for more information
                hist_fig.update_traces(hovertemplate='AQI: %{x}<br>Frequency: %{y}', marker=dict(line=dict(color='#000000', width=1)))

                # Update layout settings
                hist_fig.update_layout(height=500)

                # Display the histogram
                st.plotly_chart(hist_fig, use_container_width=True)
        

            # 7. Bollinger Bands for AQI (Simulated)
            st.markdown("### Bollinger Bands for AQI")
            bb_fig = go.Figure()

            bb_fig.add_trace(go.Scatter(
                x=dates, y=historical_aqi, mode='lines', name='AQI', line=dict(color='blue')
            ))

            # Calculate moving average and standard deviation
            df_bb = pd.DataFrame({'Date': dates, 'AQI': historical_aqi})
            df_bb['MA'] = df_bb['AQI'].rolling(window=5).mean()
            df_bb['STD'] = df_bb['AQI'].rolling(window=5).std()
            df_bb['Upper'] = df_bb['MA'] + (df_bb['STD'] * 2)
            df_bb['Lower'] = df_bb['MA'] - (df_bb['STD'] * 2)

            bb_fig.add_trace(go.Scatter(
                x=df_bb['Date'], y=df_bb['Upper'], mode='lines', name='Upper Band', line=dict(color='lightblue', dash='dash')
            ))
            bb_fig.add_trace(go.Scatter(
                x=df_bb['Date'], y=df_bb['Lower'], mode='lines', name='Lower Band', line=dict(color='lightblue', dash='dash'),
                fill='tonexty', fillcolor='rgba(173,216,230,0.2)'
            ))

            bb_fig.update_layout(height=500, title='Bollinger Bands for AQI', xaxis_title='Date', yaxis_title='AQI')
            st.plotly_chart(bb_fig, use_container_width=True)

            


            # 8. Line Graph: Temperature Over Time
            st.markdown("### Temperature Over the Past 30 Days")
            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(
                x=dates, y=historical_temp, mode='lines+markers', name='Temperature', line=dict(color='orange')
            ))
            line_fig.update_layout(title='Temperature Over Time', xaxis_title='Date', yaxis_title='Temperature (°C)', height=500)
            st.plotly_chart(line_fig, use_container_width=True)

            # 9. Heat Map: Temperature Distribution (Simulated)
            

            # 10. Scatter Plot: Temperature vs AQI
            st.markdown("### Scatter Plot: Temperature vs AQI")
            scatter_fig = px.scatter(
                x=historical_temp,
                y=historical_aqi,
                labels={'x': 'Temperature (°C)', 'y': 'AQI'},
                title='Temperature vs AQI',
                trendline='ols'
            )
            scatter_fig.update_traces(marker=dict(size=12, color='rgba(152, 0, 0, .8)', line=dict(width=2, color='DarkSlateGrey')))
            scatter_fig.update_layout(height=500)
            st.plotly_chart(scatter_fig, use_container_width=True)

            # 11. Area Chart: Historical AQI
            st.markdown("### Historical Air Quality Index Over Time (Area Chart)")
            area_fig = go.Figure()
            area_fig.add_trace(go.Scatter(
                x=dates, y=historical_aqi, fill='tozeroy', name='AQI', line=dict(color='green')
            ))
            area_fig.update_layout(title='Historical AQI Over Time', xaxis_title='Date', yaxis_title='AQI', height=500)
            st.plotly_chart(area_fig, use_container_width=True)

            # 12. Map: Display AQI on Map for Selected Cities (Simulated)
            st.markdown("### Air Quality Index Map")
            map_data = pd.DataFrame({
                'lat': [28.7041, 19.0760, 12.9716, 13.0827, 25.5941, 22.5726, 23.0225, 26.9124, 17.3850, 10.8505],
                'lon': [77.1025, 72.8777, 77.5946, 80.2707, 85.1376, 88.3639, 72.5714, 75.7873, 78.4867, 76.2711],
                'city': ["Delhi", "Mumbai", "Bangalore", "Chennai", "Lucknow", "Kolkata", "Ahmedabad", "Jaipur", "Hyderabad", "Kochi"],
                'aqi': np.random.randint(50, 300, size=10)  # Simulated AQI values
            })
            map_fig = px.scatter_mapbox(
                map_data,
                lat="lat",
                lon="lon",
                hover_name="city",
                hover_data=["aqi"],
                color="aqi",
                size="aqi",
                color_continuous_scale=px.colors.cyclical.IceFire,
                size_max=15,
                zoom=3
            )
            map_fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":0,"l":0,"b":0},
                height=600,
                title='AQI Levels in Major Indian Cities'
            )
            st.plotly_chart(map_fig, use_container_width=True)

            # 13. Histogram: AQI Distribution
            
        else:
            st.error("No data found for the selected location.")