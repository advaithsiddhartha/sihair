import streamlit as st
import requests
import base64
import api
import lat
import model
from streamlit_navigation_bar import st_navbar

# Function to load and encode image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Set page layout to wide
st.set_page_config(layout="wide")

# Load the banner image
banner_image_base64 = get_base64_image('./1.png')  # Make sure the path is correct

# Add a banner-type header with a colored stripe background using the base64-encoded image
st.markdown(f"""
    <style>
    .banner-image {{
        width: 100%;
        height: auto;
    }}
    </style>
    <div>
        <img src="data:image/png;base64,{banner_image_base64}" class="banner-image">
    </div>
""", unsafe_allow_html=True)

# Navbar for page navigation
page = st_navbar(["API WINDOW", "LATITUDE AND LONGITUDE TRACING", "LSTM MODEL WINDOW", "LOGIN", "PROFILE"])

# Title and instructions
st.title("SIH - Team Sirius - 1734")
st.success("Select the navigation item to switch between pages")

# Page routing logic based on navbar selection
if page == "API WINDOW":
    # Dropdown for selecting city
    city_options = ["Hyderabad", "Mumbai", "Chennai", "Delhi", "Ahmedabad", "Visakhapatnam", "Bengaluru"]
    city = st.selectbox("Select a City", city_options)

    # Dictionary to map city names to image filenames
    city_images = {
        "Hyderabad": "hyd.png",
        "Mumbai": "mum.png",
        "Chennai": "chn.png",
        "Delhi": "del.png",
        "Ahmedabad": "ahm.png",
        "Visakhapatnam": "vskp.png",
        "Bengaluru": "bng.png"
    }

    # Display the corresponding image when a city is selected
    if city:
        image_path = city_images.get(city)
        if image_path:
            st.image(image_path, caption=f"Image of {city}", use_column_width=True)

    # Add your API logic to get air quality data
    def get_air_quality_data(city, state, country):
        BASE_URL = "your_base_url_here"  # Replace with your actual base URL
        API_KEY = "your_api_key_here"      # Replace with your actual API key
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

# Ensure each module (`api`, `lat`, `model`) has a `show_page()` function to render the respective page.
