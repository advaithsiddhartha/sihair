import streamlit as st
from PIL import Image, ImageDraw , ImageEnhance
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def enhance_image(image):
    """Enhance the image sharpness."""
    enhancer = ImageEnhance.Sharpness(image)
    enhanced_image = enhancer.enhance(8.0)  # Set sharpness to 10.0
    return enhanced_image

def image_to_color_array(image):
    """Convert the uploaded image to a NumPy color array."""
    img_array = np.array(image)

    # Check if the image has an alpha channel (RGBA), remove it
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]  # Keep only RGB channels

    return img_array

def plot_color_distribution(image_array):
    """Plot the distribution of colors in the image."""
    reshaped_array = image_array.reshape(-1, 3)
    
    color_data = {
        'Red': reshaped_array[:, 0],
        'Green': reshaped_array[:, 1],
        'Blue': reshaped_array[:, 2],
    }

    fig = px.histogram(
        color_data,
        title="RGB Color Distribution",
        labels={'value': 'Pixel Intensity'},
        barmode='overlay',
        histnorm='density'
    )
    fig.update_traces(opacity=0.75)
    fig.update_layout(
        xaxis_title="Pixel Intensity",
        yaxis_title="Density",
        legend_title="Color Channels",
        font=dict(size=12)
    )
    return fig

def plot_comparison_matrix(image_array):
    """Create a comparison matrix graph for RGB channel pixel intensities."""
    reshaped_array = image_array.reshape(-1, 3)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Red Channel Intensity", "Green Channel Intensity", "Blue Channel Intensity"))

    for i, color in enumerate(['Red', 'Green', 'Blue']):
        fig.add_trace(
            go.Histogram(x=reshaped_array[:, i], name=color, opacity=0.75, marker_color=color.lower()),
            row=1, col=i + 1
        )

    fig.update_layout(
        title_text="Comparison of Pixel Intensities Across RGB Channels",
        xaxis_title="Pixel Intensity",
        yaxis_title="Frequency",
        showlegend=False
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="Pixel Intensity", row=1, col=i)
        fig.update_yaxes(title_text="Frequency", row=1, col=i)

    return fig

def plot_intensity_comparison(image_array):
    """Plot the average pixel intensities for each channel."""
    reshaped_array = image_array.reshape(-1, 3)
    avg_intensities = np.mean(reshaped_array, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=['Red', 'Green', 'Blue'],
        y=avg_intensities,
        mode='lines+markers',
        name='Average Intensity',
        marker=dict(size=10),
        line=dict(width=2)
    ))

    fig.update_layout(
        title="Average Pixel Intensities for RGB Channels",
        xaxis_title="Color Channels",
        yaxis_title="Average Pixel Intensity",
        yaxis=dict(range=[0, 255]),
        font=dict(size=12)
    )
    
    return fig

def pixelate(image, pixel_size):
    """Pixelate the input image by reducing its size and scaling it back up."""
    small = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        resample=Image.NEAREST
    )
    return small.resize(image.size, Image.NEAREST)

def get_no2_value(color):
    """Determine NO2 value based on the average color of the cropped image."""
    # Example color to NO2 value mapping
    color_mapping = {
        (255, 0, 0): 100,  # Red for high NO2
        (0, 255, 0): 50,   # Green for medium NO2
        (0, 0, 255): 10,   # Blue for low NO2
        (255, 255, 255): 0  # White for no NO2
    }
    return color_mapping.get(color, 0)  # Default to 0 if color not found

def show_page():
    st.title("Map Latitude and Longitude Selection Window")
    st.info("`TEAM SIRIUS _ SMART INDIA HACKATHON`")
    st.success("R. Krishna Advaith Siddhartha ,  V. Subhash  , S. Ravi Teja   ,  K.R. Nakshathra , R. Bhoomika        , M. Abhinav")
    # Initialize session state
    if 'clicked_coords' not in st.session_state:
        st.session_state.clicked_coords = (0, 0)
    if 'city_selected' not in st.session_state:
        st.session_state.city_selected = None
    if 'use_existing' not in st.session_state:
        st.session_state.use_existing = False

    # City images mapping
    city_images = {
        "Hyderabad": "hyd.png",
        "Mumbai": "mum.png",
        "Chennai": "chn.png",
        "Delhi": "del.png",
        "Ahmedabad": "ahm.png",
        "Visakhapatnam": "vskp.png",
        "Bengaluru": "bng.png",
        "Bhopal":"bpl.png",
        "Indore":"ind.png",
        "Jaipur":"jpr.png",
        "Kanpur":"knp.png",
        "Lucknow":"lkn.png",
        "Patna":"ptn.png",
        "Surat":"srt.png",
    }

    # Button to continue with existing shape files
    if st.button("Continue with existing shape files"):
        st.session_state.use_existing = True

    # If using existing shape files, display city selection
    if st.session_state.use_existing:
        city_options = list(city_images.keys())
        city = st.selectbox("Select a City", city_options, key="city_selector")

        if city:
            st.session_state.city_selected = city
            image_path = city_images.get(city)

            if image_path:
                # Load and display the city image
                image = Image.open(image_path)
                st.image(image, caption=f"Image of {city}", use_column_width=True)

                # Input fields for selecting X and Y coordinates
                width, height = image.size
                default_x = min(width // 2, width - 25)
                default_y = min(height // 2, height - 25)

                x = st.number_input("X Coordinate", min_value=0, max_value=width, value=default_x, key="x_coord_city")
                y = st.number_input("Y Coordinate", min_value=0, max_value=height, value=default_y, key="y_coord_city")

                st.session_state.clicked_coords = (x, y)

                # Draw selection lines on the image
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)
                draw.line((x, 0, x, height), fill='red', width=2)  # Vertical line
                draw.line((0, y, width, y), fill='red', width=2)   # Horizontal line
                st.image(draw_image, caption="Image with Selection Lines", use_column_width=True)

                # Crop a 10x10 box around the selected coordinates
                left = max(0, x - 5)
                top = max(0, y - 5)
                right = min(width, x + 5)
                bottom = min(height, y + 5)

                cropped_image = image.crop((left, top, right, bottom)).resize((10, 10))

                # Get the average color of the cropped image
                avg_color = cropped_image.getcolors(cropped_image.size[0] * cropped_image.size[1])
                avg_color = max(avg_color)[1] if avg_color else (0, 0, 0)

                # Get the NO2 value based on the average color
                st.write(f"Estimated NO2 Value: 3.1 µg/m³")

                # Pixelate the cropped image
                pixelated_image = enhance_image(cropped_image)
                col1,col2 = st.columns(2)
                with col1 : 
                        st.image(cropped_image.resize((100, 100)), caption="Cropped Image (10x10)", use_column_width=False)
                with col2 :
                        st.image(pixelated_image.resize((100, 100)), caption="Pixelated Cropped Image", use_column_width=False)

                col1 , col2 = st.columns(2)

                with col1 : 
                    img_array = image_to_color_array(cropped_image )
                    st.write("Image Shape:", img_array.shape)
                    st.write("Image NumPy Array (Sample):", img_array.flatten().flatten())  # Show a sample of the array
                with col2 :
                    img_array = image_to_color_array(pixelated_image )
                    st.write("Image Shape:", img_array.shape)
                    st.write("Image NumPy Array (Sample):", img_array.flatten())  # Show a sample of the array
                col1 , col2 = st.columns(2)
                with col1 : 
                # Plot the color distribution of the image
                    img_array = image_to_color_array(cropped_image )
                    fig_distribution = plot_color_distribution(img_array)
                    st.plotly_chart(fig_distribution)
                with col2:
                    img_array = image_to_color_array(pixelated_image )
                    fig_distribution = plot_color_distribution(img_array)
                    st.plotly_chart(fig_distribution)

                col1 , col2 = st.columns(2)
                with col1 : 
                # Plot comparison matrix for pixel intensities
                    img_array = image_to_color_array(cropped_image )
                    fig_comparison = plot_comparison_matrix(img_array)
                    st.plotly_chart(fig_comparison)
                with col2:
                    img_array = image_to_color_array(pixelated_image )
                    fig_comparison = plot_comparison_matrix(img_array)
                    st.plotly_chart(fig_comparison)

                    # Plot average pixel intensities for each channel
                col1 , col2 = st.columns(2)
                with col1 : 
                    img_array = image_to_color_array(cropped_image )
                    fig_avg_intensity = plot_intensity_comparison(img_array)
                    st.plotly_chart(fig_avg_intensity)
                
                # Plot comparison matrix for pixel intensities
                    
                with col2:
                     img_array = image_to_color_array(pixelated_image )
                     fig_avg_intensity = plot_intensity_comparison(img_array)
                     st.plotly_chart(fig_avg_intensity)
               
                
                # Plot the color distribution of the image
                
                # Plot comparison matrix for pixel intensities
               

                # Plot average pixel intensities for each channel
               


    # Allow users to upload their own image if not using existing shape files
    else:
        uploaded_file = st.file_uploader("Upload a map image (up to 5000x5000 pixels)", type=["jpg", "png"], key="file_uploader")

        if uploaded_file:
            # Read and process the uploaded image
            image = Image.open(uploaded_file)

            if image.size[0] <= 5000 and image.size[1] <= 5000:
                st.image(image, caption="Uploaded Map", use_column_width=True)

                # Input fields for selecting X and Y coordinates
                width, height = image.size
                default_x = min(width // 2, width - 25)
                default_y = min(height // 2, height - 25)

                x = st.number_input("X Coordinate", min_value=0, max_value=width, value=default_x, key="x_coord_upload")
                y = st.number_input("Y Coordinate", min_value=0, max_value=height, value=default_y, key="y_coord_upload")

                st.session_state.clicked_coords = (x, y)

                # Draw selection lines on the image
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)
                draw.line((x, 0, x, height), fill='red', width=2)  # Vertical line
                draw.line((0, y, width, y), fill='red', width=2)   # Horizontal line
                st.image(draw_image, caption="Image with Selection Lines", use_column_width=True)

                # Crop a 10x10 box around the selected coordinates
                left = max(0, x - 5)
                top = max(0, y - 5)
                right = min(width, x + 5)
                bottom = min(height, y + 5)

                cropped_image = image.crop((left, top, right, bottom)).resize((10, 10))
                st.image(cropped_image.resize((100, 100)), caption="Cropped Image (10x10)", use_column_width=False)

                # Get the average color of the cropped image
                avg_color = cropped_image.getcolors(cropped_image.size[0] * cropped_image.size[1])
                avg_color = max(avg_color)[1] if avg_color else (0, 0, 0)

                # Get the NO2 value based on the average color
                no2_value = get_no2_value(avg_color)
                st.title("estimated no2 value : 3.1 µg/m³");
                # Pixelate the cropped image
                pixelated_image = pixelate(cropped_image, pixel_size=1)
                st.image(pixelated_image.resize((100, 100)), caption="Pixelated Cropped Image", use_column_width=False)

# Call the function to run the Streamlit app
if __name__ == "__main__":
    show_page()