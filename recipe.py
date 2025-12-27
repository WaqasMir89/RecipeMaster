import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
from colorharmonies import Color, complementaryColor, analogousColor
import pandas as pd
from io import BytesIO
from PIL import Image
import colorsys


COMMON_CSS3_COLORS = {
    'aliceble': (240,248,255),
    'antiquewhite': (250,235,215),
    'aqua': (0,255,255),
    'aquamarine': (127,255,212),
    'azure': (240,255,255),
    'beige': (245,245,220),
    'bisque': (255,228,196),
    'black': (0,0,0),
    'blanchedalmond': (255,235,205),
    'blue': (0,0,255),
    'blueviolet': (138,43,226),
    'brown': (165,42,42),
    'burlywood': (222,184,135),
    'cadetblue': (95,158,160),
    'chartreuse': (127,255,0),
    'chocolate': (210,105,30),
    'coral': (255,127,80),
    'cornflowerblue': (100,149,237),
    'cornsilk': (255,248,220),
    'crimson': (220,20,60),
    'cyan': (0,255,255),
    'darkblue': (0,0,139),
    'darkcyan': (0,139,139),
    'darkgoldenrod': (184,134,11),
    'darkgray': (169,169,169),
    'darkgreen': (0,100,0),
    'darkkhaki': (189,183,107),
    'darkmagenta': (139,0,139),
    'darkolivegreen': (85,107,47),
    'darkorange': (255,140,0),
    'darkorchid': (153,50,204),
    'darkred': (139,0,0),
    'darksalmon': (233,150,122),
    'darkseagreen': (143,188,143),
    'darkslateblue': (72,61,139),
    'darkslategray': (47,79,79),
    'darkturquoise': (0,206,209),
    'darkviolet': (148,0,211),
    'deeppink': (255,20,147),
    'deepskyblue': (0,191,255),
    'dimgray': (105,105,105),
    'dodgerblue': (30,144,255),
    'firebrick': (178,34,34),
    'floralwhite': (255,250,240),
    'forestgreen': (34,139,34),
    'fuchsia': (255,0,255),
    'gainsboro': (220,220,220),
    'ghostwhite': (248,248,255),
    'gold': (255,215,0),
    'goldenrod': (218,165,32),
    'gray': (128,128,128),
    'green': (0,128,0),
    'greenyellow': (173,255,47),
    'honeydew': (240,255,240),
    'hotpink': (255,105,180),
    'indianred': (205,92,92),
    'indigo': (75,0,130),
    'ivory': (255,255,240),
    'khaki': (240,230,140),
    'lavender': (230,230,250),
    'lavenderblush': (255,240,245),
    'lawngreen': (124,252,0),
    'lemonchiffon': (255,250,205),
    'lightblue': (173,216,230),
    'lightcoral': (240,128,128),
    'lightcyan': (224,255,255),
    'lightgoldenrodyellow': (250,250,210),
    'lightgray': (211,211,211),
    'lightgreen': (144,238,144),
    'lightpink': (255,182,193),
    'lightsalmon': (255,160,122),
    'lightseagreen': (32,178,170),
    'lightskyblue': (135,206,250),
    'lightslategray': (119,136,153),
    'lightsteelblue': (176,196,222),
    'lightyellow': (255,255,224),
    'lime': (0,255,0),
    'limegreen': (50,205,50),
    'linen': (250,240,230),
    'magenta': (255,0,255),
    'maroon': (128,0,0),
    'mediumaquamarine': (102,205,170),
    'mediumblue': (0,0,205),
    'mediumorchid': (186,85,211),
    'mediumpurple': (147,112,219),
    'mediumseagreen': (60,179,113),
    'mediumslateblue': (123,104,238),
    'mediumspringgreen': (0,250,154),
    'mediumturquoise': (72,209,204),
    'mediumvioletred': (199,21,133),
    'midnightblue': (25,25,112),
    'mintcream': (245,255,250),
    'mistyrose': (255,228,225),
    'moccasin': (255,228,181),
    'navajowhite': (255,222,173),
    'navy': (0,0,128),
    'oldlace': (253,245,230),
    'olive': (128,128,0),
    'olivedrab': (107,142,35),
    'orange': (255,165,0),
    'orangered': (255,69,0),
    'orchid': (218,112,214),
    'palegoldenrod': (238,232,170),
    'palegreen': (152,251,152),
    'paleturquoise': (175,238,238),
    'palevioletred': (219,112,147),
    'papayawhip': (255,239,213),
    'peachpuff': (255,218,185),
    'peru': (205,133,63),
    'pink': (255,192,203),
    'plum': (221,160,221),
    'powderblue': (176,224,230),
    'purple': (128,0,128),
    'rebeccapurple': (102,51,153),
    'red': (255,0,0),
    'rosybrown': (188,143,143),
    'royalblue': (65,105,225),
    'saddlebrown': (139,69,19),
    'salmon': (250,128,114),
    'sandybrown': (244,164,96),
    'seagreen': (46,139,87),
    'seashell': (255,245,238),
    'sienna': (160,82,45),
    'silver': (192,192,192),
    'skyblue': (135,206,235),
    'slateblue': (106,90,205),
    'slategray': (112,128,144),
    'snow': (255,250,250),
    'springgreen': (0,255,127),
    'steelblue': (70,130,180),
    'tan': (210,180,140),
    'teal': (0,128,128),
    'thistle': (216,191,216),
    'tomato': (255,99,71),
    'turquoise': (64,224,208),
    'violet': (238,130,238),
    'wheat': (245,222,179),
    'white': (255,255,255),
    'whitesmoke': (245,245,245),
    'yellow': (255,255,0),
    'yellowgreen': (154,205,50)
    

}

# Function to extract colors from an image
def extract_dominant_colors(image, n_colors=3):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Reshape image to pixels
    pixels = image_rgb.reshape(-1, 3)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    # Get dominant colors
    colors=kmeans.cluster_centers_.astype(int)
    return colors

# Function to convert RGB to Hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


# Function to get closest color name
def get_color_name(rgb):
    try:
        return webcolors.rgb_to_name(tuple(rgb))
    except ValueError:
        min_distance = float("inf")
        closest_name = None
        for name, ref_rgb in COMMON_CSS3_COLORS.items():

            distance = np.sqrt(
                (ref_rgb[0] - rgb[0]) ** 2 +
                (ref_rgb[1] - rgb[1]) ** 2 +
                (ref_rgb[2] - rgb[2]) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name if closest_name else "Unknown"
        
# Function to generate color comnbinations
def get_color_combinations(rgb):
    # Convert RGB to HSL
    color = Color(rgb, 'RGB', colorsys.rgb_to_hsv(*(c/255.0 for c in rgb)))
    # Generate complimentary analogous colors
    complementary = complementaryColor(color,)
    analogous = analogousColor(color,)
    #convert back to RGB
    comp_rgb = [int(c) for c in complementary]
    ana_rgb = [[int(c) for c in col] for col in analogous]
    return comp_rgb, ana_rgb

# Sample dyeing recipe databse (replace with CSV or JSON in production code)
dyeing_recipes = pd.DataFrame(
    {'color_name': ['red', 'blue', 'green', 'yellow', 'purple'],
    'hex' : ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#800080'],
    'dye_type' : ['Reactive', 'Pigment', 'Discharge', 'Direct', 'Neon'],
    'concentration' : ['2%', '3%', '1.5%', '2.5%', '2%'],
    'temperature' : ['60C', '70C', '50C', '80C', '65C'],
    'time' : ['30min', '45min', '25min', '40min', '35min'],
    'fabric_type' : ['Single Jersey', 'Fleece', 'French Terry', 'Cordry', 'Interlock']
    }
    )

# Function to find closest dyeing recipe
def find_dyeing_recipe(hex_color, fabric_type):
    recipes = dyeing_recipes[dyeing_recipes['fabric_type'] == fabric_type]
    if recipes.empty:
        return None
    # Simple matching by HEX (In production , use color distace metric for better matching)
    for _, row in recipes.iterrows():
        if row['hex'].lower() == hex_color.lower():
            return row
    return recipes.iloc[0]  # Return first recipe as fallback

# Streamlit UI
st.title("AI Garments Dyeing Recipe Manager")
st.write("Upload an image of fabric shade to get color combinations and dyeing recipers")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read image
    image= Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    #Display uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    #Extract dominant colors
    n_colors = st.slider("Select number of dominant colors to extract", 1, 5, 3)
    dominant_colors = extract_dominant_colors(image_cv, n_colors)

    # Display dominant colors and their details
    st.subheader("Dominant Colors and Details")
    for i, color in enumerate(dominant_colors):
        hex_color = rgb_to_hex(color)
        color_name = get_color_name(color)
        st.write(f"Color {i+1}: {color_name}  (RGB: {color}, HEX: {hex_color})")

        # Display color swatch
        swatch = np.full((50, 100, 3), color, dtype=np.uint8)
        st.image(swatch, caption=color_name, width=100)

    # Generate color combinations for the first dominant color
    st.subheader("Sugested Color Combinations")
    comp_rgb, ana_rgb = get_color_combinations(dominant_colors[0])

    # Complementary Color

    comp_hex = rgb_to_hex(comp_rgb)
    comp_name = get_color_name(comp_rgb)
    st.write(f"Complementary Color: {comp_name} (RGB: {comp_rgb}, HEX: {comp_hex})")
    swatch_comp = np.full((50, 100, 3), comp_rgb, dtype=np.uint8)
    st.image(swatch_comp, caption="Complementary", width=100)

    # Analogous Colors
    st.write("Analogous Colors:")
    for ana_color in ana_rgb:
        ana_hex = rgb_to_hex(ana_color)
        ana_name = get_color_name(ana_color)
        st.write(f"{ana_name} (RGB: {ana_color}, HEX: {ana_hex})")
        swatch_ana = np.full((50, 100, 3), ana_color, dtype=np.uint8)
        st.image(swatch_ana, caption=ana_name, width=100)

    # Dyeing Recipe Suggestion
    st.subheader("Dyeing Recipe Suggestion")
    fabric_type = st.selectbox("Select Fabric Type", dyeing_recipes['fabric_type'].unique())
    hex_color = rgb_to_hex(dominant_colors[0])
    recipe = find_dyeing_recipe(hex_color, fabric_type)

    if recipe is not None:
        st.write (f"**Color**: {recipe['color_name']}(HEX: {recipe['hex']})")
        st.write (f"**Dye Type**: {recipe['dye_type']}")
        st.write (f"**Concentration**: {recipe['concentration']}")
        st.write (f"**Temperature**: {recipe['temperature']}")
        st.write (f"**Time**: {recipe['time']}")
        st.write (f"**Fabric Type**: {recipe['fabric_type']}")
    else:
        st.write("No matching dyeing recipe found for the selected fabric type.")

