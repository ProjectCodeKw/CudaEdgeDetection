import streamlit as st
from streamlit_extras import add_vertical_space as avs
from time import sleep
from PIL import Image
import subprocess
import os
import hashlib
import time

local_image_path = "og_image.ppm"
local_new_image_path = "edge_result.ppm"
cuda_path = "edgeDetectionKernel.exe"

def run_cuda_program():
    try:
        result = subprocess.run([cuda_path], capture_output=True, text=True, check=True, timeout=5)
    
        if result.stdout:
            print("CUDA Program Output:")
            print(result.stdout)
            html_content = f'<div style="text-align: center; color: yellow; font-size: 15px; font-family: Verdana;">{result.stdout}</div>'
            st.markdown(html_content, unsafe_allow_html=True)
        
        if result.stderr:
            print("CUDA Program Error:")
            print(result.stderr)
        
        print("Exited the process.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running CUDA program: {e}")
        print("Error output:", e.stderr)
    except FileNotFoundError:
        print(f"CUDA program '{cuda_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def read_edge():
    # Wait for the file to become accessible
    try:
        return Image.open(local_new_image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

def delete_file(file):
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted file: {file}")
    else:
        print(f"File {file} does not exist, nothing to delete.")

def convert_to_ppm():
    with Image.open(local_image_path) as img:
        img = img.convert("L")  # Convert to grayscale P5
        img.save(local_image_path, format='PPM')  # Overwrite the image

def get_image_hash(img):
    # Generate a hash for the image file to track changes
    return hashlib.md5(img.getvalue()).hexdigest()

if 'old_image_hash' not in st.session_state:
    st.session_state.old_image_hash = ''

if __name__ == "__main__":

    st.markdown("""
        <div style="text-align: center; color: white; font-size: 30px; font-family: Verdana;">
            CUDA Image Processing App
        </div>
        """, unsafe_allow_html=True)
    avs.add_vertical_space(1)
    st.markdown("""
        <div style="text-align: center; color: white; font-size: 20px; font-family: Verdana;">
            Converts your RGB image to Grayscale then Edge Detection
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    avs.add_vertical_space(1)
    
    if st.session_state.old_image_hash == '':
        img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        # Check if the image has changed based on the hash
        if img:
            img_hash = get_image_hash(img)
            # Save the image to the specified location
            with open(local_image_path, "wb") as f:
                f.write(img.getbuffer())

            edge_image = None
            

            st.image(local_image_path, caption="Original Image", use_container_width=True)
            st.markdown("""
                <div style="text-align: center; color: white; font-size: 20px; font-family: Verdana;">
                    Your Image after edge detection:
                </div>
                """, unsafe_allow_html=True)
            avs.add_vertical_space(1)
            
            # Convert the image to PPM P5
            convert_to_ppm()
            # Call the CUDA program
            run_cuda_program()
            # Read the edge-detected image
            edge_image = read_edge()
            # Display the edge-detected image
            if edge_image:
                st.image(edge_image, caption="Edge Detected Image", use_container_width=True)
            else:
                st.error("Edge detection failed or the image file could not be found.")
            
            st.session_state.old_image_hash = img_hash  # Update the hash to track this image
    else:
        st.warning("Please refresh the page to try another image.")