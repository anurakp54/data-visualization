import streamlit as st
from multipage import MultiPage
from pages import Tunnel_Convergence
import os


app = MultiPage()
app.add_page("Denchai_ChiangRai_ChiangKhong", Tunnel_Convergence.app)

# Upload file

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    if "aligned_coordinate_df" in uploaded_file.name:
        # Save file to 'data' folder
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Your file was saved to {file_path}")
    else:
        st.error("You are not authorized")


app.run()
