import streamlit as st
from multipage import MultiPage
from pages import Tunnel_Convergence

app = MultiPage()
app.add_page("Denchai_ChiangRai_ChiangKhong", Tunnel_Convergence.app)

app.run()
