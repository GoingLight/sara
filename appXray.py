import streamlit as st
from models import (PRED_MOD,clf_model)
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date

# import plotly.express as px
# Title and header
st.title("SI-TASHXIS")


st.header("Sun'iy intellekt asosida tashxis qo'yish")
# image
st.image('./images/main.png')


# sidebar-info
st.sidebar.title('BOSHQARISH PANELI')

img = st.sidebar.file_uploader('X-ray rasmini yuklang')


st.sidebar.title('DASTUR HAQIDA')
st.sidebar.text("""Ushbu tashxis qo'yuvchi dasturimiz
Sun'iy Intellekt texnologiyasi (CNN)
asosida qurildi va 10,000 dan ortiq
X-ray ma'lumotlardan foydalanilindi.""")
st.sidebar.image('./images/Logo_dark.png')
st.sidebar.code('Muallif: Mansurbek Abdullayev\nManba: @python_ai_uz\nEmail: mansurbek.comchemai@gmail.com ')

tips_normal = "Sizning holatingiz normal, sog'ligingizga e'toborli bo'ling 👋🏻"
tips_covid = " Sizda COVID tashxisi aniqlandi, o'zingizga yaqinroq joydagi kasalxonaga murojat qiling! 🏥 "
tips_viral = " Sizda virus aniqlandi, o'zingizga yaqinroq joydagi kasalxonaga murojat qiling! 🏥  "
def plotting(x):
    labels= ['normal', 'virusli', 'covid-19']
    fig = go.Figure(go.Bar(
            x=x,
            y=labels,
            orientation='h',
            marker=dict(
        color='rgba(236, 15, 122, 0.8)',
        line=dict(color='rgba(236, 15, 122, 0.8)', width=3)
    )))
    fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
    fig.update_layout(title='TASHXIS', xaxis=dict(title='Natija (%)',
        titlefont_size=16,
        tickfont_size=14,),
        yaxis=dict(title='Holat',
        titlefont_size=16,
        tickfont_size=14,)
        )
    fig.update_xaxes(tickvals=list(range(0,101, 20)))
    return st.plotly_chart(fig)


while img is not None:
    st.title("NATIJA")
    st.subheader(date.today())
    if img.name.endswith(('png', 'jpg', 'jpeg')):
        if clf_model(img) == 'x_ray':
            accu = PRED_MOD(img)
            accu = np.where(accu<0, 0, accu)
            accu = ((accu/np.sum(accu))*100).reshape(3,).tolist()
            # print(accu)
            plotting(accu)
            if np.argmax(accu) == 0:
                st.success(tips_normal)
            elif np.argmax(accu) == 1:
                st.warning(tips_viral)
            else:
                st.error(tips_covid)
            # st.error(tips_normal)
            # st.bar_chart(plotting())
            st.image('result.png')
            break
        else:
            st.error('Iltimos, X_ray rasmini yuklang')
            break
    else:
        st.error('Iltimos, png yoki jpg formatdagi rasmni yuklang')
        break
