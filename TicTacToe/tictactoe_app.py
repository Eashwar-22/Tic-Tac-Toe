import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

st.title("Tic Tac Toe")
st.markdown('''---''')

# nav = st.sidebar.radio("MENU",['Play','Simulate','Contribute'])


inter=st.container()
option=st.container()
left,right=option.columns(2)

data=[['1', '4', '7'],['2', '5', '8'], ['3','6','9']]
with inter:
    st.header('Figure')
    x=left.radio('X: Pick one', [1,2,3,4,5,6,7,8,9])
    o=right.radio('O: Pick one', [1, 2, 3, 4, 5, 6, 7, 8, 9])

    d=np.array(data).reshape(-1)
    d[int(x)-1]='X'
    d[int(o) - 1] = 'O'
    data=np.array(d).reshape(3,3)

    fig = go.Figure(data=[go.Table(header=dict(values=['', '','']),
                                   cells=dict(values=data
                                              ,align='center'))
                          ])
    inter.write(fig)
    fig.update_layout()





