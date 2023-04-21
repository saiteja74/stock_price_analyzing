import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
        page_title="Stock Price Analyzing",
)

st.write("""
CryptoCurrency Dashboard Application
Visualizing three types of data (DogeCoin, Ethereum and Zomato) from JAN 1 2022 to APR 20 2023
""")

st.write("""
Important Note:-
This app takes three types of stocks they are (DogeCoin, Ethereum and Zomato)
""")

st.write("""
To choose different stock please type DOGE for DogeCoin in left side User Input in Crypto Symbol
""")

st.write("""
And also type eth for Ethereum and zomato for ZOMATO
""")




st.sidebar.header("User Input")

def get_input():
    start_date = st.sidebar.text_input("Start Date", "JAN 1 2022")
    end_date = st.sidebar.text_input("End Date", "APR 20 2023")
    crypto_symbol = st.sidebar.text_input("Crypto Symbol", "DOGE")
    return start_date, end_date, crypto_symbol


def get_crypto_name(symbol):
    symbol = symbol.upper()
    if symbol == "DOGE":
        return "DogeCoin"
    elif symbol == "ETH":
        return "Ethereum"
    elif symbol == "ZOMATO":
        return "Zomato"
    else:
        return "None"

def get_data(symbol, start, end):
    symbol = symbol.upper()
    if symbol == "DOGE":
        df = pd.read_csv('DOGE-INR.csv')
    elif symbol == "ETH":
        df = pd.read_csv('ETH-INR.csv')
    elif symbol == "ZOMATO":
        df = pd.read_csv('ZOMATO.NS.csv')
    else:
        df = pd.DataFrame(columns=['Date', 'Close', 'Open', 'Volume', 'Adj Close'])

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    df = df.set_index(pd.DatetimeIndex(df['Date'].values))

    return df.loc[start:end]

start, end, symbol = get_input()
df = get_data(symbol, start, end)
crypto_name = get_crypto_name(symbol)

fig = go.Figure(
    data = [go.Candlestick(
        x = df.index,
        open = df['Open'],
        high = df['High'],
        low = df['Low'],
        close = df['Close'],
        increasing_line_color = 'green',
        decreasing_line_color = 'red'
    )]
)

figg = go.Figure(
    data = [go.Scatter3d(
        
        x = df['Open'],
        y = df['High'],
        z = df['Low'],
        mode="markers",
        marker = dict(
                    size = 12,
                    color = df['Close'],
                    colorscale ='Viridis',
                    opacity = 0.8
                )
        
    )]
)

st.header(crypto_name+" Data")
st.write(df)

st.header(crypto_name+" Data Statistics")
st.write(df.describe())

st.header(crypto_name+" Volume")
st.bar_chart(df['Volume'])

# st.header(crypto_name+" Volume")
# st.altair_chart(df['Volume'])

st.header(crypto_name+" Open")
st.line_chart(df['Open'])

st.header(crypto_name+" Close")
st.line_chart(df['Close'])

st.header(crypto_name+" Candle Stick")
st.plotly_chart(fig)

st.header(crypto_name+" ScatterPlot3D")
st.plotly_chart(figg)

st.title('Linear Regression for'+  crypto_name  + 'Stock Prices')
#st.write('This app performs linear regression on the open and close prices of', crypto_name ,'stock.')

X = df[['Open']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the R-squared value
r_squared = r2_score(y_test, y_pred)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Calculate the root mean squared error
rmse = np.sqrt(mse)

# Create a DataFrame for the regression line
regression_data = pd.DataFrame({'Open': np.linspace(X.min()[0], X.max()[0], 100)})
regression_data['Close'] = model.predict(regression_data[['Open']])

# Print the metrics
st.write('R-squared:', r_squared)
st.write('Mean squared error:', mse)
st.write('Mean absolute error:', mae)
st.write('Root mean squared error:', rmse)

# Create the chart
chart = alt.Chart(df).mark_point().encode(
    x='Open',
    y='Close'
)

regression_line = alt.Chart(regression_data).mark_line(color='red').encode(
    x='Open',
    y='Close'
)

st.altair_chart(chart + regression_line, use_container_width=True)