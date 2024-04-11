import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# df = pd.read_csv(
#     "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
# )
df = pd.read_csv("DOGE.csv")

# TODO
# https://defillama.com/ or https://www.coindesk.com/price/doge to get from API and add to the csv the missing 

# Convert the "Date" column to datetime if it's not already in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Set the "Date" column as the index
# df.set_index("Date", inplace=True)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension

# context = torch.tensor(df["#Passengers"])

context = torch.tensor(df["Close"])

prediction_length = 30  # 1 month
forecast = pipeline.predict(
    context, prediction_length
)  # shape [num_series, num_samples, prediction_length]

# visualize the forecast
# forecast_index = range(len(df), len(df) + prediction_length)

# Generate forecast index for plotting
last_date = df["Date"].iloc[-1]
forecast_index = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=prediction_length, freq='D'
)

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# plt.figure(figsize=(8, 4))
plt.figure(figsize=(10, 6))
# plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(df["Date"], df["Close"], color="royalblue", label="Historical Data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(
    forecast_index,
    low,
    high,
    color="tomato",
    alpha=0.3,
    label="80% prediction interval",
)
plt.title("Close Price Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.savefig("forecast_plot.png")
plt.show()
# Zoom in on the forecasted portion
plt.xlim(last_date, last_date + pd.Timedelta(days=prediction_length))
plt.savefig("forecast_plot_zoomed.png")

