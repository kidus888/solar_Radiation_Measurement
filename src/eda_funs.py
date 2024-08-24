import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def summary_statistics(df):
    """Calculate and return summary statistics of the dataframe."""
    return df.describe()

def data_quality_check(df):
    """Check for missing values and outliers in the dataframe."""
    missing_values = df.isnull().sum()
    outliers = df.apply(lambda x: np.sum(np.abs(zscore(x)) > 3), axis=0)
    return missing_values, outliers

def plot_time_series(df, columns, title):
    """Plot time series data for the specified columns."""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df[columns].plot(figsize=(15, 7))
    plt.title(title)
    plt.show()

def plot_correlation_heatmap(df, columns):
    """Plot a heatmap showing correlations between specified columns."""
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_wind_polar(df):
    """Plot a polar plot of wind speed and direction."""
    wind_speed = df['WS']
    wind_dir = df['WD']
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(np.deg2rad(wind_dir), wind_speed)
    plt.title('Wind Speed and Direction')
    plt.show()

def plot_histograms(df, columns):
    """Plot histograms for the specified columns."""
    df[columns].hist(bins=20, figsize=(15, 10))
    plt.suptitle('Histograms')
    plt.show()

def calculate_z_scores(df):
    """Calculate Z-scores for the dataframe."""
    return df.apply(zscore)

def plot_bubble_chart(df, x, y, size, color):
    """Plot a bubble chart with the specified parameters."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y], s=df[size]*100, c=df[color], cmap='viridis', alpha=0.6)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y} Bubble Chart')
    plt.colorbar(label=color)
    plt.show()

def clean_data(df):
    """Clean the dataset by handling anomalies and missing values."""
    df = df.dropna(subset=['GHI', 'DNI', 'DHI'])
    df = df[df['GHI'] >= 0]
    df = df[df['DNI'] >= 0]
    df = df[df['DHI'] >= 0]
    return df
