import numpy as np
import pandas as pd
from scipy.fft import fft
import pywt

def extract_fourier_features(time_series):
    # Perform Fourier transform
    fourier_transform = fft(time_series)
    magnitudes = np.abs(fourier_transform)
    phases = np.angle(fourier_transform)

    # Extract features
    dominant_frequency = np.argmax(magnitudes)
    power_spectrum = np.square(magnitudes)
    feature_names = ['Dominant Frequency', 'Power Spectrum']
    features = [dominant_frequency, power_spectrum]

    return feature_names, features


def extract_wavelet_features(time_series):
    # Perform wavelet transform
    wavelet_coeffs, wavelet_scales = pywt.cwt(time_series, np.arange(1, 31), 'morl')

    # Extract features
    feature_names = ['Wavelet Coeff_' + str(scale) for scale in wavelet_scales]
    features = wavelet_coeffs.T.tolist()

    return feature_names, features

from neuralprophet import NeuralProphet

# Calculate the Neural Prophet model fit error as a measure of trend strength
def calculate_trend_strength(data, window):
    trend_strength = []
    for i in range(window, len(data)):
        model = NeuralProphet()
        model_fit = model.fit(data[['timestamp', 'price']].iloc[i-window:i], freq='1D')
        future = model.make_future_dataframe(data[['timestamp', 'price']].iloc[i-window:i], periods=1)
        forecast = model.predict(future)
        trend_strength.append(abs(forecast['yhat1'].values[-1] - data['price'].iloc[i]))
    return pd.Series(trend_strength, index=data.index[window:])



def save_features_to_csv(feature_names, features, filename):
    feature_df = pd.DataFrame([features], columns=feature_names)
    feature_df.to_csv(filename, index=False)


# Example usage
time_series = np.random.randn(1000)  # Replace with your own time series data

# Extract Fourier features
fourier_feature_names, fourier_features = extract_fourier_features(time_series)

# Extract Wavelet features
wavelet_feature_names, wavelet_features = extract_wavelet_features(time_series)

# Save features to CSV files
save_features_to_csv(fourier_feature_names, fourier_features, 'fourier_features.csv')
save_features_to_csv(wavelet_feature_names, wavelet_features, 'wavelet_features.csv')
