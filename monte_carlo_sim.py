import json
import numpy as np
from sklearn.cluster import KMeans


# Load tickers
with open('tickers.json', 'r') as f:
    tickers = json.load(f)

# Monte Carlo Simulation
class MonteCarloSimulation:
    def __init__(self, returns_series, start_price, num_simulations, days):
        self.returns_series = returns_series
        self.start_price = start_price
        self.num_simulations = num_simulations
        self.days = days

    def simulate(self):
        mu = float(self.returns_series.mean())
        sigma = float(self.returns_series.std())
        simulated_prices = np.zeros((self.days, self.num_simulations))
        simulated_prices[0] = self.start_price
        for t in range(1, self.days):
            rand = np.random.normal(0, 1, self.num_simulations)
            simulated_prices[t] = simulated_prices[t - 1] * np.exp((mu - 0.5 * sigma**2) + sigma * rand)
        return simulated_prices

# Volatility Clustering
class VolatilityCluster:
    def __init__(self, data, window=21):
        self.data = data
        self.window = window

    def cluster(self):
        df = self.data.copy()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=self.window).std() * np.sqrt(252)
        df = df.dropna()

        if len(df) < self.window * 2:
            raise ValueError("Insufficient data for clustering")

        X = df[['volatility']]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        df['volatility_cluster'] = kmeans.labels_
        cluster_means = df.groupby('volatility_cluster')['volatility'].mean()
        stable = cluster_means.idxmin()
        df['volatility_regime'] = df['volatility_cluster'].map({stable: 'stable', 1 - stable: 'volatile'}).astype(str)
        return df