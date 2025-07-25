import numpy as np
import yfinance as yf
import streamlit as st
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
from monte_carlo_sim import tickers, MonteCarloSimulation, VolatilityCluster
import warnings
warnings.filterwarnings("ignore")

# Streamlit UI
st.set_page_config(layout="wide", page_title="Stock Risk Analyzer")
st.title("📊 Monte-Carlo Drawdown Analyzer")

# Sidebar controls
st.sidebar.header("⚙️ Parameters")
selected_company = st.sidebar.selectbox("Select Company", options=list(tickers.keys()))
ticker = tickers[selected_company]
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 365, 90)
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000, step=1000)
drawdown_threshold = st.sidebar.slider("Drawdown Threshold (%)", 5, 50, 15) / 100
drawdown_horizon = st.sidebar.slider("Risk Horizon (days)", 5, 90, 30)

if st.sidebar.button("🔍 Analyze"):
    with st.spinner("Processing..."):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)[['Close']].dropna()
            if data.empty:
                st.error("No data available for the selected period")
                st.stop()

            clusterer = VolatilityCluster(data)
            clustered_data = clusterer.cluster()
            latest = clustered_data.iloc[-1]

            current_regime = 'stable' if 'stable' in str(latest['volatility_regime']) else 'volatile'
            current_price = float(latest['Close'])
            current_volatility = float(latest['volatility'])
            returns = clustered_data['log_return'].dropna()

            st.subheader(f"📌 {selected_company} ({ticker}) Market Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Volatility Regime", current_regime)
            col3.metric("Annualized Volatility", f"{current_volatility:.2%}")

            st.subheader("📈 Stable Regime Analysis")
            stable_data = clustered_data[clustered_data['volatility_regime'] == 'stable']

            # Summary stats
            mean_price_stable = stable_data['Close'][ticker].mean()
            median_price_stable = stable_data['Close'][ticker].median()
            std_price_stable = stable_data['Close'][ticker].std()

            mean_vol_stable = stable_data['volatility'].mean()
            median_vol_stable = stable_data['volatility'].median()
            std_vol_stable = stable_data['volatility'].std()

            # Display stats
            st.markdown(f"""
            **Price Statistics (Stable Regime)**  
            - Mean: ${mean_price_stable:.2f}  
            - Median: ${median_price_stable:.2f}  
            - Std Dev: ${std_price_stable:.2f}

            **Volatility Statistics (Stable Regime)**  
            - Mean: {mean_vol_stable:.2%}  
            - Median: {median_vol_stable:.2%}  
            - Std Dev: {std_vol_stable:.2%}
            """)

            # Plot
            fig_stable = make_subplots(specs=[[{"secondary_y": True}]])
            fig_stable.add_trace(go.Scatter(x=stable_data.index, y=stable_data['Close'][ticker], name='Price', line=dict(color='#1f77b4')), secondary_y=False)
            fig_stable.add_trace(go.Scatter(x=stable_data.index, y=stable_data['volatility'], name='Volatility', line=dict(color='#ff7f0e')), secondary_y=True)
            fig_stable.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig_stable.update_yaxes(title_text="Volatility", secondary_y=True, tickformat=".0%")
            fig_stable.update_layout(title="Stable Period - Price and Volatility", height=500)
            st.plotly_chart(fig_stable, use_container_width=True)

            st.subheader("📈 Volatile Regime Analysis")
            volatile_data = clustered_data[clustered_data['volatility_regime'] == 'volatile']

            # Summary stats
            mean_price_volatile = volatile_data['Close'][ticker].mean()
            median_price_volatile = volatile_data['Close'][ticker].median()
            std_price_volatile = volatile_data['Close'][ticker].std()

            mean_vol_volatile = volatile_data['volatility'].mean()
            median_vol_volatile = volatile_data['volatility'].median()
            std_vol_volatile = volatile_data['volatility'].std()

            # Display stats
            st.markdown(f"""
            **Price Statistics (Volatile Regime)**  
            - Mean: ${mean_price_volatile:.2f}  
            - Median: ${median_price_volatile:.2f}  
            - Std Dev: ${std_price_volatile:.2f}

            **Volatility Statistics (Volatile Regime)**  
            - Mean: {mean_vol_volatile:.2%}  
            - Median: {median_vol_volatile:.2%}  
            - Std Dev: {std_vol_volatile:.2%}
            """)

            # Plot
            fig_volatile = make_subplots(specs=[[{"secondary_y": True}]])
            fig_volatile.add_trace(go.Scatter(x=volatile_data.index, y=volatile_data['Close'][ticker], name='Price', line=dict(color='#1f77b4')), secondary_y=False)
            fig_volatile.add_trace(go.Scatter(x=volatile_data.index, y=volatile_data['volatility'], name='Volatility', line=dict(color='#ff7f0e')), secondary_y=True)
            fig_volatile.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig_volatile.update_yaxes(title_text="Volatility", secondary_y=True, tickformat=".0%")
            fig_volatile.update_layout(title="Volatile Period - Price and Volatility", height=500)
            st.plotly_chart(fig_volatile, use_container_width=True)

            st.subheader("🎲 Monte Carlo Simulation Results")
            mc = MonteCarloSimulation(returns, current_price, num_simulations, forecast_days)
            sim_paths = mc.simulate()

            def label_drawdown_risk(sim_path, threshold, horizon):
                peak = sim_path[0]
                for i in range(1, min(horizon, len(sim_path))):
                    drawdown = (sim_path[i] - peak) / peak
                    if drawdown < -threshold:
                        return 1
                    peak = max(peak, sim_path[i])
                return 0

            labels = [label_drawdown_risk(sim_paths[:,i], drawdown_threshold, drawdown_horizon) 
                     for i in range(sim_paths.shape[1])]
            risk_prob = np.mean(labels)

            st.subheader("📉 Risk Assessment")
            risk_col1, risk_col2 = st.columns(2)
            risk_col1.metric(
                f"Probability of {int(drawdown_threshold*100)}% Drawdown",
                f"{risk_prob:.1%}",
                f"Within {drawdown_horizon} days"
            )

            fig2 = go.Figure()
            percentiles = np.percentile(sim_paths, [5, 25, 50, 75, 95], axis=1)
            fig2.add_trace(go.Scatter(
                x=list(range(forecast_days)),
                y=percentiles[2],
                line=dict(color='#1f77b4', width=2),
                name='Median Path'
            ))

            for i, (lower, upper) in enumerate([(0,1), (1,3), (3,4)]):
                fig2.add_trace(go.Scatter(
                    x=list(range(forecast_days)) + list(range(forecast_days))[::-1],
                    y=np.concatenate([percentiles[lower], percentiles[upper][::-1]]),
                    fill='toself',
                    fillcolor=f'rgba(31, 119, 180, {0.2+i*0.2})',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{5 if lower==0 else 25 if lower==1 else 75}th-{95 if upper==4 else 75 if upper==3 else 25}th percentile'
                ))

            fig2.add_hline(
                y=current_price,
                line=dict(color='#d62728', dash='dash'),
                annotation_text="Current Price",
                annotation_position="bottom right"
            )

            fig2.update_layout(
                title=f"Price Simulation Distribution ({num_simulations:,} paths)",
                xaxis_title="Days Ahead",
                yaxis_title="Price ($)",
                hovermode="x",
                height=500
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")