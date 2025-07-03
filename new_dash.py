import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

st.title("\U0001F4CA Economic Indicator Dashboard")

# === Step 1: Select Country ===
countries = ["US", "UK", "EZ", "CA", "Aussie"]
country = st.sidebar.selectbox("\U0001F30D Select Country", countries)

# === Step 2: List Files in Country Folder ===
folder_path = os.path.join("Data", country)
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

if not files:
    st.warning(f"No CSV files found in {folder_path}")
    st.stop()

file1 = st.sidebar.selectbox("\U0001F4C1 Select Target Indicator (Tier-1)", files, key="file1")
selected_soft_indicators = st.sidebar.multiselect("\U0001F4CB Select Soft Indicators (Leading)", [f for f in files if f != file1])

# === Step 3: Load CSVs ===
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Reference Period", "Release Date"])
    df['Surprise'] = df['Actual'] - df['Median_Forecast']
    df['Month'] = df['Reference Period'].dt.month
    return df

df_target = load_data(os.path.join(folder_path, file1))
df_softs = [load_data(os.path.join(folder_path, sf)) for sf in selected_soft_indicators]

# === Global Filter Controls ===
st.sidebar.markdown("### \U0001F5C2 Filter Timeframe")

df_target['Reference Period'] = pd.to_datetime(df_target['Reference Period'])

min_date, max_date = df_target['Reference Period'].min().date(), df_target['Reference Period'].max().date()

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

max_n = len(df_target)
n_obs = st.sidebar.slider("Or show last N observations", min_value=5, max_value=max_n, value=30, step=1)

# === Sync Filters ===
apply_n_obs = st.sidebar.checkbox("Apply N observations filter", value=False)

if apply_n_obs:
    df_target_filtered = df_target.sort_values('Reference Period').tail(n_obs)
else:
    df_target_filtered = df_target[(df_target['Reference Period'] >= pd.to_datetime(start_date)) &
                                   (df_target['Reference Period'] <= pd.to_datetime(end_date))]


# === Tabs ===
tabs = st.tabs([
    "📈 Time Series", "📅 Seasonality", "📊 Surprise Distribution",
    "🔮 Multi-Indicator Prediction", "🌍 Cross-Country Comparison",
    "🌐 Intermarket Lag Analysis", "💹 Market Reaction",
    "⏰ Forecast Next Release", "💾 Export",
    "📤 Upload & Compare", "📊 Rate Cycle Regimes"
])


# === Time Series ===
with tabs[0]:
    st.subheader("Actual vs Forecast - Tier 1 Indicator")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_target_filtered['Reference Period'], y=df_target_filtered['Actual'],
                             mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=df_target_filtered['Reference Period'], y=df_target_filtered['Median_Forecast'],
                             mode='lines+markers', name='Median Forecast'))
    fig.update_layout(title=file1.replace('.csv', ''), xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)


# === Seasonality ===
with tabs[1]:
    st.subheader("Monthly Surprise Seasonality")
    monthly = df_target_filtered.groupby('Month')['Surprise'].mean()
    fig = px.bar(x=monthly.index, y=monthly.values,
                 labels={'x': 'Month', 'y': 'Avg Surprise'},
                 title='Monthly Surprise Seasonality')
    fig.update_xaxes(tickmode='array', tickvals=list(range(12)),
                     ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    st.plotly_chart(fig, use_container_width=True)


# === Surprise Distribution ===
with tabs[2]:
    st.subheader("Surprise Distribution")
    fig = px.histogram(df_target_filtered, x='Surprise', nbins=30, marginal="rug", opacity=0.7,
                       title="Actual - Forecast Distribution")
    st.plotly_chart(fig, use_container_width=True)


# === Multi-Indicator Predictive Model ===
with tabs[3]:
    st.subheader("Predict Tier-1 Indicator from Soft Indicators")
    df_merge = df_target_filtered[['Reference Period', 'Actual']].rename(columns={'Actual': 'Target'})
    for i, df_soft in enumerate(df_softs):
        df_soft_filtered = df_soft[df_soft['Reference Period'].isin(df_merge['Reference Period'])]
        df_merge = pd.merge(df_merge, df_soft_filtered[['Reference Period', 'Actual']], on='Reference Period', how='left')
        df_merge.rename(columns={'Actual': selected_soft_indicators[i].replace('.csv', '')}, inplace=True)
    df_merge.dropna(inplace=True)

    if not df_merge.empty:
        X = df_merge.drop(columns=['Reference Period', 'Target'])
        y = df_merge['Target']
        if X.empty or y.empty:
            st.error("❌ Not enough data to train model. Please choose other indicators or check for missing data.")
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            prediction = model.predict(X)
            st.write(f"R² Score: **{model.score(X, y):.2f}**")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_merge['Reference Period'], y=y, name='Actual'))
            fig.add_trace(go.Scatter(x=df_merge['Reference Period'], y=prediction, name='Predicted'))
            fig.update_layout(title=f"Prediction of {file1.replace('.csv', '')}", xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig, use_container_width=True)


            st.write("\U0001F4CB **Feature Importance**")
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig = px.bar(x=imp.index, y=imp.values, labels={'x': 'Feature', 'y': 'Importance'},
                         title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough overlapping data to build model.")

# === Cross-Country ===
with tabs[4]:
    st.subheader("Compare Indicator(s) Across Countries")

    country2 = st.selectbox("Select Country for Comparison", countries)
    files2 = os.listdir(os.path.join("Data", country2))
    selected_files = st.multiselect("Select Indicator CSV(s)", [f for f in files2 if f.endswith(".csv")])

    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_target_filtered['Reference Period'], y=df_target_filtered['Actual'],
            mode='lines+markers', name=f"{file1.replace('.csv','')} ({country})"
        ))

        for sf in selected_files:
            df_other = load_data(os.path.join("Data", country2, sf))
            df_other_filtered = df_other[df_other['Reference Period'].isin(df_target_filtered['Reference Period'])]
            fig.add_trace(go.Scatter(
                x=df_other_filtered['Reference Period'], y=df_other_filtered['Actual'],
                mode='lines+markers', name=f"{sf.replace('.csv','')} ({country2})"
            ))

        fig.update_layout(title="Indicator Comparison", xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load file: {e}")


# === Intermarket Lag ===
with tabs[5]:
    st.subheader("Intermarket Lead-Lag Correlation")

    country_lag = st.selectbox("Select Country for Comparison", countries, key="lag_country_selector")
    lag_folder_path = os.path.join("Data", country_lag)
    lag_files = [f for f in os.listdir(lag_folder_path) if f.endswith(".csv")]
    lag_file = st.selectbox("Select Indicator CSV", lag_files, key="lag_file_selector")

    try:
        df_lead = df_target_filtered[['Reference Period', 'Actual']].rename(columns={"Actual": "Lead"})
        df_lag = load_data(os.path.join(lag_folder_path, lag_file))
        df_lag = df_lag[['Reference Period', 'Actual']].rename(columns={"Actual": "Lag"})

        max_lag = 36
        correlations = {}

        for lag in range(-max_lag, max_lag + 1):
            shifted = df_lag.copy()
            shifted['Reference Period'] = shifted['Reference Period'] - pd.DateOffset(months=lag)
            merged = pd.merge(df_lead, shifted, on='Reference Period')
            if len(merged) >= 3:
                corr = merged['Lead'].corr(merged['Lag'])
                correlations[lag] = corr

        best_lag = max(correlations, key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_lag]

        st.write(f"📈 Highest correlation at **{best_lag:+} month(s)** lag: **{best_corr:.2f}**")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(correlations.keys()), y=list(correlations.values()),
                                 mode='lines+markers', name='Correlation'))
        fig.add_vline(x=best_lag, line_dash="dash", line_color="red", annotation_text=f'Max Corr @ {best_lag:+}m')
        fig.update_layout(title='Correlation vs Lag', xaxis_title='Lag (months)', yaxis_title='Correlation')
        st.plotly_chart(fig, use_container_width=True)


        df_lag['Reference Period'] = df_lag['Reference Period'] - pd.DateOffset(months=best_lag)
        merged = pd.merge(df_lead, df_lag, on='Reference Period')
        fig = go.Figure()
        
        # Lead (left Y-axis)
        fig.add_trace(go.Scatter(
            x=merged['Reference Period'], y=merged['Lead'],
            name=f"{country} (Lead)", yaxis='y1', line=dict(color='blue')
        ))
        
        # Lag (right Y-axis)
        fig.add_trace(go.Scatter(
            x=merged['Reference Period'], y=merged['Lag'],
            name=f"{country_lag} (Lag {best_lag:+}m)", yaxis='y2', line=dict(color='orange')
        ))
        
        fig.update_layout(
            title="Best-Aligned Series Based on Lag",
            xaxis=dict(title='Reference Period'),
            yaxis=dict(
                title=dict(text=f"{country} (Lead)", font=dict(color='blue')),
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title=dict(text=f"{country_lag} (Lag {best_lag:+}m)", font=dict(color='orange')),
                tickfont=dict(color='orange'),
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.5, y=1.1, orientation='h'),
            margin=dict(t=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)




        max_lag = 365 * 3
        correlations = {}

        for lag in range(-max_lag, max_lag + 1):
            shifted = df_lag.copy()
            shifted['Reference Period'] = shifted['Reference Period'] + pd.DateOffset(days=lag)
            merged = pd.merge(df_lead, shifted, on='Reference Period')
            if len(merged) >= 3:
                corr = merged['Lead'].corr(merged['Lag'])
                correlations[lag] = corr

        best_lag = max(correlations, key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_lag]

        st.write(f"📈 Highest correlation at **{best_lag:+} day(s)** lag: **{best_corr:.2f}**")

    except Exception as e:
        st.warning(f"Could not compute lag correlation: {e}")



# === Market Reaction ===
with tabs[6]:
    st.subheader("Market Reaction to Surprise")
    st.markdown("Upload asset return CSV (columns: Release Date, Return) to analyze surprise-impact")
    asset_file = st.file_uploader("Upload Asset Returns CSV")
    if asset_file:
        try:
            df_asset = pd.read_csv(asset_file, parse_dates=["Release Date"])
            merged = pd.merge(df_target[['Release Date', 'Surprise']], df_asset, on='Release Date')
            st.write("Correlation Surprise vs Return: ", round(merged['Surprise'].corr(merged['Return']), 2))
            fig = px.scatter(merged, x='Surprise', y='Return', title="Market Return vs Economic Surprise")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not process file: {e}")


# === Forecast Next ===
with tabs[7]:
    st.subheader("Forecast Tier-1 Indicator for Next 12 Months (Manual Soft Indicator Input)")

    if 'model' in locals() and not df_merge.empty and not X.empty:
        forecast_horizon = 12
        future_dates = pd.date_range(
            start=df_merge['Reference Period'].max() + pd.DateOffset(months=1),
            periods=forecast_horizon, freq='MS'
        )

        st.markdown("### Input Future Values for Soft Indicators")
        future_inputs = {}

        for col in X.columns:
            default = X[col].iloc[-1]
            future_inputs[col] = []
            st.markdown(f"**{col}**")
            cols = st.columns(4)
            for i in range(forecast_horizon):
                with cols[i % 4]:
                    val = st.number_input(f"{future_dates[i].strftime('%b %Y')}", key=f"{col}_{i}", value=float(default))
                    future_inputs[col].append(val)

        # Build future_X from user input
        future_X = pd.DataFrame(future_inputs)
        future_preds = model.predict(future_X)

        # === ⚙️ Error Band Controls ===
        st.markdown("### Error Band Settings")

        recent_window = st.slider("Recent Residual Window (months)", 6, 48, 24, step=6)
        blend_weight = st.slider("Weight for Recent Std Dev (%)", 0, 100, 50, step=10) / 100

        # Residuals & Std Devs
        in_sample_preds = model.predict(X)
        residuals = y - in_sample_preds

        global_std = np.std(residuals)
        recent_std = np.std(residuals[-recent_window:])
        error_std = (1 - blend_weight) * global_std + blend_weight * recent_std

        st.write(f"Global Std Dev: **{global_std:.2f}**, Recent ({recent_window} mo): **{recent_std:.2f}**")
        st.write(f"Final Combined Std Dev: **{error_std:.2f}** (Blend: {int(blend_weight * 100)}% Recent / {100 - int(blend_weight * 100)}% Global)")

        # === Forecast DataFrame ===
        df_forecast = pd.DataFrame({
            'Reference Period': future_dates,
            'Prediction': future_preds,
            'Lower': future_preds - error_std,
            'Upper': future_preds + error_std
        })

        # === Combine Actual + Forecast ===
        df_plot = df_merge[['Reference Period', 'Target']].copy()
        df_plot['Prediction'] = in_sample_preds
        last_date = df_plot['Reference Period'].max()

        df_all = pd.concat([
            df_plot[['Reference Period', 'Target', 'Prediction']],
            df_forecast[['Reference Period', 'Prediction']].assign(Target=np.nan)
        ], ignore_index=True)

        # === Plot Forecast ===
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_all['Reference Period'], y=df_all['Target'], name='Actual', mode='lines'))
        fig.add_trace(go.Scatter(x=df_all['Reference Period'], y=df_all['Prediction'], name='Predicted', mode='lines'))
        fig.add_shape(
            type='line',
            x0=last_date, x1=last_date,
            y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='gray', dash='dot'),
        )
        
        fig.add_annotation(
            x=last_date,
            y=1.05,
            xref='x',
            yref='paper',
            showarrow=False,
            text='Forecast Start',
            font=dict(size=12, color='gray')
        )

        
        fig.add_trace(go.Scatter(
            x=df_forecast['Reference Period'], y=df_forecast['Upper'], mode='lines',
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_forecast['Reference Period'], y=df_forecast['Lower'], mode='lines',
            fill='tonexty', fillcolor='rgba(255,0,0,0.2)', line=dict(width=0),
            name='±1 Std Dev'
        ))
        
        fig.update_layout(title="Forecast with Manual Inputs",
                          xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)


        # === Forecast Table ===
        st.markdown("### 🔮 Forecast Table (Next 12 Months)")
        st.dataframe(
            df_forecast.set_index('Reference Period')
            .style.format({'Prediction': '{:.2f}', 'Lower': '{:.2f}', 'Upper': '{:.2f}'})
        )
    else:
        st.warning("Model not available or soft indicators missing.")



# === Export ===
with tabs[8]:
    st.subheader("Export Data")
    csv = df_target.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name=file1)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_target.to_excel(writer, index=False)
        # No need to call writer.save() — it's handled by the context manager
        st.download_button("Download Excel", buffer.getvalue(), file_name=file1.replace('.csv', '.xlsx'))


# === CUSTOM TAB 1: Upload Your Own Data and Compare ===

with tabs[9]:
    st.subheader("Upload Your Own Time Series and Compare with Indicators")

    uploaded_csv = st.file_uploader("Upload CSV (Columns: Date, Value)", type=["csv"], key="upload_custom_csv")
    if uploaded_csv:
        try:
            df_custom = pd.read_csv(uploaded_csv, parse_dates=[0])
            df_custom.columns = ['Date', 'Value']
            st.success("Custom CSV loaded successfully.")

            country_custom = st.selectbox("Select Country", countries, key="custom_country")
            files_custom = os.listdir(os.path.join("Data", country_custom))
            selected_custom_inds = st.multiselect("Select Indicator CSV(s)", [f for f in files_custom if f.endswith('.csv')],
                                                  key="custom_indicators")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_custom['Date'], y=df_custom['Value'],
                                     mode='lines+markers', name="Uploaded Data"))

            for f in selected_custom_inds:
                df_i = load_data(os.path.join("Data", country_custom, f))
                fig.add_trace(go.Scatter(x=df_i['Reference Period'], y=df_i['Actual'],
                                         mode='lines+markers', name=f.replace('.csv', '')))

            fig.update_layout(title="Custom Data vs Selected Indicators", xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not process uploaded file: {e}")

# === CUSTOM TAB 2: Rate Cycle Regimes with Indicators ===

with tabs[10]:
    st.subheader("Fed Rate Cycle Regimes and Overlayed Indicators")

    try:
        fed_df = pd.read_csv("FEDFUNDS (1).csv", parse_dates=[0])
        fed_df.columns = ['Date', 'Rate']

        # Identify Rate Hike / Cut Regimes
        fed_df = fed_df.sort_values('Date').reset_index(drop=True)
        fed_df['Diff'] = fed_df['Rate'].diff()
        fed_df['Regime'] = np.where(fed_df['Diff'] > 0, 'Hike', 'Cut')
        fed_df['Group'] = (fed_df['Regime'] != fed_df['Regime'].shift()).cumsum()

        fig = go.Figure()

        # Shade regimes
        for _, group_df in fed_df.groupby('Group'):
            start = group_df['Date'].min()
            end = group_df['Date'].max()
            color = 'rgba(255,0,0,0.1)' if group_df['Regime'].iloc[0] == 'Hike' else 'rgba(0,0,255,0.1)'
            fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.4, line_width=0)

        # Plot Fed Rate
        fig.add_trace(go.Scatter(x=fed_df['Date'], y=fed_df['Rate'],
                                 name='Fed Funds Rate', line=dict(color='black')))

        # Overlay Selected Indicators
        regime_country = st.selectbox("Select Country", countries, key="regime_country")
        regime_files = os.listdir(os.path.join("Data", regime_country))
        regime_indicators = st.multiselect("Select Indicator CSV(s)", [f for f in regime_files if f.endswith('.csv')],
                                           key="regime_inds")

        for ind_file in regime_indicators:
            df_i = load_data(os.path.join("Data", regime_country, ind_file))
            fig.add_trace(go.Scatter(x=df_i['Reference Period'], y=df_i['Actual'],
                                     mode='lines', name=ind_file.replace('.csv','')))

        fig.update_layout(title="Fed Rate Cycles with Selected Indicators",
                          xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load Fed Rate file or plot: {e}")
