import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io

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

# # === Global Filter Controls ===
# st.sidebar.markdown("### \U0001F5C2 Filter Timeframe")

# df_target['Reference Period'] = pd.to_datetime(df_target['Reference Period'])

# min_date, max_date = df_target['Reference Period'].min().date(), df_target['Reference Period'].max().date()

# start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
# end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# max_n = len(df_target)
# n_obs = st.sidebar.slider("Or show last N observations", min_value=5, max_value=max_n, value=30, step=1)

# # === Sync Filters ===
# apply_n_obs = st.sidebar.checkbox("Apply N observations filter", value=False)

# if apply_n_obs:
#     df_target_filtered = df_target.sort_values('Reference Period').tail(n_obs)
# else:
#     df_target_filtered = df_target[(df_target['Reference Period'] >= pd.to_datetime(start_date)) &
#                                    (df_target['Reference Period'] <= pd.to_datetime(end_date))]

# === Global Time Filter (Visible Above Tabs) ===
st.markdown("### ⏱️ Select Time Range for All Tabs")

min_date, max_date = df_target['Reference Period'].min(), df_target['Reference Period'].max()

selected_range = st.slider(
    "Filter by Reference Period",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="MMM YYYY"
)

# Add caption to show selected range
st.caption(f"Showing data from {selected_range[0].strftime('%b %Y')} to {selected_range[1].strftime('%b %Y')}")

# Filter df_target and soft indicators
df_target_filtered = df_target[
    (df_target['Reference Period'] >= selected_range[0]) &
    (df_target['Reference Period'] <= selected_range[1])
].copy()

df_softs_filtered = []
for df_soft in df_softs:
    df_soft_filtered = df_soft[
        (df_soft['Reference Period'] >= selected_range[0]) &
        (df_soft['Reference Period'] <= selected_range[1])
    ].copy()
    df_softs_filtered.append(df_soft_filtered)


# === Tabs ===
tabs = st.tabs([
    "\U0001F4C8 Time Series", "\U0001F4C5 Seasonality", "\U0001F4C9 Surprise Distribution",
    "\U0001F52C Multi-Indicator Prediction", "\U0001F30D Cross-Country Comparison",
    "\U0001F310 Intermarket Lag Analysis", "\U0001F4B2 Market Reaction", "⏰ Forecast Next Release", "\U0001F4BE Export"])

# === Time Series ===
with tabs[0]:
    st.subheader("Actual vs Forecast - Tier 1 Indicator")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_target_filtered['Reference Period'], df_target_filtered['Actual'], label='Actual', color='blue')
    ax.plot(df_target_filtered['Reference Period'], df_target_filtered['Median_Forecast'], label='Median Forecast', linestyle='--', color='orange')
    ax.legend()
    ax.set_title(file1.replace('.csv', ''))
    st.pyplot(fig)

# === Seasonality ===
with tabs[1]:
    st.subheader("Monthly Surprise Seasonality")
    monthly = df_target_filtered.groupby('Month')['Surprise'].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=monthly.index, y=monthly.values, ax=ax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylabel("Avg Surprise")
    st.pyplot(fig)

# === Surprise Distribution ===
with tabs[2]:
    st.subheader("Surprise Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_target_filtered['Surprise'], kde=True, ax=ax)
    ax.set_title("Actual - Forecast Distribution")
    st.pyplot(fig)

# === Multi-Indicator Predictive Model ===
with tabs[3]:
    st.subheader("Predict Tier-1 Indicator from Soft Indicators")
    df_merge = df_target_filtered[['Reference Period', 'Actual']].rename(columns={'Actual': 'Target'})
    for i, df_soft in enumerate(df_softs_filtered):
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

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_merge['Reference Period'], y, label='Actual', color='blue')
            ax.plot(df_merge['Reference Period'], prediction, label='Predicted', color='red')
            ax.set_title(f"Prediction of {file1.replace('.csv', '')}")
            ax.legend()
            st.pyplot(fig)

            st.write("\U0001F4CB **Feature Importance**")
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(imp)
    else:
        st.warning("Not enough overlapping data to build model.")

# === Cross-Country ===
with tabs[4]:
    st.subheader("Compare Indicator Across Countries")
    country2 = st.selectbox("Select Country to Compare", [c for c in countries if c != country])
    files2 = os.listdir(os.path.join("Data", country2))
    match_file = st.selectbox("Select Same Indicator CSV", [f for f in files2 if f.endswith(".csv")])

    try:
        df_other = load_data(os.path.join("Data", country2, match_file))
        df_other_filtered = df_other[df_other['Reference Period'].isin(df_target_filtered['Reference Period'])]
        merged = pd.merge(df_target_filtered[['Reference Period', 'Actual']],
                          df_other_filtered[['Reference Period', 'Actual']],
                          on='Reference Period', suffixes=(f'_{country}', f'_{country2}'))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(merged['Reference Period'], merged[f'Actual_{country}'], label=country)
        ax.plot(merged['Reference Period'], merged[f'Actual_{country2}'], label=country2)
        ax.legend()
        ax.set_title(f"{file1.replace('.csv', '')} Comparison")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not load file: {e}")

# === Intermarket Lag ===
with tabs[5]:
    st.subheader("Intermarket Lead-Lag Correlation")

    country_lag = st.selectbox("Select Country for Comparison", countries)
    lag_folder_path = os.path.join("Data", country_lag)
    lag_files = [f for f in os.listdir(lag_folder_path) if f.endswith(".csv")]
    lag_file = st.selectbox("Select Indicator CSV", lag_files)

    try:
        df_lead = df_target_filtered[['Reference Period', 'Actual']].rename(columns={"Actual": "Lead"})
        df_lag = load_data(os.path.join(lag_folder_path, lag_file))
        df_lag = df_lag[['Reference Period', 'Actual']].rename(columns={"Actual": "Lag"})

        max_lag = 12
        correlations = {}

        for lag in range(-max_lag, max_lag + 1):
            shifted = df_lag.copy()
            shifted['Reference Period'] = shifted['Reference Period'] + pd.DateOffset(months=lag)
            merged = pd.merge(df_lead, shifted, on='Reference Period')
            if len(merged) >= 3:
                corr = merged['Lead'].corr(merged['Lag'])
                correlations[lag] = corr

        best_lag = max(correlations, key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_lag]

        st.write(f"📈 Highest correlation at **{best_lag:+} month(s)** lag: **{best_corr:.2f}**")

        fig, ax = plt.subplots()
        ax.plot(list(correlations.keys()), list(correlations.values()), marker='o')
        ax.axvline(x=best_lag, color='red', linestyle='--', label=f'Max Corr @ {best_lag:+}m')
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Correlation")
        ax.set_title("Correlation vs Lag")
        ax.legend()
        st.pyplot(fig)

        df_lag['Reference Period'] = df_lag['Reference Period'] + pd.DateOffset(months=best_lag)
        merged = pd.merge(df_lead, df_lag, on='Reference Period')
        fig2, ax2 = plt.subplots()
        ax2.plot(merged['Reference Period'], merged['Lead'], label=f"{country} (Lead)")
        ax2.plot(merged['Reference Period'], merged['Lag'], label=f"{country_lag} (Lag, {best_lag:+}m)")
        ax2.legend()
        ax2.set_title("Best-Aligned Series Based on Lag")
        st.pyplot(fig2)

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
            fig, ax = plt.subplots()
            sns.scatterplot(data=merged, x='Surprise', y='Return', ax=ax)
            ax.set_title("Market Return vs Economic Surprise")
            st.pyplot(fig)
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
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_all['Reference Period'], df_all['Target'], label='Actual', color='blue')
        ax.plot(df_all['Reference Period'], df_all['Prediction'], label='Predicted', color='red', linestyle='--')
        ax.axvline(x=last_date, color='gray', linestyle=':', label='Forecast Start')

        ax.fill_between(df_forecast['Reference Period'], df_forecast['Lower'], df_forecast['Upper'],
                        color='red', alpha=0.2, label='±1 Std Dev')

        ax.set_title(f"Forecast with Manual Inputs - {file1.replace('.csv', '')}")
        ax.legend()
        st.pyplot(fig)

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

