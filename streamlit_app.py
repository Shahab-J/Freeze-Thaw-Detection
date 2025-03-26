import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("Freeze-Thaw Detection")

st.write(
    "This app identifies freeze-thaw transitions in temperature data. "
    "Upload a CSV file with date and temperature columns, or use the example data."
)

# File uploader for user data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Read user-provided dataset
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    # Use built-in example data when no file is uploaded
    st.info("No file uploaded. Using example data for demonstration.")
    dates = pd.date_range(start="2025-01-01", periods=14, freq='D')
    temps = [-5, -3, -1, 1, 4, 2, -2, -6, -1, 1, 3, -4, 0, 5]  # Example temperature series
    df = pd.DataFrame({"date": dates, "temperature": temps})

# Ensure proper dtypes
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
else:
    st.error("No 'date' column found in data. Please include dates for analysis.")
    st.stop()

# UI control for freezing threshold
threshold = st.slider("Freezing threshold (°C)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)

# Determine frozen/thawed state
df['frozen'] = df['temperature'] <= threshold

# Compute transitions: +1 = thaw->freeze, -1 = freeze->thaw
states = df['frozen'].astype(int).to_numpy()
transitions = np.diff(states, prepend=states[0])  # prepend first state to align lengths

# Identify transition events
freeze_to_thaw_events = np.where(transitions == -1)[0]  # indices where freeze->thaw
thaw_to_freeze_events = np.where(transitions == 1)[0]   # indices where thaw->freeze

# Count transitions
freeze_to_thaw_count = len(freeze_to_thaw_events)
thaw_to_freeze_count = len(thaw_to_freeze_events)

# Display summary metrics
st.markdown(f"**Freeze-to-thaw transitions** (freeze → thaw events): **{freeze_to_thaw_count}**")
st.markdown(f"**Thaw-to-freeze transitions** (thaw → freeze events): **{thaw_to_freeze_count}**")

# If any transitions, list their dates
if freeze_to_thaw_count > 0:
    ft_dates = df.loc[freeze_to_thaw_events, 'date'].dt.strftime("%Y-%m-%d").tolist()
    st.write(f"Freeze → Thaw on: {', '.join(ft_dates)}")
if thaw_to_freeze_count > 0:
    tf_dates = df.loc[thaw_to_freeze_events, 'date'].dt.strftime("%Y-%m-%d").tolist()
    st.write(f"Thaw → Freeze on: {', '.join(tf_dates)}")

# Create a line chart of temperature over time
line_chart = alt.Chart(df).mark_line(color='gray').encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('temperature:Q', title='Temperature (°C)')
)

# Mark transition points on the chart
points = alt.Chart(df).mark_point(size=80).encode(
    x='date:T',
    y='temperature:Q',
    color=alt.condition(df['frozen'], alt.value('blue'), alt.value('red')),
    shape=alt.ShapeValue('triangle-up')  # use triangle markers for emphasis
).transform_filter(
    # Filter points to only those indices in transition events
    alt.FieldOneOfPredicate(field='index', oneOf=freeze_to_thaw_events.tolist() + thaw_to_freeze_events.tolist())
).encode(tooltip=['date:T', 'temperature:Q'])

# Combine line and points
chart = line_chart + points
st.altair_chart(chart, use_container_width=True)

# Show the data table with freeze/thaw indicator
st.write("**Data Preview:**")
st.dataframe(df[['date', 'temperature', 'frozen']].head(20))
