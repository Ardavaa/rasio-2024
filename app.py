import streamlit as st
import pandas as pd
import numpy as np
from functools import reduce
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.markdown(f"<h1 style='text-align: center;'>World GDP Growth</h1>", unsafe_allow_html=True)

FDI = pd.read_csv('main_data-processed/FDI_melt.csv')
GDP = pd.read_csv('main_data-processed/GDP_melt.csv')
Inflation = pd.read_csv('main_data-processed/inflation_melt.csv')
Trade = pd.read_csv('main_data-processed/trade_melt.csv')
Unemployment = pd.read_csv('main_data-processed/unemployment_melt.csv')
WDI = pd.read_csv('main_data-processed/WDI_melt.csv')

FDI.rename(columns={'Value': 'Value_FDI'}, inplace=True)
GDP.rename(columns={'Value': 'GDP'}, inplace=True)
Inflation.rename(columns={'Value': 'Value_Inflation'}, inplace=True)
Trade.rename(columns={'Value': 'Value_Trade'}, inplace=True)
Unemployment.rename(columns={'Value': 'Value_Unemployment'}, inplace=True)
WDI.rename(columns={'Value': 'Value_WDI'}, inplace=True)

dataframes = [FDI, GDP, Inflation, Trade, Unemployment, WDI]

merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Country Name', 'Year'], how='outer'), dataframes)

col1, col2 = st.columns([3, 1])

selected_year = st.slider('Select a Year', min_value=2000, max_value=2023, value=2000, step=1)

with col1:
    st.markdown('### World Map')
    
    gdp_filtered = GDP[GDP['Year'] == selected_year].dropna(subset=['GDP'])

    if gdp_filtered.empty:
        st.warning("No GDP data available for the selected year.")
    else:
        fig = px.choropleth(gdp_filtered,
                            locations='Country Name',
                            locationmode='country names',
                            color='GDP',
                            hover_name='Country Name',
                            color_continuous_scale=['red', 'white', 'green'],
                            range_color=(-5, 5),
                            )

        fig.update_layout(
            width=1500,  # Adjust width
            height=450,  # Adjust height
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig)

with col2:
    st.markdown('### Country List')
    
    filtered_df = merged_df[merged_df['Year'] == selected_year]
    
    table_gdp = filtered_df[['Country Name', 'GDP']].dropna(subset=['GDP'])
    
    if table_gdp.empty:
        st.warning("No GDP data available to display.")
    else:
        st.dataframe(table_gdp)

st.markdown(f"<h2 style='text-align: center;'>Data Indicator</h2>", unsafe_allow_html=True)

with st.container():
    average_fdi = merged_df[merged_df['Year'] == selected_year]['Value_FDI'].mean()
    average_wdi = merged_df[merged_df['Year'] == selected_year]['Value_WDI'].mean()
    average_unemploy = merged_df[merged_df['Year'] == selected_year]['Value_Unemployment'].mean()
    average_trade = merged_df[merged_df['Year'] == selected_year]['Value_Trade'].mean()
    average_inflation = merged_df[merged_df['Year'] == selected_year]['Value_Inflation'].mean()

    mean_values_Value_FDI = merged_df.groupby('Year')['Value_FDI'].mean().reset_index()
    mean_values_Value_WDI = merged_df.groupby('Year')['Value_WDI'].mean().reset_index()
    mean_values_Value_Unemployment = merged_df.groupby('Year')['Value_Unemployment'].mean().reset_index()
    mean_values_Value_Trade = merged_df.groupby('Year')['Value_Trade'].mean().reset_index()
    mean_values_Value_Inflation = merged_df.groupby('Year')['Value_Inflation'].mean().reset_index()

    st.markdown("""
    <style>
        .indicator-container {
            background-color: #FFFFFF;
            border: 1px solid #CCCCCC;
            padding: 1rem;  /* Increased padding */
            border-radius: 5px;
            border-left: 0.5rem solid #y9AD8E1 !important;
            box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
            text-align: center;  /* Center text in container */
            margin-top: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create columns for metrics
    col_container1, col_container2, col_container3 = st.columns(3)

    # Define metrics
    metrics = [
        ("Foreign Direct Investment", average_fdi, mean_values_Value_FDI, 'FDI'),
        ("World Development Indicator", average_wdi, mean_values_Value_WDI, 'WDI'),
        ("Unemployment Rate", average_unemploy, mean_values_Value_Unemployment, 'Unemployment'),
    ]
    
    # Function to create and display the plot for each metric
    def create_plot(values, column, title, selected_year, show_marker=True):
        fig_plot = px.line(values, x='Year', y=f'Value_{column}', title=title)

        # Add markers if required
        if show_marker:
            fig_plot.update_traces(mode='lines+markers')  # Display both lines and markers
        
        # Add a marker for the selected year
        selected_year_value = values[values['Year'] == selected_year][f'Value_{column}']
        if not selected_year_value.empty:
            fig_plot.add_trace(go.Scatter(
                x=[selected_year],
                y=selected_year_value,
                mode='markers',
                marker=dict(size=12, color='red', symbol='cross'),
                name='Selected Year'
            ))
        
        fig_plot.update_layout(
            xaxis_title='Time Period',
            yaxis_title='Average Value',
            title_font_size=20,
            xaxis_title_font_size=18,
            yaxis_title_font_size=18,
        )

        return fig_plot

    # Display metrics
    for col, (label, value, values, column) in zip([col_container1, col_container2, col_container3], metrics):
        with col:
            st.markdown(f'<div class="indicator-container"><strong>{label} (Year {selected_year})</strong><br>{value:.2f}</div>', unsafe_allow_html=True)
            fig_plot = create_plot(values, column, f'{label} Graphic per Year (2000 - 2024)', selected_year, show_marker=True)
            st.plotly_chart(fig_plot)

    # Additional columns for Inflation and Trading Index
    kol1, kol2 = st.columns(2)
    with kol1:
        st.markdown(f'<div class="indicator-container"><strong>Inflation Rate (Year {selected_year})</strong><br>{average_inflation:.2f}</div>', unsafe_allow_html=True)
        fig_plot_inflation = create_plot(mean_values_Value_Inflation, 'Inflation', 'Inflation Graphic per Year (2000 - 2024)', selected_year, show_marker=True)
        st.plotly_chart(fig_plot_inflation)

    with kol2:
        st.markdown(f'<div class="indicator-container"><strong>Trading Index (Year {selected_year})</strong><br>{average_trade:.2f}</div>', unsafe_allow_html=True)
        fig_plot_trade = create_plot(mean_values_Value_Trade, 'Trade', 'Trading Graphic Index per Year (2000 - 2024)', selected_year, show_marker=True)
        st.plotly_chart(fig_plot_trade)

    

# Centered Subheader for Trend
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Trend (2000 - 2024)")

# Add a multi-select box for country selection
selected_countries = st.multiselect(
    "Select Countries to Display GDP Trends", 
    options=merged_df['Country Name'].unique(),  # List of unique country names
    default=["World", "Europe & Central Asia (IDA & IBRD countries)", "Fragile and conflict affected situations",]  # Default selection
)

if selected_countries:
    fig_gdp = px.line(
        merged_df[merged_df['Country Name'].isin(selected_countries)], 
        x='Year', 
        y='GDP', 
        color='Country Name', 
        labels={'GDP': 'Nilai GDP'},
        markers=True
    )

    st.plotly_chart(fig_gdp)
else:
    st.warning("Please select at least one country to display GDP trends.")


st.subheader("Average GDP Over Years & Forecasted GDP")

forecast_df = pd.read_csv('main_data-processed/merge_and_forecast.csv')

average_gdp = forecast_df.groupby('Year', as_index=False)['Value_GDP'].mean()

# Separate the data into highlighted years and others
highlight_years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
highlight_data = average_gdp[average_gdp['Year'].isin(highlight_years)]
normal_data = average_gdp[~average_gdp['Year'].isin(highlight_years)]

# Create the figure
fig_forecast_combined = go.Figure()

# Add the normal data trace
fig_forecast_combined.add_trace(go.Scatter(x=normal_data['Year'], y=normal_data['Value_GDP'],
                         mode='lines+markers',
                         name='Historical GDP'))

# Add the highlighted data trace
fig_forecast_combined.add_trace(go.Scatter(x=highlight_data['Year'], y=highlight_data['Value_GDP'],
                         mode='lines+markers',
                         name='Forecasted GDP',
                         line=dict(color='orange', width=2)))

# Update layout
fig_forecast_combined.update_layout(title='',
                  xaxis_title='Year',
                  yaxis_title='Average GDP Value',
                  template='plotly_white')

# Show the figure
st.plotly_chart(fig_forecast_combined)