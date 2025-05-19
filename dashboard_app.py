import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import numpy as np

# Load data
data = pd.read_csv("Housing.csv")

# Handle missing data
data.dropna(inplace=True)

# Feature engineering
data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']
data['households_rooms'] = data['total_rooms'] / data['households']
data['population_per_household'] = data['population'] / data['households']
data['price_per_room'] = data['median_house_value'] / data['total_rooms']

# Normalize latitude and longitude for better visualization
data['lat_norm'] = (data['latitude'] - 32) / 10
data['lon_norm'] = (data['longitude'] + 124) / 10

# Create price categories for filtering
price_bins = [0, 100000, 200000, 300000, 400000, 500000, float('inf')]
price_labels = ['<100k', '100k-200k', '200k-300k', '300k-400k', '400k-500k', '>500k']
data['price_category'] = pd.cut(data['median_house_value'], bins=price_bins, labels=price_labels)

# Initialize the app with a theme
app = dash.Dash(
    __name__, 
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ]
)
server = app.server
app.title = "California Housing Dashboard"

# Custom CSS
colors = {
    'background': '#e6f2ff',   # Light sky blue – subtle and fresh
    'text': '#2c3e50',         # Deep navy – strong contrast for readability
    'primary': '#1abc9c',      # Teal – resembles ocean/river hues
    'secondary': '#16a085',    # Deep teal – accents and buttons
    'panel': '#ffffff',        # Clean white panels
}


# App layout with better styling
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("California Housing Data Dashboard", 
                style={'color': colors['text'], 'textAlign': 'center', 'marginBottom': '10px'}),
        html.P("Interactive visualization of California housing market data", 
               style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '18px'})
    ], style={'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.H3("Filters", style={'marginBottom': '15px'}),
            
            # Price Range Slider
            html.Label("Price Range ($)", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.RangeSlider(
                id='price-range-slider',
                min=int(data['median_house_value'].min()),
                max=int(data['median_house_value'].max()),
                step=10000,
                marks={
                    int(data['median_house_value'].min()): {'label': f"${int(data['median_house_value'].min()//1000)}k"},
                    100000: {'label': '$100k'},
                    200000: {'label': '$200k'},
                    300000: {'label': '$300k'},
                    400000: {'label': '$400k'},
                    500000: {'label': '$500k'},
                    int(data['median_house_value'].max()): {'label': f"${int(data['median_house_value'].max()//1000)}k"}
                },
                value=[int(data['median_house_value'].min()), int(data['median_house_value'].max())],
                allowCross=False
            ),
            
            # Ocean Proximity Dropdown
            html.Label("Ocean Proximity", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='ocean-proximity-dropdown',
                options=[{'label': loc, 'value': loc} for loc in data['ocean_proximity'].unique()],
                value=list(data['ocean_proximity'].unique()),
                multi=True
            ),
            
            # Feature Selection
            html.Label("Select Features to Analyze", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[
                    {'label': 'Median House Value', 'value': 'median_house_value'},
                    {'label': 'Median Income', 'value': 'median_income'},
                    {'label': 'Total Rooms', 'value': 'total_rooms'},
                    {'label': 'Total Bedrooms', 'value': 'total_bedrooms'},
                    {'label': 'Population', 'value': 'population'},
                    {'label': 'Households', 'value': 'households'},
                    {'label': 'Bedroom Ratio', 'value': 'bedroom_ratio'},
                    {'label': 'Rooms per Household', 'value': 'households_rooms'},
                    {'label': 'Population per Household', 'value': 'population_per_household'},
                    {'label': 'Price per Room', 'value': 'price_per_room'}
                ],
                value='median_house_value'
            ),
            
            # Reset Button
            html.Button(
                'Reset Filters', 
                id='reset-button', 
                n_clicks=0,
                style={
                    'marginTop': '20px',
                    'backgroundColor': colors['secondary'],
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 15px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold'
                }
            ),
        ], style={'width': '100%', 'marginBottom': '20px'}),
        
        # Key Stats
        html.Div(id='key-stats', className='stats-container')
        
    ], style={'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'marginTop': '20px'}),

    # Visualization Panels
    html.Div([
        # Map and histogram in the first row
        html.Div([
            # Map Panel
            html.Div([
                html.H3("Geographic Distribution", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='scatter-map',
                    config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d']}
                )
            ], style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'width': '58%', 'display': 'inline-block'}),
            
            # Histogram Panel
            html.Div([
                html.H3("Distribution Analysis", style={'textAlign': 'center'}),
                dcc.Graph(id='histogram')
            ], style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'width': '38%', 'float': 'right'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'}),
        
        # Scatter plot and correlation heatmap in the second row
        html.Div([
            # Scatter Plot Panel
            html.Div([
                html.H3("Feature Relationships", style={'textAlign': 'center'}),
                html.Div([
                    html.Label("X-axis:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[
                            {'label': 'Median Income', 'value': 'median_income'},
                            {'label': 'Total Rooms', 'value': 'total_rooms'},
                            {'label': 'Total Bedrooms', 'value': 'total_bedrooms'},
                            {'label': 'Population', 'value': 'population'},
                            {'label': 'Households', 'value': 'households'},
                            {'label': 'Bedroom Ratio', 'value': 'bedroom_ratio'},
                            {'label': 'Rooms per Household', 'value': 'households_rooms'},
                            {'label': 'Population per Household', 'value': 'population_per_household'}
                        ],
                        value='median_income',
                        clearable=False,
                        style={'width': '48%', 'display': 'inline-block'}
                    ),
                    html.Label("Y-axis:", style={'fontWeight': 'bold', 'marginRight': '10px', 'marginLeft': '20px'}),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[
                            {'label': 'Median House Value', 'value': 'median_house_value'},
                            {'label': 'Median Income', 'value': 'median_income'},
                            {'label': 'Total Rooms', 'value': 'total_rooms'},
                            {'label': 'Total Bedrooms', 'value': 'total_bedrooms'},
                            {'label': 'Population', 'value': 'population'},
                            {'label': 'Households', 'value': 'households'},
                            {'label': 'Bedroom Ratio', 'value': 'bedroom_ratio'}
                        ],
                        value='median_house_value',
                        clearable=False,
                        style={'width': '48%', 'float': 'right'}
                    ),
                ], style={'marginBottom': '10px'}),
                dcc.Graph(id='scatter-plot')
            ], style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'width': '58%', 'display': 'inline-block'}),
            
            # Correlation Heatmap Panel
            html.Div([
                html.H3("Correlation Matrix", style={'textAlign': 'center'}),
                dcc.Graph(id='correlation-heatmap')
            ], style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'width': '38%', 'float': 'right'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'}),
        
        # Ocean Proximity Analysis
        html.Div([
            html.H3("Ocean Proximity Analysis", style={'textAlign': 'center'}),
            dcc.Graph(id='ocean-proximity-boxplot')
        ], style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'marginTop': '20px'})
    ]),
    
    # Footer
    html.Div([
        html.P("California Housing Dashboard © 2025 | Data source: California Housing Dataset", 
               style={'textAlign': 'center', 'color': colors['text']})
    ], style={'marginTop': '30px', 'padding': '10px', 'backgroundColor': colors['panel'], 'borderRadius': '10px'})
    
], style={'backgroundColor': colors['background'], 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

# Callback for filtering data
@callback(
    [Output('scatter-map', 'figure'),
     Output('histogram', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('ocean-proximity-boxplot', 'figure'),
     Output('key-stats', 'children')],
    [Input('price-range-slider', 'value'),
     Input('ocean-proximity-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('reset-button', 'n_clicks')]
)
def update_figures(price_range, ocean_proximity, selected_feature, x_axis, y_axis, n_clicks):
    # Handle the reset button
    ctx = dash.callback_context
    if ctx.triggered and 'reset-button' in ctx.triggered[0]['prop_id']:
        # This will trigger another callback with default values
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Filter data based on selections
    filtered_data = data[
        (data['median_house_value'] >= price_range[0]) & 
        (data['median_house_value'] <= price_range[1]) &
        (data['ocean_proximity'].isin(ocean_proximity))
    ]
    
    if filtered_data.empty:
        # If no data matches filters, return empty figures
        return create_empty_figures()
    
    # Create scatter map
    scatter_map_fig = create_scatter_map(filtered_data, selected_feature)
    
    # Create histogram
    histogram_fig = create_histogram(filtered_data, selected_feature)
    
    # Create scatter plot
    scatter_plot_fig = create_scatter_plot(filtered_data, x_axis, y_axis)
    
    # Create correlation heatmap
    correlation_fig = create_correlation_heatmap(filtered_data)
    
    # Create boxplot for ocean proximity analysis
    boxplot_fig = create_boxplot(filtered_data, selected_feature)
    
    # Create key stats
    stats_panel = create_stats_panel(filtered_data)
    
    return scatter_map_fig, histogram_fig, scatter_plot_fig, correlation_fig, boxplot_fig, stats_panel

# Helper function to create empty figures
def create_empty_figures():
    empty_fig = {
        'data': [],
        'layout': {
            'title': 'No data matches your filter criteria',
            'xaxis': {'title': ''},
            'yaxis': {'title': ''}
        }
    }
    return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, html.Div("No data available for your selection")

# Function to create scatter map
def create_scatter_map(filtered_data, selected_feature):
    # Create a sample if the dataset is too large for smooth rendering
    if len(filtered_data) > 1000:
        map_data = filtered_data.sample(1000)
    else:
        map_data = filtered_data
    
    # Create colorscale based on the feature
    max_value = filtered_data[selected_feature].max()
    min_value = filtered_data[selected_feature].min()
    
    # Get nice label for the feature
    feature_label = selected_feature.replace('_', ' ').title()
    
    # Create the map figure
    fig = px.scatter_mapbox(
        map_data, 
        lat="latitude", 
        lon="longitude", 
        color=selected_feature,
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=[min_value, max_value],
        zoom=5.5,
        center={"lat": 37.5, "lon": -119.5},  # Center of California
        mapbox_style="carto-positron",
        height=500,
        opacity=0.7,
        hover_data={
            'median_house_value': True,
            'median_income': True,
            'ocean_proximity': True,
            'total_rooms': True,
            'total_bedrooms': True,
            'population': True,
            'latitude': False,
            'longitude': False
        }
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title=feature_label
        )
    )
    
    return fig

# Function to create histogram
def create_histogram(filtered_data, selected_feature):
    # Get nice label for the feature
    feature_label = selected_feature.replace('_', ' ').title()
    
    fig = px.histogram(
        filtered_data, 
        x=selected_feature, 
        nbins=50,
        marginal='box',
        color_discrete_sequence=['#3366cc'],
        opacity=0.7
    )
    
    fig.update_layout(
        title=f'Distribution of {feature_label}',
        xaxis_title=feature_label,
        yaxis_title='Count',
        bargap=0.05,
        height=500
    )
    
    return fig

# Function to create scatter plot
def create_scatter_plot(filtered_data, x_axis, y_axis):
    # Get nice labels for axes
    x_label = x_axis.replace('_', ' ').title()
    y_label = y_axis.replace('_', ' ').title()
    
    # Create a sample if the dataset is too large
    if len(filtered_data) > 2000:
        plot_data = filtered_data.sample(2000)
    else:
        plot_data = filtered_data
    
    fig = px.scatter(
        plot_data,
        x=x_axis,
        y=y_axis,
        color='ocean_proximity',
        opacity=0.7,
        height=500,
        trendline='ols',  # Add regression line
        trendline_color_override='red'
    )
    
    # Add regression equation and correlation coefficient
    corr = filtered_data[x_axis].corr(filtered_data[y_axis])
    
    fig.update_layout(
        title=f'Relationship between {x_label} and {y_label}<br>Correlation: {corr:.2f}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title='Ocean Proximity'
    )
    
    return fig

# Function to create correlation heatmap
def create_correlation_heatmap(filtered_data):
    # Select numeric columns for correlation
    numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])
    
    # Drop geographic coordinates and computed columns for clarity
    cols_to_exclude = ['latitude', 'longitude', 'lat_norm', 'lon_norm']
    numeric_data = numeric_data.drop(columns=[c for c in cols_to_exclude if c in numeric_data.columns])
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text:.2f}',
        hoverongaps=False
    ))
    
    # Update layout
    fig.update_layout(
        height=500,
        title='Feature Correlation Matrix',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(corr_matrix.columns))),
            ticktext=[col.replace('_', '<br>') for col in corr_matrix.columns]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(corr_matrix.columns))),
            ticktext=corr_matrix.columns
        )
    )
    
    return fig

# Function to create boxplot
def create_boxplot(filtered_data, selected_feature):
    # Get nice label for the feature
    feature_label = selected_feature.replace('_', ' ').title()
    
    # Create boxplot grouped by ocean proximity
    fig = px.box(
        filtered_data,
        x='ocean_proximity',
        y=selected_feature,
        color='ocean_proximity',
        height=500,
        points='outliers'  # Show outliers only
    )
    
    # Calculate and add mean values as text
    for category in filtered_data['ocean_proximity'].unique():
        mean_val = filtered_data[filtered_data['ocean_proximity'] == category][selected_feature].mean()
        fig.add_annotation(
            x=category,
            y=mean_val,
            text=f'Mean: {mean_val:.2f}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40
        )
    
    fig.update_layout(
        title=f'{feature_label} by Ocean Proximity',
        xaxis_title='Ocean Proximity',
        yaxis_title=feature_label,
        showlegend=False
    )
    
    return fig

# Function to create stats panel
def create_stats_panel(filtered_data):
    # Calculate key statistics
    total_records = len(filtered_data)
    avg_price = filtered_data['median_house_value'].mean()
    med_price = filtered_data['median_house_value'].median()
    min_price = filtered_data['median_house_value'].min()
    max_price = filtered_data['median_house_value'].max()
    avg_income = filtered_data['median_income'].mean()
    avg_rooms = filtered_data['total_rooms'].mean()
    
    # Create stats components
    stats_divs = [
        html.Div([
            html.H3("Key Statistics", style={'marginBottom': '15px'}),
            html.Div([
                # First row
                html.Div([
                    stat_card("Total Records", f"{total_records:,}", "database"),
                    stat_card("Avg. House Value", f"${avg_price:,.2f}", "home"),
                    stat_card("Median House Value", f"${med_price:,.2f}", "trending-up"),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '15px'}),
                
                # Second row
                html.Div([
                    stat_card("Min House Value", f"${min_price:,.2f}", "arrow-down"),
                    stat_card("Max House Value", f"${max_price:,.2f}", "arrow-up"),
                    stat_card("Avg. Income", f"${avg_income:,.2f}", "dollar-sign"),
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ])
        ])
    ]
    
    return stats_divs

# Helper function to create stat cards
def stat_card(title, value, icon=None):
    return html.Div([
        html.H4(title, style={'margin': '0', 'fontSize': '14px', 'color': '#666'}),
        html.P(value, style={'margin': '5px 0', 'fontSize': '18px', 'fontWeight': 'bold'})
    ], style={
        'backgroundColor': '#f0f8ff',
        'padding': '15px',
        'borderRadius': '8px',
        'width': '30%',
        'textAlign': 'center',
        'boxShadow': '0px 0px 5px rgba(0,0,0,0.1)'
    })

# Add callback for reset button
@callback(
    [Output('price-range-slider', 'value'),
     Output('ocean-proximity-dropdown', 'value'),
     Output('feature-dropdown', 'value'),
     Output('x-axis-dropdown', 'value'),
     Output('y-axis-dropdown', 'value')],
    [Input('reset-button', 'n_clicks')]
)
def reset_filters(n_clicks):
    if n_clicks > 0:
        return [
            [int(data['median_house_value'].min()), int(data['median_house_value'].max())],
            list(data['ocean_proximity'].unique()),
            'median_house_value',
            'median_income',
            'median_house_value'
        ]
    raise PreventUpdate

# Run the app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)