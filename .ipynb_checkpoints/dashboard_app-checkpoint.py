import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load data
data = pd.read_csv("Housing.csv")

# Handle missing data
data.dropna(inplace=True)

# Basic feature engineering
data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']
data['households_rooms'] = data['total_rooms'] / data['households']

# Initialize the app
app = dash.Dash(__name__)
app.title = "California Housing Dashboard"

# Layout
app.layout = html.Div([
    html.H1("California Housing Data Dashboard"),
    
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in data.select_dtypes(include=['float64', 'int64']).columns],
        value='median_house_value'
    ),

    dcc.Graph(id='histogram'),

    dcc.Graph(
        id='scatter-map',
        figure=px.scatter_mapbox(
            data, 
            lat="latitude", 
            lon="longitude", 
            color="median_house_value",
            zoom=5,
            mapbox_style="carto-positron",
            height=500
        )
    )
])

# Callback for interactive histogram
@app.callback(
    dash.dependencies.Output('histogram', 'figure'),
    [dash.dependencies.Input('feature-dropdown', 'value')]
)
def update_histogram(selected_feature):
    fig = px.histogram(data, x=selected_feature, nbins=50)
    fig.update_layout(title=f'Distribution of {selected_feature}')
    return fig

# Run the app
if __name__ == '__main__':
app.run(debug=True)

