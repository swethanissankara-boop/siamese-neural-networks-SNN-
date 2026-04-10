import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
import json

# Load pre-saved data
embeddings_2d = np.load('data/embeddings_2d.npy', allow_pickle=True)
import json
with open('data/images_base64.json', 'r') as f:
    images_base64 = json.load(f)


y_test = np.load('data/y_test.npy')
# Initialize Dash app
app = dash.Dash(__name__)
# Create initial embedding scatter plot
fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    color=y_test.astype(str),
    labels={'x': 't-SNE Dim 1', 'y': 't-SNE Dim 2', 'color': 'Class'},
    title='t-SNE Visualization of Siamese Network Embeddings'
)

app.layout = html.Div([
    html.H1('Siamese Network Embedding Dashboard'),

    dcc.Graph(
        id='embedding-plot',
        figure=fig,
        style={'width': '80vw', 'height': '80vh'}
    ),

    html.Div(id='selected-image-container', style={'marginTop': '20px', 'textAlign': 'center'})
])


@app.callback(
    Output('selected-image-container', 'children'),
    Input('embedding-plot', 'clickData')
)
def display_selected_image(clickData):
    if clickData is None:
        return "Click on a point in the plot to see the test image and class label."

    # Get index of clicked point
    point_index = clickData['points'][0]['pointIndex']

    # Get base64 image and class
    img_b64 = images_base64[point_index]
    class_label = y_test[point_index]

    return html.Div([
        html.Img(src=img_b64, style={'height': '128px', 'border': '2px solid black'}),
        html.P(f'Class Label: {class_label}')
    ])


if __name__ == '__main__':
    app.run(debug=True)
