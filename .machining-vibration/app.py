import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_table
import plotly.figure_factory as ff

from textwrap import dedent as d
import json

import pandas as pd
from PIL import Image
import os, base64
from pathlib import Path
import numpy as np
import scipy.stats as st
import flask
from dash_table.Format import Format

app = dash.Dash(__name__)
# css_directory = os.getcwd()
#
# stylesheets = ['styles.css']
# static_css_route = '/static/'
# @app.server.route('{}<stylesheet>'.format(static_css_route))
# def serve_stylesheet(stylesheet):
#     if stylesheet not in stylesheets:
#         raise Exception(
#             '"{}" is excluded from the allowed static files'.format(
#                 stylesheet
#             )
#         )
#     return flask.send_from_directory(css_directory, stylesheet)
# for stylesheet in stylesheets:
#     app.css.append_css({"external_url": "/static/{}".format(stylesheet)})

"""
'clearance_factor', 
'crest_factor', 
'impulse_factor',
'index_of_dispersion', 
'kurtosis', 
'rms', 
'shape_factor', 
'skew', 
'var',
'vpeak',
'label', 
'slotname', 
'time', 
'kesim_paramsexcel', 
'n_flutes',
'feed_per_tooth', 
'depth_of_cut', 
'spindle_speed', 
'feed_rate',
'slotno', 
'sample_no'
"""


def move_column(df, column, insert_idx):
    temp = df[column]
    df = df.drop([column], axis=1)
    df.insert(insert_idx, column, temp)
    return df


DEFAULT_INPUT_PARAMS = {'slotname': 'kanal51',
                        'slotno': 51,
                        'feature_name': 'slotno',
                        'spindle_speed': 3400,
                        'depth_of_cut': 8,
                        'sample_no': 1}


def input_handler(name, value):
    return value if value is not None else DEFAULT_INPUT_PARAMS[name]


df = pd.read_csv("./input/preprocessed_data/alu_v2/dash_data.csv")
df['slotno'] = df['slotname'].apply(lambda name: int(name.split('kanal')[-1]))
df['sample_no'] = df['slotno'].apply(lambda x: 1 if x % 2 == 1 else 2)
df = move_column(df, column='slotno', insert_idx=0)
df = move_column(df, column='sample_no', insert_idx=1)
# df = move_column(df, column='time', insert_idx=2)
# df = move_column(df, column='slotname', insert_idx=3)
df = move_column(df, column='depth_of_cut', insert_idx=2)
df = move_column(df, column='spindle_speed', insert_idx=3)
df = df.sort_values(by=['slotno'])

feature_names = list(df.columns)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

image_paths = {slotno: Path(f"./input/preprocessed_data/alu_v2/kanal{slotno}/00") for slotno in df['slotno'].values}
encoded_images = \
    {
        'time_acc': {slotno: base64.b64encode(open(image_paths[slotno] / "plot_acc.png", 'rb').read()) for slotno in
                     df['slotno'].values},
        'time_acoustic': {slotno: base64.b64encode(open(image_paths[slotno] / "plot_acoustic.png", 'rb').read()) for
                          slotno in df['slotno'].values},
        'fft_acc': {slotno: base64.b64encode(open(image_paths[slotno] / "plot_fft_acc.png", 'rb').read()) for slotno in
                    df['slotno'].values},
        'fft_acoustic': {slotno: base64.b64encode(open(image_paths[slotno] / "plot_fft_acoustic.png", 'rb').read()) for
                         slotno in df['slotno'].values},
        'spect_acc': {slotno: base64.b64encode(open(image_paths[slotno] / "plot_spectrogram_acc.png", 'rb').read()) for
                      slotno in df['slotno'].values},
        'spect_acoustic': {
            slotno: base64.b64encode(open(image_paths[slotno] / "plot_spectrogram_acoustic.png", 'rb').read()) for
            slotno in
            df['slotno'].values},
    }

sample_graph = dcc.Graph(
    figure={
        'data': [
            {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
            {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
        ],
        'layout': {
            'title': 'Dash Data Visualization'
        }
    }
)

data_table = dash_table.DataTable(
    id='data-table',
    columns=[{"name": i,
              "id": i,
              "format": Format(precision=3),
              "type": 'numeric' if i not in ['kesim_paramsexcel', 'slotname'] else 'text',
              "deletable": True,
              "selectable": True} for i in
             df.columns],
    data=df.to_dict('records'),
    fixed_rows={'headers': True, 'data': 0},
    # editable=True,
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    # column_selectable="single",
    # row_selectable="multi",
    # row_deletable=True,
    # selected_columns=[],
    # selected_rows=[],
    # page_action="native",
    # page_current=0,
    # page_size=10,
    style_table={'overflowX': 'scroll',
                 'overflowY': 'scroll'},
    # style_data_conditional=[
    #     {
    #         "if": {"column_id": 'slotno'},
    #         "backgroundColor": "#3D9970",
    #         'color': 'white'
    #     },
    # ],
    # style_header={
    #     'backgroundColor': "#3D9970",
    #     'fontWeight': 'bold',
    #     'minWidth': '0px', 'maxWidth': '10px',
    # },
    # style_data={
    #     'whiteSpace': 'normal',
    #     'height': 'auto',
    #     'width': 'auto'
    # },
    style_cell={
        'height': 'auto',
        'minWidth': '100px', 'maxWidth': '180px',
        'whiteSpace': 'normal'
    }
)

header_layout = html.Div([
    html.Div([
        html.A([
            html.Img(id="plotly-image",
                     # src="https://dash-gallery.plotly.host/dash-oil-and-gas/assets/dash-logo.png",
                     src=app.get_asset_url("tobb_etu__logo_white.png"),
                     style={"height": "60px", "width": "auto", "margin-bottom": "25px"}),
        ], href='https://www.etu.edu.tr')

    ], className="one-third column"),  # logo
    html.Div([
        html.Div([
            html.H3("Vibration Analysis Toolkit", style={"margin-bottom": "0px"}),
            html.H5("Data Visualization", style={"margin-top": "0px"}),
        ])
    ], id="title", className="one-half column"),  # title
    html.Div([
        html.A([
            html.Button("Learn More", id="learn-more-button")
        ], href="https://plot.ly/dash/pricing/"),
    ], id="button", className="one-third column"),  # learn more button
], id="header", className="row flex-display", style={"margin-bottom": "25px"})
left_column_layout = html.Div([
    html.Div([  # controls
        html.H6("Controls"),
    ], className="pretty_container"),
    html.Div([  # hover
        dash_table.DataTable(id='hover-table',
                             columns=[{"name": i, "id": i,
                                       } for i in ['feature', 'hover1', 'hover2']],

                             data=[],
                             )
    ], className="pretty_container", style={'display': 'flex', 'justify-content': 'center'})
], className="four columns")
right_column_layout = html.Div([
    html.Div([
        html.Div([
            html.H6("Heatmap"),
            dcc.Dropdown(id='zvalue-dropdown',
                         options=[{'label': f, 'value': f} for f in feature_names],
                         value=feature_names[0],
                         ),

        ], className="pretty_container five columns"),
        html.Div([
            html.H6("Info3"),
            dcc.RadioItems(
                id='sample-no',
                options=[{'label': i, 'value': i} for i in [1, 2]],
                value=1,
                labelStyle={'display': 'inline-block'}
            ),
        ], className="mini_container", id='info3'),
        html.Div([
            html.H6("Info4"),
            html.P("Info4 subnote")
        ], className="mini_container", id='info4'),
        html.Div([
            html.H6("Info5"),
            html.P("Info5 subnote")
        ], className="mini_container", id='info5'),
    ], id="info-container", className="row container-display"),
    html.Div([
        dcc.Graph(id='heatmap-graph'),
    ], className="pretty_container")
], id="right-column", className="eight columns")

app.layout = html.Div(
    [
        html.Div([
            html.Div([]),  # output-client-side
            header_layout,
            html.Div([
                left_column_layout,
                right_column_layout,
            ], className="row flex-display"),  # top-block
            html.Div([
                html.Div([
                    data_table,
                ], className="pretty_container twelve columns"),
            ], className="row flex-display"),  # data-table
            html.Div([
                html.Div(id="time-plot-1", className="pretty_container six columns"),
                html.Div(id="time-plot-2", className="pretty_container six columns")
            ], className="row flex-display"),  # time-plot
            html.Div([
                html.Div(id="fft-plot-1", className="pretty_container six columns"),
                html.Div(id="fft-plot-2", className="pretty_container six columns"),
            ], className="row flex-display"),  # fft-plot
            html.Div([
                html.Div(id="spectrogram-plot-1", className="pretty_container six columns"),
                html.Div(id="spectrogram-plot-2", className="pretty_container six columns"),
            ], className="row flex-display"),  # spectrogram-plot
        ], id="mainContainer", style={"display": "flex", "flex-direction": "column"}),

        # html.H1('Vibration Data Analysis'),
        #
        # html.Div(id="debug-output"),
        # html.Div([
        #     dcc.Tabs(id="tabs", value='tab-1', children=[
        #         dcc.Tab(label='Plot', value='tab-1'),
        #         dcc.Tab(label='Dataset', value='tab-2',
        #                 children=[
        #                     dash_table.DataTable(
        #                         id='data-table',
        #                         columns=[{"name": i, "id": i, "deletable": True, "selectable": True} for i in
        #                                  df.columns],
        #                         data=df.to_dict('records'),
        #                         editable=True,
        #                         filter_action="native",
        #                         sort_action="native",
        #                         sort_mode="multi",
        #                         column_selectable="single",
        #                         row_selectable="multi",
        #                         row_deletable=True,
        #                         selected_columns=[],
        #                         selected_rows=[],
        #                         page_action="native",
        #                         page_current=0,
        #                         page_size=10,
        #                         style_table={'overflowX': 'scroll',
        #                                      'overflowY': 'scroll'},
        #                         style_data_conditional=[
        #                             {
        #                                 "if": {"column_id": 'slotno'},
        #                                 "backgroundColor": "#3D9970",
        #                                 'color': 'white'
        #                             },
        #                         ],
        #                         style_header={
        #                             'backgroundColor': "#3D9970",
        #                             'fontWeight': 'bold'
        #                         }),
        #                 ]),
        #     ]),
        # ]),
        # html.Div([
        #     html.Div([
        #     ], id='subplot-div1'),
        #     dcc.RadioItems(id='selected-sample_no'),
        #     dcc.Graph(id='dataset-graphic'),
        #     html.Div([
        #         html.H3("Sample 1 Data"),
        #         html.Pre(id='sample1-data', style=styles['pre']),
        #     ]),
        #     html.Div([
        #         html.H3("Sample 1 Data"),
        #         html.Pre(id='sample2-data', style=styles['pre']),
        #     ]),
        #     html.Div([dcc.RangeSlider(id='feature-range-slider')],
        #              id='range-slider-div'),
        #     dcc.Dropdown(id='selected-feature'),
        # ], id='tabs-content'),

    ])


# @app.callback(Output('tabs-content', 'children'),
#               [Input('tabs', 'value')])
# def render_content(tab):
#     print('render_content:', tab)
#     if tab == 'tab-1':
#         return html.Div([
#             html.Div([
#                 html.Div([
#                     html.Div([
#                         html.P('Feature Name', className='control_label'),
#                     ], className='pretty_container four columns'),
#                     html.Div([
#                         html.Div([
#                             html.Div([
#                                 html.H6("6392"),
#                                 html.P("No. of Wells"),
#                             ], className='mini_container')
#                         ], className='row container-display')
#                     ], className='eight columns')
#                 ], className='row flex-display'),
#
#                 dcc.Dropdown(
#                     id='selected-feature',
#                     options=[{'label': i, 'value': i} for i in feature_names],
#                     value=f'{feature_names[0]}',
#                     className='twelwe columns',
#                     style={'width': '100%'}
#                 ),
#
#                 html.Hr(),
#                 html.Div([
#                     dcc.Markdown(d("""
#                             **Odd or Even Number Samples?**
#                             """)),
#                     dcc.RadioItems(
#                         id='selected-sample_no',
#                         options=[{'label': i, 'value': i} for i in ['1', '2']],
#                         value='1',
#                         labelStyle={'display': 'inline-block'}
#                     ),
#                 ]),
#
#                 #     dcc.RadioItems(
#                 #         id='selected-slotname',
#                 #         options=[{'label': i, 'value': i} for i in range(1, 33)],
#                 #         value='Linear',
#                 #         labelStyle={'display': 'inline-block'}
#                 #     ),
#             ], style={'width': '20%', 'display': 'inline-block'}),
#
#             html.Hr(),
#             html.Div([
#                 html.Div([
#                     html.Div([
#                         html.Div([  # graphic-div
#                             dcc.Graph(id='dataset-graphic'),
#                             html.Div([dcc.RangeSlider(id='feature-range-slider')],
#                                      id='range-slider-div'),
#                             html.Hr(),
#                             html.Div(children=[
#
#                                 html.Div([
#                                     dcc.Markdown(d("""
#                                                         **Sample 1 Data**
#                                                     """)),
#                                     html.Pre(id='sample1-data', style=styles['pre'])
#                                 ]),
#                                 html.Hr(),
#                                 html.Div([
#                                     dcc.Markdown(d("""
#                                                         **Sample 2 Data**
#                                                     """)),
#                                     html.Pre(id='sample2-data', style=styles['pre']),
#                                 ]),
#                             ])])],
#                         # style={'display': 'table-row'},
#                     ),
#
#                 ]),
#
#             ]),
#             html.Div(id='subplot-div1'),
#
#         ])
#
#     # elif tab == 'tab-2':
#     #     return html.Div([
#     #         html.Div([  # data-table-div
#     #             html.Div([
#     #                 # html.H3("Dataset"),
#     #                 # html.Hr(),
#     #                 # dash_table.DataTable(
#     #                 #     id='data-table',
#     #                 #     columns=[{"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns],
#     #                 #     data=df.to_dict('records'),
#     #                     # editable=True,
#     #                     # filter_action="native",
#     #                     # sort_action="native",
#     #                     # sort_mode="multi",
#     #                     # column_selectable="single",
#     #                     # row_selectable="multi",
#     #                     # row_deletable=True,
#     #                     # selected_columns=[],
#     #                     # selected_rows=[],
#     #                     # page_action="native",
#     #                     # page_current=0,
#     #                     # page_size=10,
#     #                     # style_table={'overflowX': 'scroll',
#     #                     #              'overflowY': 'scroll'},
#     #                     # style_data_conditional=[
#     #                     #     {
#     #                     #         "if": {"column_id": 'slotno'},
#     #                     #         "backgroundColor": "#3D9970",
#     #                     #         'color': 'white'
#     #                     #     },
#     #                     # ],
#     #                     # style_header={
#     #                     #     'backgroundColor': "#3D9970",
#     #                     #     'fontWeight': 'bold'
#     #                     # }
#     #                 # ),
#     #             ],
#     #             )],
#     #
#     #         )
#     #     ])
#
#
# @app.callback(Output('subplot-div1', 'children'),
#               [Input('selected-sample_no', 'value'),
#                Input('dataset-graphic', 'clickData')])
# def update_subplot(sample_no, clickdata):
#     print('update_subplot:', sample_no, clickdata)
#     # if len(hoverdata['points']) > 1: # two trace
#
#     sample_no = input_handler('sample_no', sample_no)
#
#     if clickdata is None:
#         spindle_speed = input_handler('spindle_speed', None)
#         depth_of_cut = input_handler('depth_of_cut', None)
#     else:
#         spindle_speed = clickdata['points'][0]['x']
#         depth_of_cut = clickdata['points'][0]['y']
#
#     dff = df.loc[(df['spindle_speed'] == spindle_speed) &
#                  (df['depth_of_cut'] == depth_of_cut) &
#                  (df['sample_no'] == sample_no)]
#     slotno = 1
#     if dff.shape[0] != 0:
#         # print('[update_subplot] Slotno(val0): ', dff['slotno'].values[0])
#         print('[update_subplot] Slotno(item): ', dff['slotno'].item())
#         slotno = dff['slotno'].item()
#
#     # image = Image.open('../images/stability_lobs.png')
#
#     html.Div([
#
#     ])
#
#     return html.Div([
#         html.Div([
#             html.Div([
#                 html.Img(src='data:image/png;base64,{}'.format(encoded_images['time_acc'][slotno].decode()),
#                          style={'width': '33%', 'display': 'inline-block'}),
#                 html.Img(src='data:image/png;base64,{}'.format(encoded_images['fft_acc'][slotno].decode()),
#                          style={'width': '33%', 'display': 'inline-block'}),
#                 html.Img(src='data:image/png;base64,{}'.format(encoded_images['spect_acc'][slotno].decode()),
#                          style={'width': '33%', 'display': 'inline-block'}),
#             ]),
#             html.Div([
#                 html.Img(src='data:image/png;base64,{}'.format(encoded_images['time_acoustic'][slotno].decode()),
#                          style={'width': '33%', 'display': 'inline-block'}),
#                 html.Img(src='data:image/png;base64,{}'.format(encoded_images['fft_acoustic'][slotno].decode()),
#                          style={'width': '33%', 'display': 'inline-block'}),
#
#                 html.Img(src='data:image/png;base64,{}'.format(encoded_images['spect_acoustic'][slotno].decode()),
#                          style={'width': '33%', 'display': 'inline-block'}),
#             ])
#
#         ]),
#
#     ])
#
#     # return html.Img(src='../images/stability_lobs.png')
#
#
# @app.callback(Output('sample1-data', 'children'),
#               [Input('selected-sample_no', 'value'),
#                Input('dataset-graphic', 'clickData'),
#                ])
# def set_sample1_data(sample_no, clickdata):
#     print('set_sample1_data:', sample_no, clickdata)
#     # if len(hoverdata['points']) > 1: # two trace
#     sample_no = input_handler('sample_no', sample_no)
#     if clickdata is None:
#         spindle_speed = input_handler('spindle_speed', None)
#         depth_of_cut = input_handler('depth_of_cut', None)
#     else:
#         spindle_speed = clickdata['points'][0]['x']
#         depth_of_cut = clickdata['points'][0]['y']
#
#     dff = df.loc[(df['spindle_speed'] == spindle_speed) &
#                  (df['depth_of_cut'] == depth_of_cut) &
#                  (df['sample_no'] == sample_no)]
#
#     # ps = list()
#     # for key, value in dff.to_dict().items():
#     #     ps.append(html.P(f"{key}--{list(value.values())[0]}"))
#     # return html.Div(ps)
#
#     return dash_table.DataTable(
#         columns=[{"name": i, "id": i} for i in dff.columns],
#         data=dff.to_dict('records'),
#     )
#
#
# @app.callback(Output('sample2-data', 'children'),
#               [Input('selected-sample_no', 'value'),
#                Input('dataset-graphic', 'clickData')])
# def set_sample2_data(sample_no, clickdata):
#     print('set_sample1_data:', sample_no, clickdata)
#     # if len(hoverdata['points']) > 1: # two trace
#     sample_no = input_handler('sample_no', sample_no)
#     if clickdata is None:
#         spindle_speed = input_handler('spindle_speed', None)
#         depth_of_cut = input_handler('depth_of_cut', None)
#     else:
#         spindle_speed = clickdata['points'][0]['x']
#         depth_of_cut = clickdata['points'][0]['y']
#
#     dff = df.loc[(df['spindle_speed'] == spindle_speed) &
#                  (df['depth_of_cut'] == depth_of_cut) &
#                  (df['sample_no'] != sample_no)]
#
#     # ps = list()
#     # for key, value in dff.to_dict().items():
#     #     ps.append(html.P(f"{key}--{list(value.values())[0]}"))
#     return dash_table.DataTable(
#         columns=[{"name": i, "id": i} for i in dff.columns],
#         data=dff.to_dict('records'),
#     )
#
#
# @app.callback(
#     Output(component_id='range-slider-div', component_property='children'),
#     [Input('selected-feature', 'value')]
# )
# def update_feature_range_slider(feature_name):
#     print('update_feature_range_slider:', feature_name)
#     if feature_name is None:
#         feature_name = 'slotno'
#     max = df[feature_name].max()
#     min = df[feature_name].min()
#     increment = (max - min) / 10
#     marks = [min + increment * i for i in range(10 + 1)]
#     return dcc.RangeSlider(
#         id='feature-range-slider',
#         min=min,
#         max=max,
#         value=[min, max],
#         marks={m: f'{m:.2f}' for m in marks},
#         allowCross=False,
#         step=None
#
#     )
#
#


# @app.callback(
#     Output('zvalue-name', 'value'),
#     [Input('zvalue-dropdown', 'value')])
# def get_zvalue_name(zvalue_dropdown_value):
#     return zvalue_dropdown_value


@app.callback(Output('hover-table', 'data'),
              [Input('zvalue-dropdown', 'value'),
               Input('sample-no', 'value'),
               Input('heatmap-graph', 'clickData')])
def update_hover(zvalue_name, sample_no, click_data):
    print(f"[update_hover] z: {zvalue_name} \t sample_no: {sample_no} \t click_data: {click_data}")

    spindle_speed = click_data['points'][0]['x']
    depth_of_cut = click_data['points'][0]['y']
    dff = df.loc[
        (df['sample_no'] == sample_no) & (df['spindle_speed'] == spindle_speed) & (df['depth_of_cut'] == depth_of_cut)]
    return [{'feature': k,
             'hover1': f"{dff[k].item():.2f}" if k not in ['kesim_paramsexcel', 'slotname'] else dff[k].item(),
             'hover2': f"{dff[k].item():.2f}" if k not in ['kesim_paramsexcel', 'slotname'] else dff[k].item()}
            for k in dff.columns]


@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('zvalue-dropdown', 'value'),
     Input('sample-no', 'value')
     ])
def update_heatmap_graph(zvalue_name, sample_no):
    print('update_heatmap_graph:', zvalue_name, sample_no)
    # feature_name = input_handler('feature_name', feature_name)
    # min = df[feature_name].min()
    # max = df[feature_name].max()
    # if feature_minmax is not None:
    #     min, max = feature_minmax
    # dff = df.loc[(df[feature_name] > min) &
    #              (df[feature_name] < max) &
    #              (df['sample_no'] == sample_no)
    #              ]
    dff = df.loc[df['sample_no'] == sample_no]

    return {
        'data': [
            go.Heatmap(name='heatmap',
                       x=dff['spindle_speed'],
                       y=dff['depth_of_cut'],
                       z=np.log(dff[zvalue_name].values),
                       zsmooth='best',
                       hoverinfo='none',
                       colorscale='Viridis'),

            go.Scatter(name='base-trace',
                       x=dff['spindle_speed'],
                       y=dff['depth_of_cut'],
                       text=dff['depth_of_cut'],
                       mode='markers',
                       marker=dict(
                           symbol='circle',
                           opacity=0.7,
                           color='white',
                           size=8,
                           line=dict(width=1),
                       )
                       ),

        ],
        'layout': go.Layout(
            xaxis={
                'title': 'Spindle Speed',
                'showgrid': False
            },
            yaxis={
                'title': 'Depth of Cut',
                'showgrid': False
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


# @app.callback(
#     Output("debug-output", "children"),
#     [Input("selected-feature", "value"),
#      Input("selected-sample_no", "value")
#      ],
# )
# def update_output(feature_name, sample_no):
#     print('update_output:', feature_name, sample_no)
#     return html.Div([
#         html.P(f"Feature: {feature_name}"),
#         html.P(f"Sample No: {sample_no}"),
#     ])


if __name__ == '__main__':
    app.run_server(debug=True, host='10.5.150.165')
