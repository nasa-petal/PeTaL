import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px

from dash.dependencies import Input, Output

import webbrowser

class Dashboard:
    '''
    A live graph dashboard for use in ML modules!
    '''
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div(
            html.Div([
                html.H4('Custom live feed'),
                html.Div(id='live-update-text'),
                dcc.Graph(id='live-update-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=1*1000, # in milliseconds
                    n_intervals=0
                )
            ])
        )

        self.app.callback(Output('live-update-graph', 'figure'), [Input('interval-component', 'n_intervals')])(self.update_graph_live)
        self.app.callback(Output('live-update-text', 'children'), [Input('interval-component', 'n_intervals')])(self.update_metrics)

    def update_metrics(self, n):
        style = {'padding': '5px', 'fontSize': '16px'}
        return [
            html.Span('Text updates here', style=style),
        ]

    def update_graph_live(self, n):
        '''
        Default plot: An increasing line...
        Make this anything, i.e. loss over time?
        '''
        line = list(range(n))
        return px.line(x=line, y=line)

    def run(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':
    webbrowser.open_new_tab('http://127.0.0.1:8050')
    dashboard = Dashboard()
    dashboard.run()
