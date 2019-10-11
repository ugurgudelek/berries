import plotly

from plotly import tools
import plotly.graph_objs as go
import os
from collections import defaultdict

OFFLINE = True

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if isnotebook():
    if OFFLINE:
        from plotly.offline import iplot as plotlyplot
        plotly.offline.init_notebook_mode(connected=True)
    else:
        from plotly.plotly import iplot as plotlyplot


else:
    if OFFLINE:
        from plotly.offline import plot as plotlyplot
    else:
        from plotly.plotly import plot as plotlyplot


class XY():
    def __init__(self, x, y, name, opacity=0.5, linewidth=2, dash=None):
        self.x = x
        self.y = y
        self.name = name
        self.opacity = opacity
        self.dash = dash  # dash options include 'dash', 'dot', and 'dashdot'
        self.linewidth = linewidth


    def __len__(self):
        return len(self.x)


class BaseCurve():
    def __init__(self):
        self.container = defaultdict(XY)

    def add_vector(self, name, y, x=None, opacity=0.5, linewidth=2, dash=None):
        if x is None:
            x = list(range(len(y)))
        self.container[name] = XY(x, y, name=name, opacity=opacity, linewidth=linewidth, dash=dash)

    def to_traces(self):
        return [go.Scatter(x=data.x,
                           y=data.y,
                           mode='lines+markers',
                           opacity=0.5,
                           name=name,
                           line=dict(width=data.linewidth,
                                     dash=data.dash)
                           )
                for name, data in self.container.items()]

    def __len__(self):
        return self.container.__len__()


class LearningCurve(BaseCurve):
    def __init__(self):
        BaseCurve.__init__(self)


class PredictionCurve(BaseCurve):
    def __init__(self):
        BaseCurve.__init__(self)

class LSTMInnerCurve(BaseCurve):
    def __init__(self):
        BaseCurve.__init__(self)


class Plotter():

    def __init__(self, title, path):
        self.title = title
        self.path = f"{os.path.join(path, self.title)}.html"
        fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=False, shared_yaxes=False,
                                  subplot_titles=('Prediction Curve', 'LSTM Inner States', 'Learning Curve'))
        fig['layout'].update(title=title,
                             yaxis1=dict(range=[-1., 2.]),
                             yaxis2=dict(range=None))
        self.figure = go.FigureWidget(fig)
        self.plot(filename=self.path, auto_open=True)

        # Learning Curve
        self.learning_curve = LearningCurve()

        # Prediction Curve
        self.predicton_curve = PredictionCurve()

        # LstmInner Curve
        self.lstminner_curve = LSTMInnerCurve()

    def reset_traces(self):
        self.figure.data = []  # reset traces

    def update(self):
        self.reset_traces()

        self.figure.add_traces(self.predicton_curve.to_traces(),
                               rows=[1]*len(self.predicton_curve),
                               cols=[1]*len(self.predicton_curve))
        self.figure.add_traces(self.lstminner_curve.to_traces(),
                               rows=[2] * len(self.lstminner_curve),
                               cols=[1] * len(self.lstminner_curve))
        self.figure.add_traces(self.learning_curve.to_traces(),
                               rows=[3] * len(self.learning_curve),
                               cols=[1] * len(self.learning_curve))


        self.plot(filename=self.path, auto_open=False)

    def plot(self, **kwargs):
        plotlyplot(self.figure, **kwargs)
