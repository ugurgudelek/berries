
class Visualizer:
    """

    """

    def __init__(self):
        pass


def display_errors(history):

    t_error = []
    for epoch, epoch_history in history.items():
        t_error.append(epoch_history['train']['loss'].mean())

    print(t_error[-1])
    plt.plot(np.log(t_error))

    fig.canvas.draw()
    plt.pause(0.0001)