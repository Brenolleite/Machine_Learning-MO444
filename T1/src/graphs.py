import matplotlib.pyplot as plt

def plot(title, xlabel, ylabel, x, y):
    fig, ax = plt.subplots(nrows = 1, ncols=1 )
    ax.plot([0,1,2], [10,20,3])
    fig.savefig('path/to/save/image/to.png')
    plt.close(fig)