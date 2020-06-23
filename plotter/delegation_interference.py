
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import math
import operator


def plot(ctx):
    "Volume that was delegated"
    data = ctx.statistics.get('metrics.ds.flowtable_cnt_backdelegations')
    _sum = 0
    data_y = []
    for i, v in enumerate(data):
        _sum += v
        data_y.append(_sum)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(len(data_y)), data_y, color='black', linewidth=1, label='Sorted Demands')
    ax.fill_between(np.arange(len(data_y)), data_y, 0,  color='orange', alpha=0.3)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('#rules affected') 

    ax.legend(bbox_to_anchor=(0.01, 1), loc=2, borderaxespad=0.)
    plt.gca().yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    plt.show()
