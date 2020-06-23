import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import math
import operator


def plot(ctx):
    "Link utilization for observed links"

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.tight_layout(pad=2.7) 

    data = ctx.statistics.get('metrics.link_utils')

    for link_data in data:
        demand_over_time = [x/1000.0 for x in link_data.get('demand_over_time')]
        label = link_data.get('label')
        print(label, len(demand_over_time))

        #fig, ax = plt.subplots(figsize=(16, 8))
        linestyle = '-'
        if 'ES' in label:
            linestyle = ':'
        ax.plot(np.arange(len(demand_over_time)), demand_over_time, label=label, linewidth=1, linestyle=linestyle)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('link utilization (Mbit/s)')
    ax.set_xlim(0,450)
    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.xaxis.grid(True, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.legend()
    plt.show()
