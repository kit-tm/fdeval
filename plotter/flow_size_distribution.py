import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import math

def plot(ctx):
    "Plot overutilization for a single run"

    demands = []
    flowtimes = []
    diffs = []
    for flow in ctx.flows:
        flowtimes.append(flow.finished_at - flow.start)
        time_actual = flow.finished_at - flow.start
        demands.append(flow.total_demand)
        expected = flow.total_demand/float(flow.demand_per_tick) + flow.path_delay_summed
        diff = expected-time_actual
        if diff < 0.00000001:
            diffs.append(0)
        else:
            diffs.append(abs(expected-time_actual))

    flowtimes = list(sorted(flowtimes))
    demands = list(sorted(demands))

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(np.arange(len(flowtimes)), flowtimes, color='green', label='Sorted Flow Duration')     

    ax2 = ax.twinx() 
    ax2.plot(np.arange(len(demands)), demands, color='red', label='Sorted Demands')

    ax.legend(bbox_to_anchor=(0.01, 1), loc=2, borderaxespad=0.)
    ax2.legend(bbox_to_anchor=(0.01, 0.95), loc=2, borderaxespad=0.)
    plt.show()

