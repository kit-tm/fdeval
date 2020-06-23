import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches

from matplotlib.backends.backend_pdf import PdfPages
import math
import operator


def plot(ctx):
    "Delegation status over time"

    switches = [opts['_switch'] for _, opts in ctx.topo.graph.nodes(data=True) if opts.get('_switch')]
    fig, ax = plt.subplots(figsize=(16, 6))
    tvalues = []
    for switch in switches:
        if switch.label == 'DS':
            for t, report in enumerate(switch.reports):
                #print(t, report.cnt_port_delegation_status)
                for p, status in enumerate(report.cnt_port_delegation_status):
                    if status == 1:
                        tvalues.append(t)
                        p = patches.Rectangle((t-1,p-0.5), 1,1, color='orange', alpha=0.3)
                        ax.add_patch(p)
                    #else:
                    #    p = patches.Rectangle((t-1,p-0.5), 1,1, color='grey', alpha=0.1)
                    #    ax.add_patch(p)

    for p in range(1,21):               
        ax.hlines(p-0.5, 0, len(tvalues), color='grey', linestyle='-', linewidth=1, alpha=0.3)

    x1 = (int(math.ceil(min(tvalues) / 25.0)) * 25) - 25
    x2 =  int(math.ceil(max(tvalues) / 25.0)) * 25              
    ax.set_xlim(x1,x2)
    ax.set_ylim(-0.5,20.5)
    ax.set_yticks(np.arange(21))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('port number') 


    plt.gca().xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    plt.show()