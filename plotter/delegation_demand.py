import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import math
import operator

def putval(x1, x2, demand_per_tick, result):
    """
    Helper function to calculate the demand over time (stored in result)
    based on (x1=start, x2=end, x3=demand_per_second)
    """
    for i in range(math.floor(x1)-1, math.floor(x2)+2):
        demand = 0
        if i-x1 >= 1 and x2-i >= 0:
            demand = demand_per_tick   
        if i-x1 < 1 and i-x1 > 0:
            demand = (i-x1)*demand_per_tick
        if i-x2 < 1 and i-x2 > 0:
            demand = (1-(i-x2))*demand_per_tick    
        if not result.get(i): result[i] = 0
        result[i] += demand

def plot(ctx):
    "Volume that was delegated"

    result = dict()
    result2 = dict()

    # extract total demand over time
    events2 = []
    for flow in ctx.flows:
        events2.append((flow.start, flow.demand_per_tick))
        events2.append((flow.start+flow.duration, -flow.demand_per_tick))
        putval(flow.start, flow.start+flow.duration, flow.demand_per_tick, result2)

    # extract delegated demand over time
    events = []
    demand_per_port = {}
    demand_delegated = 0
    demand_total = 0
    per_tick = {}
    for flow in ctx.flows:
        demand_total += flow.duration * flow.demand_per_tick
        if len(flow.delegation.reports) > 1:
            for r1, r2 in zip(flow.delegation.reports, flow.delegation.reports[1:]):
                # start and end time of delegation are recorded
                if r1.action == 1 and r2.action == 0:
                    demand =  (r2.tick-r1.tick)*flow.demand_per_tick
                    demand_delegated += demand
                    putval(r1.tick, r2.tick, flow.demand_per_tick, result)
                    assert(demand >= 0)
                    events.append((r1.tick, demand))
                    events.append((r2.tick, -demand))
            rlast =  flow.delegation.reports[-1]
            if rlast.action == 1:
                demand =  (flow.finished_at-rlast.tick)*flow.demand_per_tick
                demand_delegated += demand
                assert(demand >= 0)
                putval(rlast.tick, flow.finished_at, flow.demand_per_tick, result)
                events.append((rlast.tick, demand))
                events.append((flow.finished_at, -demand))

        if len(flow.delegation.reports) == 1:
            r1 = flow.delegation.reports[0]
            demand =  (flow.finished_at-r1.tick)*flow.demand_per_tick
            demand_delegated += demand
            assert(demand >= 0)
            putval(r1.tick, flow.finished_at, flow.demand_per_tick, result)
            events.append((r1.tick, demand))
            events.append((flow.finished_at, -demand))


    fig, ax = plt.subplots(figsize=(8, 3))
    fig.tight_layout(pad=2.7) 

    xvalues = []
    yvalues = []
    for t, v in sorted(result.items()):
        xvalues.append(int(t))
        yvalues.append(v/1000.0)

    #fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(xvalues, yvalues, color='black', linewidth=1) 
    ax.set_xlabel('time (s)')
    ax.set_ylabel('delegated (Mbit/s)')
    #fill_underutil = [True if x < threshold and x+e > threshold else False for x, e in zip(cnt_active_flows, cnt_active_flows_evicted)]
    ax.fill_between(xvalues, yvalues, 0,  color='orange', alpha=0.3)
    #ax.set_title('%s (%s)' % (names[solver], metrics[objective]), fontsize=10, fontweight='bold')
    ax.set_xlim(0,450)
    #ax.set_ylim(0,350)
    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.xaxis.grid(True, color='red', linestyle='--', linewidth=1, alpha=0.5)


    plt.show()
