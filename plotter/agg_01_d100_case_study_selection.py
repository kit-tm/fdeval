import logging, math, json, pickle, os
import matplotlib.pyplot as plt
import numpy
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import time
import importlib  
from matplotlib.gridspec import GridSpec
from topo.static import LAYOUTS

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
matplotlib.rcParams.update(params)

from . import agg_2_utils as utils

logger = logging.getLogger(__name__)

SKIP_SEEDS = [321241]


def plot(blob, **kwargs):
    """
    Case Study Selection
    """

    utils.EXPORT_BLOB = blob
   
    includes = [
        'scenario_switch_cnt',
        'scenario_table_util_avg_total',
        'scenario_table_util_max_total',
    ]
    includes += blob.find_columns('table_datay_raw')

    blob.include_parameters(**dict.fromkeys(includes, 1))
    runs = blob.filter(**dict())
    allseeds = []
    ratios = []
    seed_ratio = {}
    ratio_by_seed = {}
    for run in runs:
        thresh = run.get('param_topo_switch_capacity_overwrite')
        algo = run.get('param_dts_algo')
        if thresh != 95: continue
        if algo != 2: continue

        seed = run.get('param_topo_seed')
        if not seed in allseeds:
            allseeds.append(seed)
            util_values = []
            util_max = []
            for switch in range(0, run.get('scenario_switch_cnt')):
                d2 = 'dts_%d_table_datay_raw' % (switch)
                if run.get(d2):
                    data = [x for x in run.get(d2) if x > 0]
                    if len(data) > 0:
                        util_values.append(sum(data)/float(len(data)))
                        util_max.append(max(data))

            if len(util_values) > 0:
                util_avg = sum(util_values)/float(len(util_values))
                util_max = max(util_max)
                ratio = (float(util_avg) / float(util_max))*100.0
                ratios.append(ratio)
                seed_ratio[ratio] = seed
                ratio_by_seed[seed] = ratio

    ratios = sorted(ratios)
    seed_ratio_sorted = sorted(seed_ratio.keys())

    examples = []

    fig, ax, = plt.subplots(figsize=(10, 6))
    fig.tight_layout(pad=2.7) 
    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_ylabel('CDF', fontsize=15)
    ax.set_xlabel(r'Flow table utilization ratio (\%)', fontsize=15)
    utils.plotcdf(ax, ratios, label='ratios', linewidth=2, color='red')

    x = seed_ratio_sorted[1]
    y = 2 / float(len(ratios))
    ax.vlines(x, y, y-0.2, color='black', linestyle=':', linewidth=2, alpha=1)
    ax.text(x, y-0.2, r'\noindent Scenario $z_{136505}$ \\with  ratio ' + ('%.2f' % x) + '\\%%',
        fontsize=18, fontweight='normal', color="black", va='center', ha="left", 
        bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    x = seed_ratio_sorted[-1]
    y = 1
    ax.vlines(x, y, y-0.2, color='black', linestyle=':', linewidth=2, alpha=1)
    ax.text(x, y-0.2, r'\noindent Scenario $z_{166890}$ \\with  ratio ' + ('%.2f' % x) + '\\%%',
        fontsize=18, fontweight='normal', color="black", va='center', ha="right", 
        bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    x = ratio_by_seed.get(155603)
    y = 34/100.0
    ax.vlines(x, y, y-0.2, color='black', linestyle=':', linewidth=2, alpha=1)
    ax.text(x, y-0.2, r'\noindent Scenario $z_{155603}$ \\with  ratio ' + ('%.2f' % x) + '\\%%',
        fontsize=18, fontweight='normal', color="black", va='center', ha="left", 
        bbox=dict(boxstyle="square", ec='white', fc='white',)
    )


    x = ratio_by_seed.get(84812)
    y = 63/100.0
    ax.vlines(x, y, y-0.2, color='black', linestyle=':', linewidth=2, alpha=1)
    ax.text(x, y-0.2, r'\noindent Scenario $z_{84812}$ \\with  ratio ' + ('%.2f' % x) + '\\%%',
        fontsize=18, fontweight='normal', color="black", va='center', ha="left", 
        bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    ax.set_ylim(-0.3, 1.1)
    ax.set_yticks([0,0.25,0.5,0.75,1])

    utils.export(fig, 'scenario_selection.pdf', folder='case-study-selection')
    plt.close()
