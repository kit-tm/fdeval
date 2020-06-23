import logging, math, json, pickle, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import statistics
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logger = logging.getLogger(__name__)

from . import agg_2_utils as utils

FIGSIZE = (14, 9)
XOFFSET = 60
USE_SEEDS = [1,2,3]

def plot(blob, **kwargs):
    "Example topologies used"

    utils.EXPORT_BLOB = blob
    
    includes = [ 'scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch']
    includes += blob.find_columns('hit_timelimit')
    includes += blob.find_columns('scenario_table_datax')
    includes += blob.find_columns('scenario_table_datay')

    blob.include_parameters(**dict.fromkeys(includes, 1))

    for seed in USE_SEEDS:

        # create filter object (helper, used below with **f())
        def params(param_topo_num_switches, param_topo_num_hosts, param_topo_scenario_ba_modelparam):
            return dict(param_topo_num_hosts=param_topo_num_hosts,
                param_topo_num_switches=param_topo_num_switches,
                param_topo_scenario_ba_modelparam=param_topo_scenario_ba_modelparam,
                param_topo_seed=seed,
                param_topo_idle_timeout=3         
            )

        # -----------------------
        # Figure: Different topologies
        # -----------------------
        if True:
            fig = plt.figure(figsize=FIGSIZE)
            maingrid = gridspec.GridSpec(3, 3, figure=fig, 
                wspace=0.2, hspace=0.25, left=0.05, bottom=0.05, right=0.95, top=0.9)  
            overallcnt = 0
            rowcnt = 0
            axes = []
            for switches in [4,8,12]:
                axcnt = 0
                maxy = 0
                for parameter_m, hosts  in [(1,100), (2,150), (3,200)]:
                    overallcnt += 1
                    runs = blob.filter(**params(switches, hosts, parameter_m))
                    assert(len(runs) == 1)
                    run = runs[0]
                    switch = 0
                    try:
                        ax = fig.add_subplot(maingrid[rowcnt,axcnt], sharey=axes[0])
                    except IndexError:
                        ax = fig.add_subplot(maingrid[rowcnt,axcnt])
                    axes.append(ax)
                    axcnt += 1
                    
                    hosts_of_switch = {}
                    edges = run.get('scenario_edges')
                    for k, v in run.get('scenario_hosts_of_switch').items():
                        hosts_of_switch[int(k)] = v
                    plt_switches = list(range(0, run.get('scenario_switch_cnt')))
                    utils.plot_topo_small(ax, hosts_of_switch, edges, plt_switches , [],
                        switch_node_size=100, font_size=12)
                    

                    # show bottleneck parameters
                    w1 = str(run.get('param_topo_num_switches'))
                    w2 = str(run.get('param_topo_num_hosts'))
                    w3 = str(run.get('param_topo_scenario_ba_modelparam'))
                    circled_number = str(chr(96+overallcnt))
                    ax.text(0.08, 1.13, circled_number, fontsize=14, 
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, color='black', alpha=1,
                        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black')
                    )
                    ax.text(0.18, 1.17, r'\noindent Switches: ' + w1 +
                        r'\\Hosts: ' + w2 + 
                        r'\\Parameter m: ' + w3, fontsize=12, 
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, color='black', alpha=1,
                        #bbox=dict(facecolor='white', edgecolor='black')
                    )
                rowcnt += 1


            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
            utils.export(fig, 'example_topo_topologies_seed_%d.pdf' % seed, folder='topos')