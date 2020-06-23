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
from matplotlib.gridspec import GridSpec
from topo.static import LAYOUTS
logger = logging.getLogger(__name__)

from . import agg_2_utils as utils

FIGSIZE = (14, 5)
XOFFSET = 60
USE_SEEDS = [1,2,3]
BOTTLENECKS = 5

def plot(blob, **kwargs):
    "Example for a single scenario"

    utils.EXPORT_BLOB = blob
    
    includes = [ 'scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch']
    includes += blob.find_columns('hit_timelimit')
    includes += blob.find_columns('scenario_table_datax')
    includes += blob.find_columns('scenario_table_datay')

    blob.include_parameters(**dict.fromkeys(includes, 1))

    for BOTTLENECKS in [0,2,5]:
        for seed in USE_SEEDS:

            # create filter object (helper, used below with **f())
            def params(seed, param_topo_concentrate_demand, param_topo_traffic_interswitch):
                return dict(param_topo_num_hosts=100,
                    param_topo_num_flows=100000,
                    param_topo_seed=seed,
                    param_topo_bottleneck_cnt=BOTTLENECKS,
                    param_topo_bottleneck_duration=50,
                    param_topo_bottleneck_intensity=150,
                    param_topo_concentrate_demand=param_topo_concentrate_demand,
                    param_topo_scenario_ba_modelparam=1,
                    param_topo_traffic_interswitch=param_topo_traffic_interswitch,
                    param_topo_idle_timeout=3         
                )

            # -----------------------
            # Figure: Different bottleneck parameters
            # -----------------------
            for param_topo_concentrate_demand in [0,1,2]:    
                for param_topo_traffic_interswitch in [20, 50, 75]:

                    fig = plt.figure(figsize=FIGSIZE)
                    runs = blob.filter(**params(seed, param_topo_concentrate_demand, param_topo_traffic_interswitch))
                    assert(len(runs) == 1)
                    run = runs[0]
                    height_ratios=[3,4]
                    maingrid = gridspec.GridSpec(2, run.get('param_topo_num_switches'), figure=fig, 
                        wspace=0.2, hspace=0.3, left=0.05, bottom=0.05, right=0.95, top=0.85,height_ratios=height_ratios)  
                    
                    rowcnt = 0
                    axes = []
                    axcnt = 0
                    maxy = 0


                    # +1 for topology plot in the top left
                    x=9999 # used with LAYOUTS; topology is placed here
                    all_axes = []
                    layout = LAYOUTS.get(run.get('param_topo_num_switches'))
                    cols = len(layout[0])
                    rows = len(layout)
                    fig = plt.figure(constrained_layout=True, figsize=(14, 5))
                    gs = GridSpec(rows, cols, figure=fig)

                    # first the topology 
                    coords = None
                    for y in range(rows):
                        for x in range(cols):
                            if layout[y][x] == 9999:
                                if coords:
                                    break;
                                coords = [y,x]
                                colspan = sum([1 if v == 9999 else 0 for v in layout[y]])
                                rowspan = sum([1 if 9999 in v else 0 for v in layout])  
                                break;
                    all_axes.append(plt.subplot(gs.new_subplotspec((coords[0], coords[1]), rowspan=rowspan, colspan=colspan)))

                    # and then all the other axes
                    oldval = 0
                    for y in range(rows):
                        for x in range(cols):
                            val = layout[y][x]
                            if val == 9999:
                                continue;
                            if val > oldval:
                                colspan = sum([1 if v == val else 0 for v in layout[y]])
                                rowspan = sum([1 if val in v else 0 for v in layout])
                                all_axes.append(plt.subplot(gs.new_subplotspec((y, x), rowspan=rowspan, colspan=colspan)))
                                oldval = val

                    # plot topology in the top left
                    #self._plot_topo(all_axes[0])


                    plotted_topo = False

                    for switch in range(0, run.get('param_topo_num_switches')):
                        #try:
                        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt], sharey=axes[0])
                        #except IndexError:
                        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt])

                        ax = all_axes[switch+1]
                        axes.append(ax)
                        
                        ax.set_xlim(-1*XOFFSET, 400)
                        

                        datax = run.get('scenario_table_datax_%d' % switch)
                        datay = run.get('scenario_table_datay_%d' % switch)

                        if max(datay) > maxy:
                            maxy = max(datay)

                        ax.plot(list(range(-1*XOFFSET,0)) + datax, [0]*XOFFSET + datay, color='black', linestyle='-', linewidth=0.75)

                        ax.fill_between(datax, [0]*len(datay), datay,
                            interpolate=True, color='orange', alpha=0.3, label='Rules in flow table')

                        # show bottleneck parameters
                        w1 = str(run.get('param_topo_bottleneck_cnt'))
                        w2 = str(run.get('param_topo_bottleneck_duration')) + "s"
                        w3 = str(run.get('param_topo_bottleneck_intensity'))
                        if run.get('param_topo_bottleneck_cnt') == 0:
                            w2 = '-'
                            w3 = '-'
                        circled_number = str(switch)
                        circled_color = 'black'
                        scenario_concentrated_switches = run.get('scenario_concentrated_switches')
                        if switch in scenario_concentrated_switches:   
                            circled_color = 'red'
                        ax.text(0.5, .95, circled_number, fontsize=14, 
                            verticalalignment='center', horizontalalignment='center',
                            transform=ax.transAxes, color='white', alpha=1,
                            bbox=dict(boxstyle='circle', facecolor=circled_color, edgecolor='black')
                        )

                        # plot bottlenecks
                        ax.hlines(-1*XOFFSET, -1*XOFFSET, 400, color='gray', linestyle='-', alpha=0.3, linewidth=7)
                        bottleneck_data = run.get('scenario_bottlenecks')
                        set_label = 0
                        for start, end, details in bottleneck_data:
                            if set_label == 0:
                                ax.hlines(-1*XOFFSET, start, end, color='red', linestyle='-', alpha=0.3, linewidth=7,
                                    label='Temporal bottleneck')
                                set_label = 1
                            else:
                                ax.hlines(-1*XOFFSET, start, end, color='red', linestyle='-', alpha=0.3, linewidth=7)

                        # ----------------- second row: topology
                        if not plotted_topo:
                            plotted_topo = True

                            #ax = fig.add_subplot(maingrid[rowcnt+1,axcnt])
                            hosts_of_switch = {}
                            edges = run.get('scenario_edges')
                            for k, v in run.get('scenario_hosts_of_switch').items():
                                hosts_of_switch[int(k)] = v
                            plt_switches = list(range(0, run.get('scenario_switch_cnt')))
                            utils.plot_topo_small(all_axes[0], hosts_of_switch, edges, plt_switches , scenario_concentrated_switches,
                                switch_node_size=250, font_size=15)
                            

                        axcnt += 1



                    rowcnt += 1

                    for ax in axes:
                        ax.set_ylim(-120, 2500)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    for ax in axes[-4:]:
                        ax.set_xlabel('Time (s)')
                    for ax in []:
                        ax.set_ylabel('Flow table utilization')


                    handles, labels = axes[-1].get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper left', ncol=1, fontsize=16)
                    utils.export(fig, 'example_topo_scenario_seed_%d_inter_%d_hotspot_%d_bottlenecks_%d.pdf' % (seed,
                        param_topo_traffic_interswitch, param_topo_concentrate_demand, BOTTLENECKS), folder='example_topo')

                    #exit(1)    }
