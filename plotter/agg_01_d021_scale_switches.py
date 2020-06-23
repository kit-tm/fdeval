import logging, math, json, pickle, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import statistics

logger = logging.getLogger(__name__)

from . import agg_2_utils as utils



def plot(blob, **kwargs):
    "Plot dts scores (overutil, underutil, overheads)"

    utils.EXPORT_BLOB = blob
  
    includes = ['scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch', 'scenario_table_util_avg_total', 'scenario_table_util_max_total',
        'rsa_solver_jobs', 'rsa_solver_stats_itercount', 'rsa_solver_stats_nodecount']

    includes += blob.find_columns('hit_timelimit')
    includes += blob.find_columns('solver_stats_time_modeling')
    includes += blob.find_columns('solver_stats_time_solving')
    includes += blob.find_columns('solver_cnt_feasable')
    includes += blob.find_columns('solver_cnt_infeasable')
    includes += blob.find_columns('solver_considered_ports')
    includes += blob.find_columns('overutil_max')

    blob.include_parameters(**dict.fromkeys(includes, 1))


    for perc in [50, 90, 99, 100]:
        fig, axes = plt.subplots(1,2,figsize=(12, 6))
        fig.tight_layout(pad=3)
        axes[0].set_xlabel(r'Number of switches', fontsize=15)
        axes[0].set_ylabel('RS-Alloc modeling time (ms)', fontsize=15)
        axes[1].set_xlabel(r'Number of switches', fontsize=15)
        axes[1].set_ylabel('RS-Alloc solving time (ms)', fontsize=15)
        timelimit = 0
        allvals = []
        for param_topo_switch_capacity, color, marker in zip([30,50,70,90], ['green', 'blue', 'red', 'm'], ['s', 'o','+','*']):
            datax = []
            datay_solving = []
            datay_modeling = []
            

            for param_topo_num_switches in range(10,300,10):
                runs = blob.filter( 
                    param_topo_switch_capacity=param_topo_switch_capacity,
                    param_topo_num_switches=param_topo_num_switches)
                modeling = []
                solving = []       
                for run in runs:
                    if run.get('hit_timelimit'):
                        timelimit += 1
                    s = [x * 1000 for x in run.get('rsa_solver_stats_time_solving') if x > 0]
                    m = [x * 1000 for x in run.get('rsa_solver_stats_time_modeling') if x > 0]
                    modeling.append(statistics.mean(m))
                    solving.append(statistics.mean(s))
                datax.append(param_topo_num_switches)
                datay_solving.append(np.percentile(solving, perc))
                datay_modeling.append(np.percentile(modeling, perc))


            axes[1].text(0.97, 0.97, 
               ('Solving (%d' % (perc)) + r'th percentiles)',
                color='black', fontsize=18,
                transform=axes[1].transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='white'))  

            axes[0].text(0.03, 0.97, 
               ('Modeling (%d' % (perc)) + r'th percentiles)',
                color='black', fontsize=18,
                transform=axes[0].transAxes, verticalalignment='top',
                horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='white'))  


            axes[0].plot(datax, datay_modeling, label=('%d' % (100-param_topo_switch_capacity)) + r'\% reduction', color=color, marker=marker)
            axes[1].plot(datax, datay_solving, label=('%d' % (100-param_topo_switch_capacity)) + r'\% reduction',  color=color, marker=marker)

            allvals += datay_modeling + datay_solving

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=16)
        fig.subplots_adjust(top=0.9) # no gap

        print("timelimit", timelimit)

        for ax in fig.axes:
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylim(0, int(max(allvals)*1.1))

        utils.export(fig, 'scale_switches_%d.pdf' % perc, folder='runtime')
        plt.close()
