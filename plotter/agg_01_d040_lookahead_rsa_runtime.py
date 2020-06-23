import logging, math, json, pickle, os
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

SHOW_OUTLIERS = False
MAX_ASSIGNMENTS = {
    1: dict(color="red"),
    5: dict(color="green"),
    20: dict(color="blue"),
    200: dict(color='m')
}

def plot(blob, **kwargs):
    "Plot look-ahead RSA"

    utils.EXPORT_BLOB = blob
  

    LA =  [1,2,3,4,5,6,7,8,9]

    includes = ['scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch', 
        'rsa_solver_cnt_infeasable',
        'rsa_solver_stats_time_modeling', 
        'rsa_solver_stats_time_solving', 
        'rsa_solver_stats_gap', 
        'rsa_table_fairness_avg',
        'rsa_link_util_delegated_mbit_max', 
        'rsa_ctrl_overhead_from_rsa',
        'rsa_table_percent_relocated']

    includes += blob.find_columns('hit_timelimit')

    blob.include_parameters(**dict.fromkeys(includes, 1))


    runs = blob.filter(**dict())

    # -----------------------
    # prepare data for plotting
    # -----------------------
    DATA = {}
    seeds = []
    ignore_seeds = []
    for run in runs:
        seed = run.get('param_topo_seed')
        param_rsa_look_ahead = run.get('param_rsa_look_ahead')
        param_rsa_max_assignments = run.get('param_rsa_max_assignments')
        infeasible = run.get('rsa_solver_cnt_infeasable')
        if infeasible > 0:
            if not seed in ignore_seed:
                ignore_seeds.append(seed)
        if not seed in seeds:
            seeds.append(seed)
        if not DATA.get(param_rsa_max_assignments):
            DATA[param_rsa_max_assignments] = {}
        if not DATA[param_rsa_max_assignments].get(param_rsa_look_ahead):
            DATA[param_rsa_max_assignments][param_rsa_look_ahead] = {}
        DATA[param_rsa_max_assignments][param_rsa_look_ahead][seed] = run

    print("ignore", len(ignore_seeds))

    # -----------------------
    # Figure: modeling and solving time based on look-ahead
    # -----------------------
    if 1:
        plt.close()
        fig, axes = plt.subplots(2, len(MAX_ASSIGNMENTS), figsize=(14, 6),  sharex=True)
        fig.tight_layout(pad=2.7)

        for ax, label in zip([axes[x][0] for x in range(0,2)], [r'Modeling time (ms)', r'Solving time (ms)']):
            ax.set_ylabel('%s' % label, fontsize=15)

        # force shared y axis
        for y in range(0,2):
            useax = [axes[y][x] for x in range(0,len(MAX_ASSIGNMENTS))]
            for ax in useax:
                ax.get_shared_y_axes().join(*useax)


        for ax in [axes[1][x] for x in range(0,len(MAX_ASSIGNMENTS))]:
            ax.set_xlabel('Look-ahead factor L', fontsize=15)
        for ax in fig.axes:
            #ax.set_yscale('log')
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)     


        colcnt = 0
        for param_rsa_max_assignments, DATA1 in sorted(DATA.items()):

            if param_rsa_max_assignments not in MAX_ASSIGNMENTS.keys():
                continue
            color = MAX_ASSIGNMENTS.get(param_rsa_max_assignments).get('color')
            label = '%d assignments' % param_rsa_max_assignments

            result_solver_stats_time_solving = {}
            result_solver_stats_time_modeling = {}
            for param_rsa_look_ahead, DATA2 in DATA1.items():
                for seed, run in sorted(DATA2.items()):

                    solver_stats_time_solving = run.get('rsa_solver_stats_time_solving')
                    solver_stats_time_modeling = run.get('rsa_solver_stats_time_modeling')

                    solver_stats_time_solving = statistics.mean(solver_stats_time_solving) * 1000
                    solver_stats_time_modeling = statistics.mean(solver_stats_time_modeling) * 1000

                    try:
                        result_solver_stats_time_solving[param_rsa_look_ahead].append(solver_stats_time_solving)
                    except KeyError:
                        result_solver_stats_time_solving[param_rsa_look_ahead] = [solver_stats_time_solving]

                    try:
                        result_solver_stats_time_modeling[param_rsa_look_ahead].append(solver_stats_time_modeling)
                    except KeyError:
                        result_solver_stats_time_modeling[param_rsa_look_ahead] = [solver_stats_time_modeling]
            
            boxdata = []        
            datax = []
            datay = []
            print("")
            print("solving time a=%d" % param_rsa_max_assignments)
            for param_rsa_look_ahead, data in sorted(result_solver_stats_time_solving.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(statistics.median(data))
                boxdata.append(data)
                print(" ", param_rsa_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))

            axes[1][colcnt].boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            axes[1][colcnt].plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)

            boxdata = []
            datax = []
            datay = []
            print("")
            print("modeling time a=%d" % param_rsa_max_assignments)  
            for param_rsa_look_ahead, data in sorted(result_solver_stats_time_modeling.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(statistics.median(data))
                boxdata.append(data)
                print(" ", param_rsa_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))

            axes[0][colcnt].boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            axes[0][colcnt].plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)
            
            colcnt += 1

        h0, l0 = fig.axes[3].get_legend_handles_labels()
        h1, l1 = fig.axes[2].get_legend_handles_labels()
        h2, l2 = fig.axes[1].get_legend_handles_labels()
        h3, l3 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h3+h2+h1+h0, l3+l2+l1+l0, loc='upper center', ncol=4, fontsize=16)
        fig.subplots_adjust(top=0.90) # padding top
        if SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_rsa_solving_time_with_outliers.pdf', folder='lookahead')
        else:
            utils.export(fig, 'lookahead_rsa_solving_time.pdf', folder='lookahead')


