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
    # failure rate summary
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.tight_layout(pad=2.7)  
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)     
        ax.set_ylabel(r'Failure rate (\%)', fontsize=15)
        ax.set_xlabel(r'Look-ahead factor L', fontsize=15)

        for param_rsa_max_assignments, DATA1 in sorted(DATA.items()):
            if param_rsa_max_assignments not in MAX_ASSIGNMENTS.keys():
                continue
            color = MAX_ASSIGNMENTS.get(param_rsa_max_assignments).get('color')
            label = '%d assignments' % param_rsa_max_assignments
            result_rsa_table_percent_relocated = {}
            result_rsa_ctrl_overhead_from_rsa = {}
            result_rsa_table_fairness_avg = {}
            for param_rsa_look_ahead, DATA2 in DATA1.items():
                for seed, run in sorted(DATA2.items()):
                    rsa_table_percent_relocated = run.get('rsa_table_percent_relocated')
                    if rsa_table_percent_relocated is not None:
                        rsa_ctrl_overhead_from_rsa = run.get('rsa_ctrl_overhead_from_rsa')
                        rsa_table_fairness_avg = run.get('rsa_table_fairness_avg')
                        try:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead].append(rsa_table_percent_relocated)
                        except KeyError:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead] = [rsa_table_percent_relocated]      
                        try:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead].append(rsa_ctrl_overhead_from_rsa)
                        except KeyError:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead] = [rsa_ctrl_overhead_from_rsa]
                        try:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead].append(rsa_table_fairness_avg)
                        except KeyError:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead] = [rsa_table_fairness_avg]
            
            boxdata = []        
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_rsa_table_percent_relocated.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(np.percentile(data, 75))
                boxdata.append(data)
            #ax.boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            ax.plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)

        h2, l2 = ax.get_legend_handles_labels()
        fig.legend(h2, l2, loc='upper center', ncol=4, fontsize=16)
        fig.subplots_adjust(top=0.88) 


        if SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_rsa_failure_rate_1_with_outliers.pdf', folder='lookahead/rsa_metrics')
        else:
            utils.export(fig, 'lookahead_rsa_failure_rate_1.pdf', folder='lookahead/rsa_metrics')


    # -----------------------
    # failure rate 4er plot
    # -----------------------
    if 1:

        plt.close()
        fig, axes = plt.subplots(2, 2, figsize=(14, 8),  sharex=True, sharey=True)
        fig.tight_layout(pad=2.7)

        for ax in fig.axes:
            ax.set_ylabel(r'Failure rate (\%)', fontsize=15)
            ax.set_xlabel(r'Look-ahead factor L', fontsize=15)
        #fig.axes[3].set_xlabel(r'Look-ahead factor L', fontsize=15)

        for ax in fig.axes:
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)     

        colcnt = 0
        for param_rsa_max_assignments, DATA1 in sorted(DATA.items()):

            if param_rsa_max_assignments not in MAX_ASSIGNMENTS.keys():
                continue
            color = MAX_ASSIGNMENTS.get(param_rsa_max_assignments).get('color')
            label = '%d assignments' % param_rsa_max_assignments
            result_rsa_table_percent_relocated = {}
            result_rsa_ctrl_overhead_from_rsa = {}
            result_rsa_table_fairness_avg = {}
            for param_rsa_look_ahead, DATA2 in DATA1.items():
                for seed, run in sorted(DATA2.items()):
                    rsa_table_percent_relocated = run.get('rsa_table_percent_relocated')
                    if rsa_table_percent_relocated is not None:
                        rsa_ctrl_overhead_from_rsa = run.get('rsa_ctrl_overhead_from_rsa')
                        rsa_table_fairness_avg = run.get('rsa_table_fairness_avg')
                        try:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead].append(rsa_table_percent_relocated)
                        except KeyError:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead] = [rsa_table_percent_relocated]      
                        try:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead].append(rsa_ctrl_overhead_from_rsa)
                        except KeyError:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead] = [rsa_ctrl_overhead_from_rsa]
                        try:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead].append(rsa_table_fairness_avg)
                        except KeyError:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead] = [rsa_table_fairness_avg]
            
            boxdata = []        
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_rsa_table_percent_relocated.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(np.percentile(data, 50))
                boxdata.append(data)
            fig.axes[colcnt].boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            fig.axes[colcnt].plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)
            colcnt += 1


        for ax in fig.axes:      
            labels = []
            for x in range(1,30):
                if x % 2 == 1:
                    labels.append(''+str(x))
                else:
                    labels.append('')
            ax.set_xticklabels(labels)

        h0, l0 = fig.axes[3].get_legend_handles_labels()
        h1, l1 = fig.axes[2].get_legend_handles_labels()
        h2, l2 = fig.axes[1].get_legend_handles_labels()
        h3, l3 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h3+h2+h1+h0, l3+l2+l1+l0, loc='upper center', ncol=4, fontsize=16)
        fig.subplots_adjust(top=0.90) # padding top
        if SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_rsa_failure_rate_4_with_outliers.pdf', folder='lookahead/rsa_metrics')
        else:
            utils.export(fig, 'lookahead_rsa_failure_rate_4.pdf', folder='lookahead/rsa_metrics')

    # -----------------------
    # allocation overhead
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.tight_layout(pad=2.7)  
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)     
        ax.set_ylabel(r'Allocation overhead (messages)', fontsize=15)
        ax.set_xlabel(r'Look-ahead factor L', fontsize=15)

        for param_rsa_max_assignments, DATA1 in sorted(DATA.items()):
            if param_rsa_max_assignments not in MAX_ASSIGNMENTS.keys():
                continue
            color = MAX_ASSIGNMENTS.get(param_rsa_max_assignments).get('color')
            label = '%d assignments' % param_rsa_max_assignments
            result_rsa_table_percent_relocated = {}
            result_rsa_ctrl_overhead_from_rsa = {}
            result_rsa_table_fairness_avg = {}
            for param_rsa_look_ahead, DATA2 in DATA1.items():
                for seed, run in sorted(DATA2.items()):
                    rsa_table_percent_relocated = run.get('rsa_table_percent_relocated')
                    rsa_ctrl_overhead_from_rsa = run.get('rsa_ctrl_overhead_from_rsa')
                    if rsa_ctrl_overhead_from_rsa is not None:
                        rsa_table_fairness_avg = run.get('rsa_table_fairness_avg')
                        try:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead].append(rsa_table_percent_relocated)
                        except KeyError:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead] = [rsa_table_percent_relocated]      
                        try:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead].append(rsa_ctrl_overhead_from_rsa)
                        except KeyError:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead] = [rsa_ctrl_overhead_from_rsa]
                        try:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead].append(rsa_table_fairness_avg)
                        except KeyError:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead] = [rsa_table_fairness_avg]
            
            boxdata = []        
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_rsa_ctrl_overhead_from_rsa.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(np.percentile(data, 75))
                boxdata.append(data)
            #ax.boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            ax.plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)

        if SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_rsa_allocation_1_with_outliers.pdf', folder='lookahead/rsa_metrics')
        else:
            utils.export(fig, 'lookahead_rsa_allocation_1.pdf', folder='lookahead/rsa_metrics')

    # -----------------------
    # allocation overhead 4er plot
    # -----------------------
    if 1:

        plt.close()
        fig, axes = plt.subplots(2, 2, figsize=(14, 8),  sharex=True, sharey=True)
        fig.tight_layout(pad=2.7)

        for ax in fig.axes:
            ax.set_ylabel(r'Allocation overhead (messages)', fontsize=15)
            ax.set_xlabel(r'Look-ahead factor L', fontsize=15)

        for ax in fig.axes:
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)     

        colcnt = 0
        for param_rsa_max_assignments, DATA1 in sorted(DATA.items()):

            if param_rsa_max_assignments not in MAX_ASSIGNMENTS.keys():
                continue
            color = MAX_ASSIGNMENTS.get(param_rsa_max_assignments).get('color')
            label = '%d assignments' % param_rsa_max_assignments
            result_rsa_table_percent_relocated = {}
            result_rsa_ctrl_overhead_from_rsa = {}
            result_rsa_table_fairness_avg = {}
            for param_rsa_look_ahead, DATA2 in DATA1.items():
                for seed, run in sorted(DATA2.items()):
                    rsa_table_percent_relocated = run.get('rsa_table_percent_relocated')
                    rsa_ctrl_overhead_from_rsa = run.get('rsa_ctrl_overhead_from_rsa')
                    rsa_table_fairness_avg = run.get('rsa_table_fairness_avg')
                    if rsa_ctrl_overhead_from_rsa is not None:
                        try:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead].append(rsa_table_percent_relocated)
                        except KeyError:
                            result_rsa_table_percent_relocated[param_rsa_look_ahead] = [rsa_table_percent_relocated]      
                        try:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead].append(rsa_ctrl_overhead_from_rsa)
                        except KeyError:
                            result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead] = [rsa_ctrl_overhead_from_rsa]
                        try:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead].append(rsa_table_fairness_avg)
                        except KeyError:
                            result_rsa_table_fairness_avg[param_rsa_look_ahead] = [rsa_table_fairness_avg]
            
            boxdata = []        
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_rsa_ctrl_overhead_from_rsa.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(np.percentile(data, 75))
                boxdata.append(data)
            fig.axes[colcnt].boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            fig.axes[colcnt].plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)
            colcnt += 1

        h0, l0 = fig.axes[3].get_legend_handles_labels()
        h1, l1 = fig.axes[2].get_legend_handles_labels()
        h2, l2 = fig.axes[1].get_legend_handles_labels()
        h3, l3 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h3+h2+h1+h0, l3+l2+l1+l0, loc='upper center', ncol=4, fontsize=16)
        fig.subplots_adjust(top=0.90) # padding top
        if SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_rsa_allocation_4_with_outliers.pdf', folder='lookahead/rsa_metrics')
        else:
            utils.export(fig, 'lookahead_rsa_allocation_4.pdf', folder='lookahead/rsa_metrics')

    # -----------------------
    # Figure: box plots for all three metrics, ordered by look-ahead
    # -----------------------
    if 0:

        plt.close()
        fig, axes = plt.subplots(3, len(MAX_ASSIGNMENTS), figsize=(14, 8),  sharex=True)
        fig.tight_layout(pad=2.7)

        for ax, label in zip([axes[x][0] for x in range(0,3)], [r'Absolute failure rate (\%)', r'Control overhead', r'Fairness deviation']):
            ax.set_ylabel('%s' % label, fontsize=15)

        # force shared y axis
        for y in range(0,3):
            useax = [axes[y][x] for x in range(0,len(MAX_ASSIGNMENTS))]
            for ax in useax:
                ax.get_shared_y_axes().join(*useax)


        for ax in [axes[2][x] for x in range(0,len(MAX_ASSIGNMENTS))]:
            ax.set_xlabel('Look-ahead factor L', fontsize=15)
        for ax in fig.axes:
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)     


        

        colcnt = 0
        for param_rsa_max_assignments, DATA1 in sorted(DATA.items()):

            if param_rsa_max_assignments not in MAX_ASSIGNMENTS.keys():
                continue
            color = MAX_ASSIGNMENTS.get(param_rsa_max_assignments).get('color')
            label = '%d assignments' % param_rsa_max_assignments


            #if param_rsa_max_assignments != 20: continue;
            result_rsa_table_percent_relocated = {}
            result_rsa_ctrl_overhead_from_rsa = {}
            result_rsa_table_fairness_avg = {}
            for param_rsa_look_ahead, DATA2 in DATA1.items():
                for seed, run in sorted(DATA2.items()):

                    rsa_table_percent_relocated = run.get('rsa_table_percent_relocated')
                    rsa_ctrl_overhead_from_rsa = run.get('rsa_ctrl_overhead_from_rsa')
                    rsa_table_fairness_avg = run.get('rsa_table_fairness_avg')

                    #datax.append(param_rsa_look_ahead)
                    try:
                        result_rsa_table_percent_relocated[param_rsa_look_ahead].append(rsa_table_percent_relocated)
                    except KeyError:
                        result_rsa_table_percent_relocated[param_rsa_look_ahead] = [rsa_table_percent_relocated]

                    
                    try:
                        result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead].append(rsa_ctrl_overhead_from_rsa)
                    except KeyError:
                        result_rsa_ctrl_overhead_from_rsa[param_rsa_look_ahead] = [rsa_ctrl_overhead_from_rsa]

                    
                    try:
                        result_rsa_table_fairness_avg[param_rsa_look_ahead].append(rsa_table_fairness_avg)
                    except KeyError:
                        result_rsa_table_fairness_avg[param_rsa_look_ahead] = [rsa_table_fairness_avg]
            
            boxdata = []        
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_rsa_table_percent_relocated.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(statistics.median(data))
                boxdata.append(data)
            axes[0][colcnt].boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            axes[0][colcnt].plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)

            boxdata = []
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_rsa_ctrl_overhead_from_rsa.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(statistics.median(data))
                boxdata.append(data)
            axes[1][colcnt].boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            axes[1][colcnt].plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)
            
            boxdata = []
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_rsa_table_fairness_avg.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(statistics.median(data))
                boxdata.append(data)
            axes[2][colcnt].boxplot(boxdata, notch=False, showfliers=SHOW_OUTLIERS)
            axes[2][colcnt].plot(datax, datay, color=color, marker='o', linestyle="--", linewidth=2, label=label)
            
            colcnt += 1

        h0, l0 = fig.axes[3].get_legend_handles_labels()
        h1, l1 = fig.axes[2].get_legend_handles_labels()
        h2, l2 = fig.axes[1].get_legend_handles_labels()
        h3, l3 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h3+h2+h1+h0, l3+l2+l1+l0, loc='upper center', ncol=4, fontsize=16)
        fig.subplots_adjust(top=0.90) # padding top
        if SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_rsa_metrics_with_outliers.pdf', folder='lookahead')
        else:
            utils.export(fig, 'lookahead_rsa_metrics.pdf', folder='lookahead')


    # -----------------------
    # Figure: modeling and solving time based on look-ahead
    # -----------------------
    if 0:
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























    # -----------------------
    # Figure: modeling and solving time based on look-ahead (individual plots, not used)
    # -----------------------
    if False:

        for param_rsa_max_assignments, DATA1 in DATA.items():

            plt.close()
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.tight_layout(pad=2.7)
            for ax, label in zip(axes, ['Modeling time (ms)', 'Solving time (ms)']):
                ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_xlabel('Look-ahead factor L', fontsize=15)
                ax.set_ylabel('%s' % label, fontsize=15)


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
            for param_rsa_look_ahead, data in sorted(result_solver_stats_time_solving.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(statistics.median(data))
                boxdata.append(data)
            axes[0].boxplot(boxdata, notch=True, showfliers=False)
            axes[0].plot(datax, datay, color="black", marker='o', linestyle=":", linewidth=2)

            boxdata = []
            datax = []
            datay = []
            for param_rsa_look_ahead, data in sorted(result_solver_stats_time_modeling.items()):
                datax.append(param_rsa_look_ahead)
                datay.append(statistics.median(data))
                boxdata.append(data)
            axes[1].boxplot(boxdata, notch=True, showfliers=False)
            axes[1].plot(datax, datay, color="black", marker='o', linestyle=":", linewidth=2)

            utils.export(fig, 'lookahead_rsa_solving_times_%d.pdf' % param_rsa_max_assignments, folder='lookahead')

    # -----------------------
    # Figure: infeasible solutions based on look-ahead
    # -----------------------
    if False:
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.tight_layout(pad=2.7)
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Look-ahead factor L', fontsize=15)
        ax.set_ylabel('Infeasible solutions', fontsize=15)

        for param_dts_algo, DATA1 in DATA.items():
            color = 'red'
            if param_dts_algo == 1:
                color = 'blue'
                continue
            result_solver_cnt_infeasable = {}
            result_solver_stats_time_modeling = {}

            for seedswitch, DATA2 in DATA1.items():
                seed, switch = seedswitch
                datax = []
                datay = []
                for param_dts_look_ahead, run in sorted(DATA2.items()):
                    if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                        continue
                    solver_cnt_infeasable = run.get('dts_%d_solver_cnt_infeasable' % (switch))
                    if solver_cnt_infeasable > 0:
                        try:
                            result_solver_cnt_infeasable[param_dts_look_ahead].append(solver_cnt_infeasable)
                        except KeyError:
                            result_solver_cnt_infeasable[param_dts_look_ahead] = [solver_cnt_infeasable]

            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_solver_cnt_infeasable.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            ax.boxplot(result_solver_cnt_infeasable.values(), notch=False, showfliers=False)
            ax.plot(datax, datay, color="black", marker='o', linestyle=":", linewidth=2)

        plt.show()
    
