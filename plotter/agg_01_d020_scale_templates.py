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
    includes += blob.find_columns('table_datay_raw')
    includes += blob.find_columns('overutil_max')
    includes += blob.find_columns('solver_stats_time_modeling_other')
    includes += blob.find_columns('solver_stats_time_modeling_demand')
    includes += blob.find_columns('table_datay_raw')
    includes += blob.find_columns('table_overhead')



    blob.include_parameters(**dict.fromkeys(includes, 1))

    fig_overhead, ax_overhead  = plt.subplots(2,2, figsize=(12, 8), sharex=True, sharey=True)
    fig_overhead.tight_layout(pad=3)

    for ax in fig_overhead.axes:
        ax.set_xlabel(r'Number of delegation templates ($|H|$)', fontsize=15)
        ax.set_ylabel('Mean table overhead', fontsize=15)
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)

    axovhc = 0
    for param_topo_switch_capacity in [30,50, 70,90]:




        fig, axes = plt.subplots(1,2,figsize=(12, 6))
        fig.tight_layout(pad=3)
        axes[0].set_xlabel(r'Number of delegation templates ($|H|$)', fontsize=15)
        axes[0].set_ylabel('Mean modeling time (ms)', fontsize=15)
        axes[1].set_xlabel(r'Number of delegation templates ($|H|$)', fontsize=15)
        axes[1].set_ylabel('Mean solving time (ms)', fontsize=15)

        ax2 = None
        maxax2 = 0

        allvals = []
        for flows, color, marker in zip([100000,200000], ['green', 'blue'], ['s', 'o']):
            datax = []
            datay_solving = []
            datay_modeling = []

            datay_solving2 = []
            datay_modeling2 = []

            datay_infeasible = []
            datay_samples = []
            datay_overhead = []
            capacities = []

            for param_topo_num_hosts in range(25,525,25):
                runs = blob.filter(param_topo_switch_capacity=param_topo_switch_capacity,param_topo_num_flows=flows, param_topo_num_hosts=param_topo_num_hosts)
                modeling = []
                solving = []   
                overhead = []
                infeasible = 0  
                samples = []
                assert(len(runs) == 10)  
                for run in runs:
                    infeasible += run.get('dts_%d_solver_cnt_infeasable' % 0)
                    s = [x * 1000 for x in run.get('dts_%d_solver_stats_time_solving' % 0) if x > 0]
                    m = [x * 1000 for x in run.get('dts_%d_solver_stats_time_modeling' % 0) if x > 0]
                    modeling.append(statistics.mean(m))
                    solving.append(statistics.mean(s))
                    samples.append(len(m))
                    capacities.append(run.get('scenario_table_capacity'))

                    thresh = run.get('scenario_table_capacity')
                    d2 = 'dts_%d_table_datay_raw' % (0)
                    raw_util = run.get(d2)
                    fill_overutil = [1 if x > thresh else 0 for x in raw_util]
                    new_table_overhead_percent = (run.get('dts_%d_table_overhead' % 0) / float(sum(fill_overutil)))

                    overhead.append(new_table_overhead_percent)

                datax.append(param_topo_num_hosts)
                datay_solving.append(statistics.mean(solving))
                datay_modeling.append(statistics.mean(modeling))

                datay_solving2.append(np.percentile(solving, 100))
                datay_modeling2.append(np.percentile(modeling, 100))

                datay_infeasible.append(infeasible)
                datay_samples.append(sum(samples))

                datay_overhead.append(np.percentile(overhead, 50))

            t1 = int(sum(capacities) / len(capacities))
            axes[0].plot(datax, datay_modeling, label='%d pairs [capacity=%d]' % (flows, t1), color=color, marker=marker)
            axes[1].plot(datax, datay_solving, label='%d pairs [capacity=%d]' % (flows, t1),  color=color, marker=marker)



            axes[0].text(0.02, 0.9, 
               ('Capacity reduction: %d' % (100-param_topo_switch_capacity)) + r'\%',
                color='black', fontsize=20,
                transform=axes[0].transAxes,
                horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='white'))  

            axes[0].plot(datax, datay_modeling2, label='maximum (100th percentil)', color=color, linestyle="--", alpha=0.3)
            axes[1].plot(datax, datay_solving2,  label='maximum (100th percentil)',  color=color, linestyle="--", alpha=0.3)

            print(100-param_topo_switch_capacity, flows, sum(datay_samples))




            fig_overhead.axes[3-axovhc].text(0.02, 0.9, 
               ('Capacity reduction: %d' % (100-param_topo_switch_capacity)) + r'\%',
                color='black', fontsize=20,
                transform=fig_overhead.axes[3-axovhc].transAxes,
                horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='white'))  

            if flows  == 100000:
                fig_overhead.axes[3-axovhc].text(0.02, 0.75, 
                   (r'$c^\texttt{Table}_s$:' +' %d' % (t1)),
                    color='green', fontsize=18,
                    transform=fig_overhead.axes[3-axovhc].transAxes,
                    horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='white'))  

            if flows  == 200000:
                fig_overhead.axes[3-axovhc].text(0.02, 0.60, 
                   (r'$c^\texttt{Table}_s$:' + ' %d' % (t1)),
                    color='blue', fontsize=18,
                    transform=fig_overhead.axes[3-axovhc].transAxes,
                    horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='white'))  

            fig_overhead.axes[3-axovhc].plot(datax, datay_overhead, label='%d pairs' % (
                flows), color=color, marker=marker)

            """
            t2 = run.get('scenario_table_util_max_total')
            if flows == 100000:
                axes[1].text(0.02, 0.9, r'\textbf{' + ('%d - %d' % (t1,t2)) + r'ms}', 
                    transform=axes[1].transAxes, color=color, fontsize=14,
                    horizontalalignment='left', bbox=dict(facecolor='white', edgecolor=color)) 

            if flows == 200000:
                axes[1].text(0.02, 0.8, r'\textbf{' + ('%d - %d' % (t1,t2)) + r'ms}', 
                    transform=axes[1].transAxes, color=color, fontsize=14,
                    horizontalalignment='left', bbox=dict(facecolor='white', edgecolor=color)) 
            """

            if sum(datay_infeasible) > 0:
                if not ax2:
                    ax2 = axes[1].twinx()
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.set_ylabel(r'Infeasible in \%', color="red", fontsize=15)
                ax2.plot(datax, [(x/max(datay_infeasible)*100.0) for x in datay_infeasible], color='red', label='Infeasible (with 100000 pairs)',
                    linestyle='--', linewidth=0.5, ms=3, marker=marker)
                if max(datay_infeasible) > maxax2:
                    maxax2 = max(datay_infeasible)


            allvals += datay_modeling + datay_solving + datay_modeling2 + datay_solving2

        handles, labels = axes[1].get_legend_handles_labels()
        ncol = 2
        if ax2:
            h2, l2 = ax2.get_legend_handles_labels()  
            handles += h2
            labels += l2  
            ncol += 1
        fig.legend(handles, labels, loc='upper center', ncol=ncol, fontsize=16)
        if ax2:
            fig.subplots_adjust(top=0.83, left=0.05, right=0.95) # no gap
        else:
            fig.subplots_adjust(top=0.83)
        for ax in fig.axes:
            if ax == ax2: 
                ax.set_ylim(0,100)
                continue
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylim(0, int(max(allvals)*1.1))

        #if ax2:
        #    ax2.set_ylim(0,1000)


        utils.export(fig, 'scale_templates_%d.pdf' % (100-param_topo_switch_capacity), folder='runtime')
        axovhc += 1


    handles, labels = fig_overhead.axes[3].get_legend_handles_labels()
    fig_overhead.legend(handles, labels, loc='upper center', ncol=2, fontsize=16)
    fig_overhead.subplots_adjust(top=0.9)

    utils.export(fig_overhead, 'scale_templates__overhead.pdf', folder='runtime')
    plt.close()