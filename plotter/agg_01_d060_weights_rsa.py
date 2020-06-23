import logging, math, json, pickle, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import time
from heapq import heappush, heappop

logger = logging.getLogger(__name__)

from . import agg_2_utils as utils

FIGSIZE = (8,10)
SHOW_GREY = False

def plot(blob, **kwargs):

    utils.EXPORT_BLOB = blob
  
    includes = [
        'scenario_switch_cnt', 
        'scenario_link_util_mbit_max', 
        'scenario_table_util_max_total',
        'scenario_rules_per_switch_avg',
        'rsa_table_fairness_avg',
        'rsa_link_util_delegated_mbit_max', 
        'rsa_ctrl_overhead_from_rsa',
        #'rsa_table_percent_shared',
        'rsa_table_percent_relocated']

    includes += blob.find_columns('hit_timelimit')
    blob.include_parameters(**dict.fromkeys(includes, 1))
    runs = blob.filter(**dict())  

    class ParameterSet(object):
        def __init__(self, run):

            self.w_table = run.get('param_rsa_weight_table')
            self.w_link = run.get('param_rsa_weight_link')
            self.w_ctrl = run.get('param_rsa_weight_ctrl')

            self.label = r'%d-%d-%d' % (self.w_table, self.w_link, self.w_ctrl)

            self.ctrl_overhead = []
            self.table_fairness = []
            self.percent_failed = []
            #self.table_overhead_percent = []
            #self.underutil_percent  = []
            self.runs = []

        def add_result(self, run):

            self.table_fairness.append((run.get('rsa_table_fairness_avg') / run.get('scenario_table_util_max_total')) * 100)
            self.ctrl_overhead.append((run.get('rsa_ctrl_overhead_from_rsa') / run.get('scenario_rules_per_switch_avg')) * 100)
            self.percent_failed.append(run.get('rsa_table_percent_relocated'))
            self.runs.append(run)
            """
            for switch in range(0, run.get('scenario_switch_cnt')):
                try:
                    if run.get('dts_%d_table_overhead_percent' % (switch)) > 0:
                        #self.switches.append(Switch(run, switch))
                        self.ctrl_overhead.append(run.get('dts_%d_ctrl_overhead_percent' % (switch)))
                        self.link_overhead_percent.append(run.get('dts_%d_link_overhead_percent' % (switch)))
                        self.table_overhead_percent.append(run.get('dts_%d_table_overhead_percent' % (switch)))
                        self.underutil_percent.append(run.get('dts_%d_underutil_percent' % (switch)))
                        self.runs.append((run, switch))
                except:
                    pass
                    #print("add failed", switch, run.get('scenario_switch_cnt'), self.label)
                    #print(run)
                    #exit()
            """


    timelimit = 0
    use_runs = []
    use_seeds = set()
    failed_seeds = []
    switchcnt = []
    for run in runs:
        if run.get('hit_timelimit') and run.get('hit_timelimit')  > 0:
            timelimit += 1
            if not run.get('param_topo_seed') in failed_seeds:
                failed_seeds.append(run.get('param_topo_seed'))
            continue

        use_runs.append(run)
        use_seeds.add(run.get('param_topo_seed'))
        switchcnt.append(run.get('scenario_switch_cnt'))

    print("len", len(runs))
    print("timelimit", timelimit)
    print("max", max(switchcnt))
    print("seeds", len(use_seeds), len(failed_seeds))


    results = {}
    for run in runs:
        if not run.get('hit_timelimit'):   
            seed = run.get('param_topo_seed')
            if seed in failed_seeds: continue;

            key = (run.get('param_rsa_weight_table'), run.get('param_rsa_weight_link'), run.get('param_rsa_weight_ctrl'))
            if key == (0,0,0):
                # this is a special case because the default parameters are chosen if all
                # weights are set to 0-0-0 (which is 1-0-5); it is therefore removed here
                continue
            try:
                results[key].add_result(run)
            except KeyError:
                results[key] = ParameterSet(run)


    def plotax(ax, data, **kwargs):
        data_x = []
        data_y = []
        for i, v in enumerate(sorted(data)):
            data_x.append(i)
            data_y.append(v)
        ax.plot(data_x, data_y,  **kwargs)


    # ---------------------------------------------
    # cdfs only
    # ---------------------------------------------
    if 1:
        plt.close()
        fig, axes = plt.subplots(2,2, figsize=(8, 8))
        fig.tight_layout(pad=0)

        cdf1 = axes[0][0]
        cdf1.set_xlabel('Allocation fairness in \\%')
        cdf2 = axes[0][1]
        cdf2.set_xlabel('Allocation overhead in \\%')
        cdf3 = axes[1][0]
        cdf3.set_xlabel('Failure rate in \\%')
        cdf4 = axes[1][1]
        cdf4.set_xlabel('Aggregated score')

        for ax in [cdf1, cdf2, cdf3, cdf4]:
            #ax.yaxis.tick_right()
            #ax.yaxis.set_label_position("right")
            ax.set_ylabel('CDF')
            #ax.set_xlim(0,80)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)


        all_results = [] 

        for m, result in results.items():
            #if 0 in m: continue;
            result.label
            combined = [(3*x+5*y+10*z)/18.0 for x,y,z in zip(result.table_fairness, result.ctrl_overhead, result.percent_failed)]

            color='lightgray'
            alpha = 0.1

            rating = sum(combined)
            all_results.append((rating, result))

            if SHOW_GREY:
                utils.plotcdf(cdf1, result.table_fairness, color=color, alpha=alpha) 
                utils.plotcdf(cdf2, result.ctrl_overhead, color=color, alpha=alpha) 
                utils.plotcdf(cdf3, result.percent_failed, color=color, alpha=alpha) 
                utils.plotcdf(cdf4, combined, color=color, alpha=alpha) 

        for m, color, marker, label, linestyle in zip([(1,0,0), (0,1,0), (0,0,1), (2,0,6)], 
            ['red', 'green', 'blue', 'black'], ['^', 's', 'o', '*'], 
            [utils.rsa_weights(1,0,0), utils.rsa_weights(0,1,0), 
                utils.rsa_weights(0,0,1), utils.rsa_weights(1,0,5) + '(best)'],
                ['-','-','-','--']):
            result = results.get(m)

            color=color
            markevery=20
            alpha = 1

            combined = [(3*x+5*y+10*z)/18.0 for x,y,z in zip(result.table_fairness, result.ctrl_overhead, result.percent_failed)]

            utils.plotcdf(cdf1, result.table_fairness, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle, label=label)
            utils.plotcdf(cdf2, result.ctrl_overhead, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle, label=label)
            utils.plotcdf(cdf3, result.percent_failed, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle, label=label)
            utils.plotcdf(cdf4, combined, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle, label=label)

        # sort results by rating
        all_results = sorted(all_results, key=lambda x: x[0])

        print("best scores:")
        for i in range(0,20):
            print(all_results[i][0], all_results[i][1].label)
        print("worst scores:")
        for i in range(1,10):
            print(all_results[-i][0], all_results[-i][1].label) 


        handles, labels = cdf4.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
        #plt.subplots_adjust(top=.9)
        plt.subplots_adjust(wspace=0.2, hspace=0.2, left = 0.07, top=.87, bottom=.08)
        utils.export(fig, 'weights_rsa_cdfonly.pdf', folder='weights')


    # ---------------------------------------------
    # Figure 1: results if only one of the three coefficients is used (e.g., wtable=1, others =0)
    # ---------------------------------------------
    if 0:
        plt.close()
        fig = plt.figure(figsize=(8, 10))
        fig.tight_layout(pad=0)
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((4, 3), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((4, 3), (2, 0), colspan=2)
        ax4 = plt.subplot2grid((4, 3), (3, 0), colspan=2)

        ax1.set_ylabel(r'Fairness deviation in ' + r'\%')
        ax1.set_xlabel('Experiment index sorted by fairness deviation')
        ax2.set_ylabel('Control overhead in \\%')
        ax2.set_xlabel('Experiment index sorted by control overhead')
        ax3.set_ylabel('Absolute failure rate in \\%')
        ax3.set_xlabel('Experiment index sorted by absolute failure rate')
        ax4.set_ylabel('Weighted score')
        ax4.set_xlabel('Experiment index sorted by weighted score') 


        cdf1 = plt.subplot2grid((4, 3), (0, 2))
        cdf1.set_xlabel('Fairness deviation in \\%')
        cdf2 = plt.subplot2grid((4, 3), (1, 2))
        cdf2.set_xlabel('Control overhead in \\%')
        cdf3 = plt.subplot2grid((4, 3), (2, 2))
        cdf3.set_xlabel('Absolute failure rate in \\%')
        cdf4 = plt.subplot2grid((4, 3), (3, 2))
        cdf4.set_xlabel('Weighted score')

        for ax in [cdf1, cdf2, cdf3, cdf4]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('CDF')
            ax.set_ylim(0,1)
            #ax.set_xlim(0,100)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)

        for ax in [ax1, ax2, ax3, ax4]:
            #ax.set_ylim(0,55)
            ax.set_xlim(0,100)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)


        all_results = [] 

        for m, result in results.items():
            #if 0 in m: continue;
            result.label
            combined = [(3*x+5*y+10*z)/18.0 for x,y,z in zip(result.table_fairness, result.ctrl_overhead, result.percent_failed)]

            color='lightgray'
            alpha = 0.1

            rating = sum(combined)
            all_results.append((rating, result))

            if SHOW_GREY:
                plotax(ax1, result.table_fairness, color=color, alpha=alpha) 
                plotax(ax2, result.ctrl_overhead, color=color, alpha=alpha)
                plotax(ax3, result.percent_failed, color=color, alpha=alpha)
                plotax(ax4, combined, color=color, alpha=alpha)

                utils.plotcdf(cdf1, result.table_fairness, color=color, alpha=alpha) 
                utils.plotcdf(cdf2, result.ctrl_overhead, color=color, alpha=alpha) 
                utils.plotcdf(cdf3, result.percent_failed, color=color, alpha=alpha) 
                utils.plotcdf(cdf4, combined, color=color, alpha=alpha) 

        for m, color, marker, label, linestyle in zip([(1,0,0), (0,1,0), (0,0,1), (1,0,5)], 
            ['red', 'green', 'blue', 'black'], ['^', 's', 'o', '*'], 
            [r'wTable=1, wLink=0, wCtrl=0 (table only)', r'wTable=0, wLink=1, wCtrl=0 (link only)', 
                r'wTable=0, wLink=0, wCtrl=1 (ctrl only)', 'wTable=1, wLink=0, wCtrl=5 (best combination)'],
                ['-','-','-','--']):
            result = results.get(m)

            color=color
            markevery=20
            alpha = 1

            combined = [(3*x+5*y+10*z)/18.0 for x,y,z in zip(result.table_fairness, result.ctrl_overhead, result.percent_failed)]

            plotax(ax1, result.table_fairness, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4, label=label, linestyle=linestyle) 
            plotax(ax2, result.ctrl_overhead, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4,  label=label, linestyle=linestyle)
            plotax(ax3, result.percent_failed, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4,  label=label, linestyle=linestyle)
            plotax(ax4, combined, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4,  label=label, linestyle=linestyle)


            utils.plotcdf(cdf1, result.table_fairness, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle) 
            utils.plotcdf(cdf2, result.ctrl_overhead, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle) 
            utils.plotcdf(cdf3, result.percent_failed, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle) 
            utils.plotcdf(cdf4, combined, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle) 

        # sort results by rating
        all_results = sorted(all_results, key=lambda x: x[0])

        print("best scores:")
        for i in range(0,20):
            print(all_results[i][0], all_results[i][1].label)
        print("worst scores:")
        for i in range(1,10):
            print(all_results[-i][0], all_results[-i][1].label) 


        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
        plt.subplots_adjust(wspace=0.1, hspace=0.3, top=.90, bottom=.05)
        utils.export(fig, 'weights_rsa.pdf', folder='weights')

