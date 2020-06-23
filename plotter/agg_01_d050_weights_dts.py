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


SHOW_GREY = False

def plot(blob, **kwargs):
    """DTS functional evaluation"""

    utils.EXPORT_BLOB = blob
  
    class ParameterSet(object):
        def __init__(self, run):

            self.w_table = run.get('param_dts_weight_table')
            self.w_link = run.get('param_dts_weight_link')
            self.w_ctrl = run.get('param_dts_weight_ctrl')

            self.label = r'%d-%d-%d' % (self.w_table, self.w_link, self.w_ctrl)

            self.ctrl_overhead_percent = []
            self.link_overhead_percent = []
            self.table_overhead_percent = []
            self.underutil_percent  = []
            self.runs = []

        def add_result(self, run):
            for switch in range(0, run.get('scenario_switch_cnt')):
                try:
                    if run.get('dts_%d_table_overhead_percent' % (switch)) > 0:
                        #self.switches.append(Switch(run, switch))
                        self.ctrl_overhead_percent.append(run.get('dts_%d_ctrl_overhead_percent' % (switch)))
                        self.link_overhead_percent.append(run.get('dts_%d_link_overhead_percent' % (switch)))
                        self.table_overhead_percent.append(run.get('dts_%d_table_overhead_percent' % (switch)))
                        self.underutil_percent.append(run.get('dts_%d_underutil_percent' % (switch)))
                        self.runs.append((run, switch))
                        assert(len(self.ctrl_overhead_percent) == len(self.link_overhead_percent) == len(self.table_overhead_percent) == len(self.underutil_percent))
                except:
                    pass
                    print("add failed", switch, run.get('scenario_switch_cnt'), self.label)
                    #print(run)
                    #exit()

    def plotax(ax, data, **kwargs):
        data_x = []
        data_y = []
        for i, v in enumerate(sorted(data)):
            data_x.append(i)
            data_y.append(v)
        ax.plot(data_x, data_y,  **kwargs)



    includes = ['scenario_switch_cnt']
    keywords = ['hit_timelimit',
        'ctrl_overhead_percent',
        'link_overhead_percent',
        'table_overhead_percent',
        'underutil_percent']
    for keyword in keywords:
        includes += blob.find_columns(keyword)

    blob.include_parameters(**dict.fromkeys(includes, 1))


    runs = blob.filter(**dict())
    
    skipped = 0
    use_seeds = set()
    ignore_seeds = []
    switchcnt = []
    results = {}
    for run in runs:
        seed = run.get('param_topo_seed')
        if run.get('hit_timelimit') and run.get('hit_timelimit')  > 0:
            skipped += 1
            if not seed in ignore_seeds:
                ignore_seeds.append(seed)
            continue


    for run in runs:
        seed = run.get('param_topo_seed')
        if not seed in ignore_seeds:
            key = (run.get('param_dts_weight_table'), run.get('param_dts_weight_link'), run.get('param_dts_weight_ctrl'))
            try:
                results[key].add_result(run)
            except KeyError:
                results[key] = ParameterSet(run)

            use_seeds.add(run.get('param_topo_seed'))
            switchcnt.append(run.get('scenario_switch_cnt'))

    print("len", len(runs))
    print("max", max(switchcnt))
    print("seeds", len(use_seeds))
    print("ignore_seeds", len(ignore_seeds))
    print("skipped", skipped)


    # ---------------------------------------------
    # Figure 1 (not used atm): results if only one of the three coefficients is used (e.g., wtable=1, others =0)
    # ---------------------------------------------
    if 0:
        plt.close()
        fig = plt.figure(figsize=(8, 10))
        fig.tight_layout(pad=0)
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((4, 3), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((4, 3), (2, 0), colspan=2)
        ax4 = plt.subplot2grid((4, 3), (3, 0), colspan=2)

        cdf1 = plt.subplot2grid((4, 3), (0, 2))
        cdf1.set_xlabel('Table overhead in %')
        cdf2 = plt.subplot2grid((4, 3), (1, 2))
        cdf2.set_xlabel('Link overhead in %')
        cdf3 = plt.subplot2grid((4, 3), (2, 2))
        cdf3.set_xlabel('Control overhead in %')
        cdf4 = plt.subplot2grid((4, 3), (3, 2))
        cdf4.set_xlabel('Combined overhead in %')

        for ax in [cdf1, cdf2, cdf3, cdf4]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('CDF')
            ax.set_xlim(0,80)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylim(0,55)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)

        for m, color, marker, label in zip([(1,0,0), (0,1,0), (0,0,1)], 
            ['red', 'green', 'blue'], ['^', 's', 'o'], 
            ['wTable=1, wLink=0, wCtrl=0', 'wTable=0, wLink=1, wCtrl=0', 
                'wTable=0, wLink=0, wCtrl=1']):
            result = results.get(m)

            color=color
            markevery=20
            alpha = 1

            combined = [(x+y+z)/3.0 for x,y,z in zip(result.ctrl_overhead_percent, result.link_overhead_percent, result.table_overhead_percent)]

            plotax(ax1, result.table_overhead_percent, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4, label=label) 
            plotax(ax2, result.link_overhead_percent, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4,  label=label)
            plotax(ax3, result.ctrl_overhead_percent, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4, label=label)
            plotax(ax4, combined, 
                color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4, label=label)

            utils.plotcdf(cdf1, result.table_overhead_percent, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha) 
            utils.plotcdf(cdf2, result.link_overhead_percent, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha) 
            utils.plotcdf(cdf3, result.ctrl_overhead_percent, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha) 
            utils.plotcdf(cdf4, combined, 
                color=color, marker=marker, markevery=15, ms=4, alpha=alpha) 

            """
            medium = -1
            for i, v in enumerate(sorted(result.table_overhead_percent)):
                if v > 20 and v < 30:
                    print(i, v)
                    medium = (i,v)
                    run, switch = result.runs[i]
                    v_link = run.get('dts_%d_link_overhead_percent' % (switch))
                    i_link = list(sorted(result.link_overhead_percent)).index(v_link)
                    medium_link = (i_link, v_link)

                    v_ctrl = run.get('dts_%d_ctrl_overhead_percent' % (switch))
                    i_ctrl = list(sorted(result.ctrl_overhead_percent)).index(v_ctrl)
                    medium_ctrl = (i_ctrl, v_ctrl)

            run, switch = result.runs[medium[0]]

            circle1 = plt.Circle(medium, 2, color='black')
            ax1.add_artist(circle1)

            circle2 = plt.Circle(medium_link, 2, color='black')
            ax2.add_artist(circle2)

            circle3 = plt.Circle(medium_ctrl, 2, color='black')
            ax3.add_artist(circle3)

            plt.show()




            #run, switch = result.runs[result.table_overhead_percent.index(max(result.table_overhead_percent))]


            #print(run, switch)
            plt.close()
            dts_fig = utils.plot_dts_utilization_over_time(blob, run, switch, filter=dict(
                param_topo_seed=run.get('param_topo_seed'),
                param_dts_weight_table=run.get('param_dts_weight_table'),
                param_dts_weight_link=run.get('param_dts_weight_link'),
                param_dts_weight_ctrl=run.get('param_dts_weight_ctrl')))
            plt.show()
            plt.close()
            dts_fig = utils.plot_dts_utilization_over_time(blob, run, switch, filter=dict(
                param_topo_seed=run.get('param_topo_seed'),
                param_dts_weight_table=0,
                param_dts_weight_link=0,
                param_dts_weight_ctrl=1))
            plt.show()
            plt.close()
            dts_fig = utils.plot_dts_utilization_over_time(blob, run, switch, filter=dict(
                param_topo_seed=run.get('param_topo_seed'),
                param_dts_weight_table=1,
                param_dts_weight_link=1,
                param_dts_weight_ctrl=1))

            plt.show()
            exit(1)
            """

        ax1.legend()
        ax1.set_ylabel('Table overhead in %')
        ax1.set_xlabel('DTS experiment index sorted by table overhead')

        ax2.legend()
        ax2.set_ylabel('Link Overhead in %')
        ax2.set_xlabel('DTS experiment index sorted by link overhead')

        ax3.legend()
        ax3.set_ylabel('Control Overhead in %')
        ax3.set_xlabel('DTS experiment index sorted by control overhead')

        ax4.legend()
        ax4.set_ylabel('Combined Overhead in %')
        ax4.set_xlabel('DTS experiment index sorted by combined overhead') 

        plt.subplots_adjust(wspace=0.1, hspace=0.5, top=.95, bottom=.05)
        plt.show()

    # ---------------------------------------------
    # only cdfs
    # ---------------------------------------------
    if 1:
        plt.close()
        all_results = [] 

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.tight_layout(pad=0)

        cdf1 = axes[0][0]
        cdf1.set_xlabel(r'Normalized table overhead in \%', fontsize=15)
        cdf2 = axes[0][1]
        cdf2.set_xlabel(r'Normalized link overhead in \%', fontsize=15)
        cdf3 = axes[1][0]
        cdf3.set_xlabel(r'Normalized control overhead in \%', fontsize=15)
        cdf4 =axes[1][1]
        cdf4.set_xlabel(r'Aggregated score', fontsize=15)

        for ax in [cdf1, cdf2, cdf3, cdf4]:
            #ax.yaxis.tick_right()
            #ax.yaxis.set_label_position("right")
            ax.set_ylabel('CDF')
            #ax.set_xlim(0,80)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)

        for m, result in results.items():
            #if 0 in m: continue;
            result.label
            combined = [(x+y+z)/3.0 for x,y,z in zip(result.ctrl_overhead_percent, result.link_overhead_percent, result.table_overhead_percent)]
            color='lightgray'
            alpha = 0.1
            rating = sum(combined)/3.0
            v1 = float(result.label.split('-')[0])
            v2 = float(result.label.split('-')[1])
            v3 = float(result.label.split('-')[2])
            try:
                all_results.append((rating, result, v1/v2, v2/v3))
            except ZeroDivisionError:
                all_results.append((rating, result, -1, -1))
            if SHOW_GREY:
                utils.plotcdf(cdf1, result.table_overhead_percent, color=color, alpha=alpha) 
                utils.plotcdf(cdf2, result.link_overhead_percent, color=color, alpha=alpha) 
                utils.plotcdf(cdf3, result.ctrl_overhead_percent, color=color, alpha=alpha) 
                utils.plotcdf(cdf4, combined, color=color, alpha=alpha) 

        for m, color, marker, label, linestyle in zip([(1,0,0), (0,1,0), (0,0,1), (6,2,1)], 
            ['red', 'green', 'blue', 'black'], ['^', 's', 'o', '*'], 
            [utils.dts_weights(1,0,0), utils.dts_weights(0,1,0), 
                utils.dts_weights(0,0,1), utils.dts_weights(6,2,1) + '(best)'],
                ['-','-','-','--']):
            result = results.get(m)
            if result:
                color=color
                markevery=20
                alpha = 0.6
                combined = [(x+y+z)/3.0 for x,y,z in zip(result.ctrl_overhead_percent, result.link_overhead_percent, result.table_overhead_percent)]
                utils.plotcdf(cdf1, result.table_overhead_percent, 
                    color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle, label=label) 
                utils.plotcdf(cdf2, result.link_overhead_percent, 
                    color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle, label=label)
                utils.plotcdf(cdf3, result.ctrl_overhead_percent, 
                    color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle, label=label)
                utils.plotcdf(cdf4, combined, 
                    color=color, marker=marker, markevery=15, ms=4, alpha=1, linestyle=linestyle, label=label)

        # sort results by rating
        all_results = sorted(all_results, key=lambda x: x[0])

        print("best scores:")
        for i in range(0,20):
            print(all_results[i][0], all_results[i][1].label, all_results[i][2:])
        print("worst scores:")
        for i in range(1,10):
            print(all_results[-i][0], all_results[-i][1].label, all_results[i][2:])  
        
        print("some selected scores:")
        for rating, result, _, _ in all_results:
            if result.label in ['11-2-1', '5-2-1']:
                print(rating, result.label)  

        handles, labels = cdf4.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
        #plt.subplots_adjust(top=.9)
        plt.subplots_adjust(wspace=0.2, hspace=0.2, left = 0.07, top=.87, bottom=.08)
        utils.export(fig, 'weights_dts_cdfonly.pdf', folder='weights')
        #plt.show()

    # ---------------------------------------------
    # Figure 2: all combinations of non-zero cost coefficients
    # ---------------------------------------------
    if 0:
        plt.close()
        all_results = [] 

        fig = plt.figure(figsize=(8, 10))
        fig.tight_layout(pad=0)
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((4, 3), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((4, 3), (2, 0), colspan=2)
        ax4 = plt.subplot2grid((4, 3), (3, 0), colspan=2)

        ax1.set_ylabel(r'Table overhead in \%', fontsize=12)
        ax1.set_xlabel(r'DTS experiment index sorted by table overhead', fontsize=12)
        ax2.set_ylabel(r'Link Overhead in \%', fontsize=12)
        ax2.set_xlabel(r'DTS experiment index sorted by link overhead', fontsize=12)
        ax3.set_ylabel(r'Control Overhead in \%', fontsize=12)
        ax3.set_xlabel(r'DTS experiment index sorted by control overhead', fontsize=12)
        ax4.set_ylabel(r'Weighted score', fontsize=12)
        ax4.set_xlabel(r'DTS experiment index sorted by weighted score', fontsize=12) 

        cdf1 = plt.subplot2grid((4, 3), (0, 2))
        cdf1.set_xlabel(r'Table overhead in \%', fontsize=12)
        cdf2 = plt.subplot2grid((4, 3), (1, 2))
        cdf2.set_xlabel(r'Link overhead in \%', fontsize=12)
        cdf3 = plt.subplot2grid((4, 3), (2, 2))
        cdf3.set_xlabel(r'Control overhead in \%', fontsize=12)
        cdf4 = plt.subplot2grid((4, 3), (3, 2))
        cdf4.set_xlabel(r'Weighted score in \%', fontsize=12)

        for ax in [cdf1, cdf2, cdf3, cdf4]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('CDF')
            ax.set_xlim(0,80)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)

        for ax in [ax1, ax2, ax3, ax4]:
            #ax.set_ylim(0,55)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)


        #fig, ax = plt.subplots(4,1, figsize=(6, 10))
        #fig.tight_layout(pad=2.7)




        for m, result in results.items():
            #if 0 in m: continue;
            result.label
            combined = [(x+y+z)/3.0 for x,y,z in zip(result.ctrl_overhead_percent, result.link_overhead_percent, result.table_overhead_percent)]
            color='lightgray'
            alpha = 0.1
            rating = sum(combined)
            all_results.append((rating, result))
            if SHOW_GREY:
                plotax(ax1, result.table_overhead_percent, color=color, alpha=alpha)
                plotax(ax2, result.link_overhead_percent, color=color, alpha=alpha)
                plotax(ax3, result.ctrl_overhead_percent, color=color, alpha=alpha)
                plotax(ax4, combined, color=color, alpha=alpha)

                utils.plotcdf(cdf1, result.table_overhead_percent, color=color, alpha=alpha) 
                utils.plotcdf(cdf2, result.link_overhead_percent, color=color, alpha=alpha) 
                utils.plotcdf(cdf3, result.ctrl_overhead_percent, color=color, alpha=alpha) 
                utils.plotcdf(cdf4, combined, color=color, alpha=alpha) 

        for m, color, marker, label, linestyle in zip([(1,0,0), (0,1,0), (0,0,1), (6,2,1)], 
            ['red', 'green', 'blue', 'black'], ['^', 's', 'o', '*'], 
            ['wTable=1, wLink=0, wCtrl=0 (table only)', 'wTable=0, wLink=1, wCtrl=0 (link only)', 
                'wTable=0, wLink=0, wCtrl=1 (ctrl only)', 'wTable=6, wLink=2, wCtrl=1 (best combination)'],
                ['-','-','-','--']):
            result = results.get(m)
            if result:
                color=color
                markevery=20
                alpha = 1
                combined = [(x+y+z)/3.0 for x,y,z in zip(result.ctrl_overhead_percent, result.link_overhead_percent, result.table_overhead_percent)]

                plotax(ax1, result.table_overhead_percent, 
                    color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4, label=label, linestyle=linestyle) 
                plotax(ax2, result.link_overhead_percent, 
                    color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4,  label=label, linestyle=linestyle)
                plotax(ax3, result.ctrl_overhead_percent, 
                    color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4, label=label, linestyle=linestyle)
                plotax(ax4, combined, 
                    color=color, alpha=alpha, marker=marker, markevery=markevery, ms=4, label=label, linestyle=linestyle)

                utils.plotcdf(cdf1, result.table_overhead_percent, 
                    color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle) 
                utils.plotcdf(cdf2, result.link_overhead_percent, 
                    color=color, marker=marker, markevery=15, ms=4, alpha=alpha, linestyle=linestyle) 
                utils.plotcdf(cdf3, result.ctrl_overhead_percent, 
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
        
        print("some selected scores:")
        for rating, result in all_results:
            if result.label in ['11-2-1', '5-2-1']:
                print(rating, result.label)  


        handles, labels = ax4.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)

        plt.subplots_adjust(wspace=0.1, hspace=0.4, top=.9, bottom=.05)
        utils.export(fig, 'weights_dts.pdf', folder='weights')
        plt.show()
