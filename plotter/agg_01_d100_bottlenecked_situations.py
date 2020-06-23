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


import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
matplotlib.rcParams.update(params)

from . import agg_2_utils as utils


logger = logging.getLogger(__name__)


def plot(blob, **kwargs):
    """
    Plot scenario overview for dataset z1
    """
    FOLDER = 'z1-scenario'
    # this filter will select exactly one cost coefficient combination for each scenario
    FILTER = dict(
        param_dts_weight_table=8,  
        param_dts_weight_link=2,
        param_dts_weight_ctrl=1)
    DATASET_NAME = 'z1'
    MAX_PAGES = 15
    COLS = 12
    ROWS = 6
    RSA_INCLUDE_TOPO = False
    FIGSIZE = (10, 14)

    utils.EXPORT_BLOB = blob
  
    includes = ['scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch']

    # 15 is the maximum number if switches in current experiments; (id 0-14)
    for switch_cnt in range(0, 15):
        includes.append('dts_%d_ctrl_overhead_percent' % (switch_cnt))
        includes.append('dts_%d_link_overhead_percent' % (switch_cnt))
        includes.append('dts_%d_table_overhead_percent' % (switch_cnt))
        includes.append('dts_%d_underutil_percent' % (switch_cnt))

        includes.append('dts_%d_ctrl_overhead' % switch_cnt)
        includes.append('dts_%d_link_overhead' % switch_cnt)
        includes.append('dts_%d_table_overhead ' % switch_cnt)  

        includes.append('dts_%d_table_datax' % (switch_cnt))
        includes.append('dts_%d_table_datay_raw' % (switch_cnt))
        includes.append('dts_%d_table_datay' % (switch_cnt))
        includes.append('dts_%d_table_datay_shared' % (switch_cnt))

    includes += blob.find_columns('hit_timelimit')
    blob.include_parameters(**dict.fromkeys(includes, 1))
    runs = blob.filter(**dict())  

    timelimit = 0
    seeds = []
    use_runs = []
    dts_runs = []

    # -----------------------
    # prepare data for plotting
    # -----------------------
    for run in runs:
        seed = run.get('param_topo_seed')
        if run.get('hit_timelimit') and run.get('hit_timelimit')  > 0:
            timelimit += 1
            continue

        if not seed in seeds:

            thresh = run.get('scenario_table_capacity')
            switches = run.get('scenario_switch_cnt')
            to_delegate_sum = [0]
            for switch_cnt in range(0, switches):
                d1 = 'dts_%d_table_datax' % (switch_cnt)
                d2 = 'dts_%d_table_datay_raw' % (switch_cnt)
                d3 = 'dts_%d_table_datay' % (switch_cnt)

                data = run.get(d2, [])
                if len(data) > 0:
                    to_delegate = sum([x-thresh for x in data if x > thresh])
                    to_delegate_sum.append(to_delegate)

                    # prepare data for plotting dts runs only
                    if to_delegate > 0:
                        dts_run = dict(
                            seed=seed,
                            switch=switch_cnt,
                            to_delegate=to_delegate,
                            thresh = thresh,
                            d1 = run.get(d1),
                            d2 = run.get(d2),
                            d3 = run.get(d3)
                        )
                        dts_runs.append(dts_run)


            run['to_delegate_sum'] = sum(to_delegate_sum)
            run['to_delegate_max'] = max(to_delegate_sum)


            use_runs.append(run)
            seeds.append(seed)
            
    print("runs: %s" % len(use_runs))
    print("dts_runs: %d" % (len(dts_runs)))
    print("timelimit: %d" % timelimit)
    #print("seeds: %s" % str(sorted(seeds)))

    # -----------------------
    # Figure 1: print only the dts runs (z1_dts_xxx)
    # -----------------------
    plt.close()
    fig, axes = plt.subplots(COLS,ROWS, figsize=FIGSIZE, sharex=False, sharey=False)
    fig.tight_layout(pad=0)
    axcnt = -1
    pos_start = 1
    pos_end = 0
    current_page = 1

    for run in sorted(dts_runs, key=lambda x: x.get('to_delegate'), reverse=True):

        axcnt += 1
        if axcnt >= ROWS*COLS:
            #plt.subplots_adjust(wspace=0, hspace=0)
            for r in range(axcnt, ROWS*COLS):
                ax = fig.axes[r]
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

            fig.suptitle("Bottleneck Situations (%d-%d)" % ( 
                pos_start, pos_end), fontsize=16)   

            fig.subplots_adjust(top=0.93, right=0.97)

            filename = "z1_dts_page_%.3d.pdf" % (current_page)
            utils.export(fig, filename, folder=FOLDER)

            #plt.show()
            #exit(1)

            if current_page >= MAX_PAGES:
                break;
            else:
                pos_start = pos_end+1
                current_page += 1
                plt.close()
                fig, axes = plt.subplots(COLS,ROWS, figsize=(10, 14), sharex=False, sharey=False)
                fig.tight_layout(pad=0)
                axcnt = 0

        ax = fig.axes[axcnt]
        pos_end += 1
        thresh = run.get('thresh')
        datax = run.get('d1')
        raw_util = run.get('d2')

        # plot threshold
        t1 = ax.hlines(thresh, 0, 400, color='blue', 
            label="Flow table capacity", linestyle='--', linewidth=1)

        ax.text(350, int(thresh-(0.1*thresh)), '%d' % (thresh), 
            fontsize=9, color='blue',
            verticalalignment='top', horizontalalignment='left', 
            alpha=1,
            #bbox=dict(boxstyle='square,pad=0.2',facecolor=background, edgecolor='black', alpha=1)
        )

        ax.plot(datax, raw_util, color='black', linewidth=0.5)
        
        fill_overutil = [True if x > thresh else False for x in raw_util]
        ax.fill_between(datax, raw_util, [thresh]*len(raw_util),
            where=fill_overutil, interpolate=True, color='red', alpha=0.5, label='Utilization with flow delegation')

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(-60,400)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        background = 'white'
        fontc = 'black'
        ax.text(-0.05, 0.95, '%d.%d' % (run.get('seed'), run.get('switch')), 
            fontsize=9, color=fontc,
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, alpha=1,
            bbox=dict(boxstyle='square,pad=0.2',facecolor=background, edgecolor='black', alpha=1)
        )  

    # print the last page       
    if not current_page >= MAX_PAGES:   
        for r in range(axcnt, ROWS*COLS):
            ax = fig.axes[r]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)  
        fig.suptitle("Bottleneck Situations (%d-%d)" % ( 
            pos_start, pos_end), fontsize=16)        
        fig.subplots_adjust(top=0.93, right=0.97)
        filename = "z1_dts_page_%.3d.pdf" % (current_page)
        utils.export(fig, filename, folder=FOLDER)


    # -----------------------
    # Figure 2: print the whole scenario with all switches
    # -----------------------

    configs = [
        dict(include_topo=True, show_green=True, figure_prefix='%s_rsa_with_topo_' % DATASET_NAME),
        dict(include_topo=False, show_green=True, figure_prefix='%s_rsa_' % DATASET_NAME),
    ]
    for config in [configs[1]]:   
        pos_start = 1
        pos_end = 0      
        plt.close()
        fig, axes = plt.subplots(COLS,ROWS, figsize=FIGSIZE, sharex=False, sharey=False)
        fig.tight_layout(pad=0)
        axcnt = 0
        current_page = 1

        for run in sorted(use_runs, key=lambda x: x.get('to_delegate_sum'), reverse=True):
            switches = run.get('scenario_switch_cnt')
            thresh = run.get('scenario_table_capacity')
            add_cnt = 0
            if config.get('include_topo'):
                add_cnt = 1
            if axcnt + switches + add_cnt > ROWS*COLS:
                #plt.subplots_adjust(wspace=0, hspace=0)

                for r in range(axcnt, ROWS*COLS):
                    ax = fig.axes[r]
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                fig.suptitle("RTS runs in dataset %s (%d-%d of %d)" % (DATASET_NAME, 
                    pos_start, pos_end, len(use_runs)), fontsize=16)
                fig.subplots_adjust(top=0.93, right=0.97)

                filename = "%spage_%.3d.pdf" % (config.get('figure_prefix'), current_page)
                utils.export(fig, filename, folder=FOLDER)

                #plt.show()
                #exit(1)

                if current_page >= MAX_PAGES:
                    break;
                else:
                    pos_start = pos_end+1
                    current_page += 1
                    plt.close()
                    fig, axes = plt.subplots(COLS,ROWS, figsize=(10, 14), sharex=False, sharey=False)
                    fig.tight_layout(pad=0)
                    axcnt = 0

            # plot topo
            if config.get('include_topo'):
                ax = fig.axes[axcnt]
                hosts_of_switch = {}
                edges = run.get('scenario_edges')
                for k, v in run.get('scenario_hosts_of_switch').items():
                    hosts_of_switch[int(k)] = v
                plt_switches = list(range(0, run.get('scenario_switch_cnt')))
                concentrated_switches= run.get('scenario_concentrated_switches', [])
                utils.plot_topo_small(ax, hosts_of_switch, edges, plt_switches , concentrated_switches)
                axcnt += 1


            # plot all switches
            switch_axes = []
            maxy = 0
            pos_end += 1
            for switch_cnt in range(0, switches):
                ax = fig.axes[axcnt]
                switch_axes.append(ax)

                d1 = 'dts_%d_table_datax' % (switch_cnt)
                d2 = 'dts_%d_table_datay_raw' % (switch_cnt)
                d3 = 'dts_%d_table_datay' % (switch_cnt)
                d4 = 'dts_%d_table_datay_shared' % (switch_cnt)

                axcnt += 1

                datax = run.get(d1)
                if run.get(d2):
                    raw_util = run.get(d2)
                else:
                    raw_util = [0]*len(datax)
                if  max(raw_util) > maxy:
                    maxy = max(raw_util)   

                # plot threshold
                t1 = ax.hlines(thresh, 0, 400, color='blue', 
                    label="Flow table capacity", linestyle='--', linewidth=1)

                ax.text(350, int(thresh-(0.1*thresh)), '%d' % (thresh), 
                    fontsize=9, color='blue',
                    verticalalignment='top', horizontalalignment='left', 
                    alpha=1,
                    #bbox=dict(boxstyle='square,pad=0.2',facecolor=background, edgecolor='black', alpha=1)
                )

                ax.plot(datax, raw_util, color='black', linewidth=0.5)
                
                fill_overutil = [True if x > thresh else False for x in raw_util]
                ax.fill_between(datax, raw_util, [thresh]*len(raw_util),
                    where=fill_overutil, interpolate=True, color='red', alpha=0.5, label='Utilization with flow delegation')


                ax.plot(run.get(d1), run.get(d3), color='green', linestyle='-', linewidth=1)

                fill_shared = [True if x < thresh else False for x in raw_util]
                ax.fill_between(datax, run.get(d3), raw_util,
                    where=fill_shared, interpolate=True, color='green', alpha=0.5, label='Utilization with flow delegation')

                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_xlim(-60,400)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                #numpy.random.seed(run.get('param_topo_seed')) 
                #ax.patch.set_facecolor(numpy.random.rand(3,))
                #ax.patch.set_alpha(0.05)

                background = 'white'
                fontc = 'black'
                if switch_cnt == 0:
                    background = 'black'
                    fontc = 'white'
                ax.text(-0.05, 0.95, '%d.%d' % (run.get('param_topo_seed'), switch_cnt), 
                    fontsize=9, color=fontc,
                    verticalalignment='top', horizontalalignment='left', 
                    transform=ax.transAxes, alpha=1,
                    bbox=dict(boxstyle='square,pad=0.2',facecolor=background, edgecolor='black', alpha=1)
                )
            for sax in switch_axes:
                sax.set_ylim(0, maxy)
        
        # print last page         
        if current_page < MAX_PAGES:
            for r in range(axcnt, ROWS*COLS):
                ax = fig.axes[r]
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
            fig.suptitle("RTS runs in dataset %s (%d-%d of %d)" % (DATASET_NAME, 
                pos_start, pos_end, len(use_runs)), fontsize=16)
            fig.subplots_adjust(top=0.93, right=0.97)
            filename = "%spage_%.3d.pdf" % (config.get('figure_prefix'), current_page)
            utils.export(fig, filename, folder=FOLDER)





if __name__ == '__main__':
    utils.run_main()