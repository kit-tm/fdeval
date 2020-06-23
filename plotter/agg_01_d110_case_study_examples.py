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

FIGSIZE = (14, 10)
XOFFSET = 60
THREHSOLDS_INCLUDED = [30,40,50,60,70,80,90]
THREHSOLDS_INCLUDED_RSA = [75,80,85,90]
THRESHOLDS_RSA_SINGLE = [40,50,60,70, 100-13, 100-37, 100-44, 100-48]

THRESHOLD_BY_SEED = {
    155603: [45,50,55,60],
    84812: [55,60,65,70]
}
SKIP_SEEDS = [321241]
USE_SEEDS = [486862, 197614, 231117,23584,272906,276798,309445,325146,32520,353999,412327]
SHOW_TOPO = False
USE_SEEDS = [139,10946,12815,14350,16104,18519,22067,23584,30750,32520,39421,51865,53665,57688,60781,63576,74420,74734,74850,82853,84812,87724,90535,91401,93422,99063,102162,102202,105622,105921,107634,120097,120332,124089,124288,128393,129083,136505,137521,150184,152851,155603,158340,160405,160436,160788,166890,181181,183962,184361,190743,196423,197614,203316,213521,214901,215645,227656,229004,231117,239799,246723,250485,251384,257009,260493,263296,272906,276798,291093,291989,298293,299376,299773,305667,308250,309445,325146,334391,335801,341369,352994,353999,357032,366403,368376,368657,370555,378144,383868,387028,412327,414493,416861,424466,428815,451837,470310,486862,498437]
#USE_SEEDS = [136505, 155603, 166890] + [22067, 74420, 120332,239799,152851]
USE_SEEDS = [136505, 155603, 84812, 166890] 
#USE_SEEDS = [136505]
#23584 # avg
#166890 # worst
#136505 # best

def plot_scenario_dts_multi_reduction(grid, fig, runs, switch, showX = False, alternate_rows=False, title_row=False):
    assert(len(runs) > 0)
    columns = 6
    width_ratios=[1,1,1,2,2,2]
    subgrid = grid.subgridspec(1, columns, wspace=0, hspace=0, width_ratios=width_ratios)
    axes = []
    for column in range(0, columns):
        try:
            axes.append(fig.add_subplot(subgrid[0,column], sharey=axes[0]))
            ax = axes[-1]
            ax.get_yaxis().set_visible(False)
            if not showX:
                ax.get_xaxis().set_visible(False)   
        except:
            axes.append(fig.add_subplot(subgrid[0,column]))

        ax = axes[-1]
        for _, spine in ax.spines.items():
            spine.set_color('lightgray')
            spine.set_linewidth(4)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if alternate_rows:
            ax.patch.set_facecolor('lightgray')
            ax.patch.set_alpha(0.2)
        else:
            ax.patch.set_facecolor('lightgray')
            ax.patch.set_alpha(0.4)         

    for ax in axes[0:1]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 

    # ------------- left-most column (recduction in percent)
    ax = axes[0]
    ax.text(0.5, 0.5, '-%d' % (100-runs[0].get('param_topo_switch_capacity_overwrite')) + '\\%', transform=ax.transAxes,
        fontsize=18, fontweight='normal', color="black", va='center', ha="center", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    # ------------- second column (overutilization)
    ax = axes[1]
    if title_row:
        ax.text(0.5, 1.1, 'Over-\nutilization', transform=ax.transAxes,
            fontsize=15, fontweight='normal', color="black", va='bottom', ha="center", 
        )
    metric_table_overhead = ''
    for algo in [1,2,3]:
        use_run = None
        for run in runs:
            if run.get('param_dts_algo') == algo:
                use_run = run
        if use_run:
            metric_table_overhead += (r'\noindent \vspace{2mm} [%d]=' % algo) + r'%.2f' % use_run.get('dts_%d_overutil_percent' % switch) + r' \\'
        else:
            metric_table_overhead += (r'\noindent \vspace{2mm} [%d]= N/A' % algo) + r' \\'

    ax.text(0.5, 0.5, metric_table_overhead, transform=ax.transAxes,
        fontsize=18, fontweight='normal', color="black", va='center', ha="center", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    # ------------- third column (underutil)
    ax = axes[2]
    if title_row:
        ax.text(0.5, 1.1, 'Under-\nutilization', transform=ax.transAxes,
            fontsize=15, fontweight='normal', color="black", va='bottom', ha="center", 
        )
    metric_table_overhead = ''
    for algo in [1,2,3]:
        use_run = None
        for run in runs:
            if run.get('param_dts_algo') == algo:
                use_run = run
        if use_run:
            metric_table_overhead += (r'\noindent \vspace{2mm} [%d]=' % algo) + r'%.2f' % use_run.get('dts_%d_underutil_percent' % switch) + r' \\'
        else:
            metric_table_overhead += (r'\noindent \vspace{2mm} [%d]= N/A' % algo) + r' \\'

    #for run in sorted(runs, key=lambda x: x.get('param_dts_algo')):
    #    algo = run.get('param_dts_algo')
    #    metric_table_overhead += (r'\noindent \vspace{2mm} [%d]=' % algo) + r'%.2f' % run.get('dts_%d_underutil_percent' % switch) + r' \\'
    ax.text(0.5, 0.5, metric_table_overhead, transform=ax.transAxes,
        fontsize=18, fontweight='normal', color="black", va='center', ha="center", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    # ------------- fourth to last column (normal switches)
    maxy = 0
    axcnt = 3
    for algo in [1,2,3]:
        use_run = None
        for run in runs:
            if run.get('param_dts_algo') == algo:
                use_run = run
        run = use_run

        ax = axes[axcnt]

        if run:
            thresh = run.get('scenario_table_capacity')

            d1 = 'dts_%d_table_datax' % (switch)
            d2 = 'dts_%d_table_datay_raw' % (switch)
            d3 = 'dts_%d_table_datay' % (switch)
            d4 = 'dts_%d_table_datay_shared' % (switch)

            datax = run.get(d1)
            if run.get(d2):
                raw_util = run.get(d2)
            else:
                raw_util = [0]*len(datax)
            if  max(raw_util) > maxy:
                maxy = max(raw_util)   

            # plot threshold
            t1 = ax.hlines(thresh, -1*XOFFSET, 400, color='blue', 
                label="Flow table capacity", linestyle='--', linewidth=1)

            if axcnt == columns-1:
                ax.text(410, thresh, '%d' % (thresh), 
                    fontsize=9, color='blue',
                    verticalalignment='top', horizontalalignment='left', 
                    alpha=1,
                    bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='blue', alpha=1)
                )

            # red colored utilization over threshold
            fill_overutil = [True if x > thresh else False for x in raw_util]
            ax.fill_between(datax, raw_util, run.get(d3),
                where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
                label='Rules relocated')

            ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
                interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')

            ax.plot(list(range(-1*XOFFSET,0)) + run.get(d1), [0]*XOFFSET + run.get(d3), color='black', linestyle='-', linewidth=0.75)
        else:
            ax.text(0.5, 0.5, 'Timeout', transform=ax.transAxes,
                fontsize=20, fontweight='normal', color="darkgray", va='center', ha="center", 
            )

        axcnt += 1
    
    # adjust height
    for ax in axes[3:]:
        ax.set_xlim(-1*XOFFSET,400)
        ax.set_xticks([0,100,200,300])
        ax.set_ylim(-500,maxy+500)

    # titles
    if title_row:
        ax = axes[0]
        ax.text(0.5, 1.1, 'Capacity\nreduction', transform=ax.transAxes,
            fontsize=15, fontweight='normal', color="black", va='bottom', ha="center", 
        ) 
        for ax, algo in zip(axes[3:6], ['Select-Opt [1]', 'Select-CopyFirst [2]', 'Select-Greedy [3]']):   
            ax.text(0.5, 1.1, algo, transform=ax.transAxes,
                fontsize=18, fontweight='normal', color="black", va='bottom', ha="center", 
            )


    return axes

def plot_single_desc(ax, run, switch, **kwargs):

    thresh = run.get('scenario_table_capacity')
    d2 = 'dts_%d_table_datay_raw' % (switch)
    raw_util = run.get(d2)
    fill_overutil = [1 if x > thresh else 0 for x in raw_util]

    new_table_overhead_percent = (run.get('dts_%d_table_overhead' % switch) / float(sum(fill_overutil)))
    new_link_overhead = ((run.get('dts_%d_link_overhead' % switch) / 1000000) / float(sum(fill_overutil)))
    new_ctrl_overhead = ((run.get('dts_%d_ctrl_overhead' % switch)) / float(sum(fill_overutil)))


    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False) 

    algo = run.get('param_dts_algo')
    if algo == 1:
        algo = 'Select-Opt'
    if algo == 2:
        algo = 'Select-CopyFirst'
    if algo == 3:
        algo = 'Select-Greedy'

    textstr = '\\noindent \\textbf{Statistics}\\\\ \\\\'

    textstr += 'Capacity reduction: %d\\%% \\\\' % (100-run.get('param_topo_switch_capacity_overwrite'))

    textstr += 'Time slots above capacity: %d \\\\' % sum(fill_overutil)

    textstr += 'DTS Alg.: %s \\\\' % algo

    #textstr += 'DTS samples: %d \\\\' % run.get('dts_%d_solver_cnt_feasable' % switch)

    textstr += 'Delegation templates: %d \\\\' % run.get('dts_%d_solver_considered_ports' % switch)

    textstr += 'Overutilization: %.2f\\%% \\\\' % (run.get('dts_%d_overutil_percent' % switch))

    textstr += 'Underutilization: %.2f\\%% \\\\' % (run.get('dts_%d_underutil_percent' % switch))

    textstr += '\\\\ \\textbf{Overhead}\\\\ \\\\'

    textstr += 'Table: %d (%.2f) \\\\' % (run.get('dts_%d_table_overhead' % switch), new_table_overhead_percent)
    
    #textstr += 'Table2: %d (%.2f) \\\\' % (run.get('dts_%d_table_overhead_max' % switch), run.get('dts_%d_table_overhead_percent' % switch))
    
    textstr += 'Link: %.2f Mbit (%.2f) \\\\' % (run.get('dts_%d_link_overhead' % switch)/1000000, new_link_overhead)

    textstr += 'Control: %d (%.2f) \\\\' % (run.get('dts_%d_ctrl_overhead' % switch), new_ctrl_overhead)

 


    ax.text(-0.2, 1, textstr, transform=ax.transAxes,
        fontsize=18, fontweight='normal', color="black", va='top', ha="left", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    ax.text(-0.2, -0.05, 'Scenario id: %d (switch %d)\nCapacity without flow delegation: %d' % (run.get('param_topo_seed'),
        switch, run.get('scenario_table_util_max_total')), transform=ax.transAxes,
        fontsize=14, fontweight='normal', color="darkgray", va='bottom', ha="left", 
    )

def plot_single_dts(ax, run, switch, **kwargs):


    thresh = run.get('scenario_table_capacity')

    d1 = 'dts_%d_table_datax' % (switch)
    d2 = 'dts_%d_table_datay_raw' % (switch)
    d3 = 'dts_%d_table_datay' % (switch)
    d4 = 'dts_%d_table_datay_shared' % (switch)

    datax = run.get(d1)
    raw_util = run.get(d2)

    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_ylim(0,max(raw_util)*1.5)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.set_ylabel('Flow table utilization', fontsize=15)

    # plot threshold
    t1 = ax.hlines(thresh, -1*XOFFSET, 400, color='blue', 
        label="Flow table capacity", linestyle='--', linewidth=1)

    ax.text(400, thresh, '%d' % (thresh), 
        fontsize=12, color='blue',
        verticalalignment='top', horizontalalignment='left', 
        alpha=1,
        bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='blue', alpha=1)
    )

    # red colored utilization over threshold
    fill_overutil = [True if x > thresh else False for x in raw_util]
    ax.fill_between(datax, raw_util, run.get(d3),
        where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
        label='Rules relocated')

    ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
        interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')

    ax.plot(list(range(-1*XOFFSET,0)) + run.get(d1), [0]*XOFFSET + run.get(d3), color='black', linestyle='-', linewidth=0.75)

    ax.legend(loc='upper left', fontsize=14)
    return





    v1 = '%.2f' % float(run.get('metrics_ds_underutil_percent')) + r'\%'
    v2 = '%.2f' % float(run.get('metrics_demand_delegated_percent')) + r'\%'
    v3 = '%.2f' % float(run.get('metrics_ds_overhead_percent')) + r'\%'
    v4 = '%.2f' % float(run.get('metrics_ds_backdelegations_percent')) + r'\%'
    #metrics_ds_overhead_percent

    params = '%s : %s \n' % (metrics[1], v1)
    params +=  '%s : %s\n' % (metrics[2], v2)
    params += '%s : %s\n' % (metrics[3], v3)
    params += '%s : %s' % (metrics[4], v4)

    cnt_active_flows = run.get('metrics_ds_flowtable_cnt_active_flows')
    cnt_active_flows_total = run.get('metrics_ds_flowtable_cnt_active_flows_total')
    cnt_active_flows_evicted = run.get('metrics_ds_flowtable_cnt_active_flows_evicted')
    cnt_ports_delegated = run.get('metrics_ds_flowtable_cnt_ports_delegated')
    threshold = int(run.get("param_topo_switch_capacity"))

    ax.text(0.01, 0.97, r''+params, fontsize=12, 
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, color='black',
        bbox=dict(facecolor='white', edgecolor='white'))

    ds, = ax.plot(np.arange(len(cnt_active_flows)), cnt_active_flows, 
        color="black", linewidth=1.5)

    ax.fill_between(np.arange(len(cnt_active_flows)), cnt_active_flows, 0, 
        interpolate=True, color='orange', alpha=0.3)

    total, = ax.plot(np.arange(len(cnt_active_flows_total)), cnt_active_flows_total, 
        linestyle=':', color='blue', label='Utilization without flow delegation', linewidth=1)



    fill_overutil = [True if x > threshold else False for x in cnt_active_flows]
    ax.fill_between(np.arange(len(cnt_active_flows)), cnt_active_flows, [threshold]*len(cnt_active_flows),
        where=fill_overutil, interpolate=True, color='orange', alpha=0.3, label='Utilization with flow delegation')
    
    fill_underutil = [True if x < threshold and x+e > threshold else False for x, e in zip(cnt_active_flows, cnt_active_flows_evicted)]
    ax.fill_between(np.arange(len(cnt_active_flows)), cnt_active_flows, [threshold]*len(cnt_active_flows), 
        where=fill_underutil, interpolate=True, color='red', alpha=1, label='Underutilization')


    # plot threshold
    t1 = ax.hlines(threshold, 0, len(cnt_active_flows), color='black', 
        label="Flow table capacity", linestyle='--', linewidth=1.5)

    # legend
    red = patches.Patch(color='red',  alpha=0.3, label='The red data')
    green = patches.Patch(color='red',  alpha=1, label='The red data')
    #orange = patches.Patch(color='orange',  alpha=0.5, label='The red data')
    ax.legend(loc=1, ncol=2)
    
    for x in range(50,450,50):
        ax.vlines(x, 0, 1150, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax2.vlines(x, 0, 22, color='grey', linestyle='--', linewidth=1, alpha=0.3)

    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_ylim(0,1150)
    ax.set_xlim(0,450)
    ax.set_title('%s ~~~ %s' % (
        utils.label_dts_solver(run), utils.label_dts_objectives_all(run)), fontsize=17,y=0.95)
    ax.set_ylabel(r'\#rules in flow table', fontsize=14)
    ax.xaxis.set_ticks_position('none') 
    ax.set_xticks([], [])
    ax.set_yticks((250,500,750,1000))
    ax.set_yticklabels(('250','500','750','1000'))

    #ax.set_xlabel(r'\textbf{time (s)}')

    # overhead
    ax2.plot(np.arange(len(cnt_ports_delegated)), cnt_ports_delegated, 
        linestyle=':', color='red', linewidth=1)
    fill_overhead = [x > 0 for x in cnt_ports_delegated]
    ax2.fill_between(np.arange(len(cnt_ports_delegated)), cnt_ports_delegated, [0]*len(cnt_active_flows), 
        where=fill_overhead, interpolate=True, color='red', alpha=0.1)

    ax2.text(0.01, -0.08, r'Number of selected delegation templates (out of 20):', fontsize=12, 
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, color='black',
        bbox=dict(facecolor='white', edgecolor='white'))

    ax2.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_xlabel('time (s)', fontsize=14)
    #ax2.set_ylabel(r'\#templates', fontsize=14)
    ax2.set_ylim(0,22)
    ax2.set_xlim(0,450)

    #ax2.tick_params(axis='both', which='major', labelsize=14)

    #plt.legend()

    # kwargs.get('exportdir')

    filename = "%d_%s_%s.pdf" % (threshold, utils.name_dts_solver(run), utils.name_dts_weights(run))
    utils.export(fig, filename, folder='dts_functional_001')
    #plt.show()
    plt.close()

def plot_single_rsa(run, ratio_by_seed, hide_topo=False, use_all=False):

    rowcnt = 0
    axes = []
    axcnt = 0
    maxy = 0


    use_switches = []
    for switch in range(0, run.get('scenario_switch_cnt')):
        if use_all:
            use_switches.append(switch) 
        else:    
            d2 = 'dts_%d_table_datay_raw' % (switch)
            d3 = 'dts_%d_table_datay' % (switch)
            d4 = 'dts_%d_table_datay_shared' % (switch)
            thresh = run.get('scenario_table_capacity')
            raw_util = run.get(d2)
            test_raw = [1 if x > thresh else 0 for x in raw_util]
            test_shared = [abs(x1-x2) for x1, x2 in zip(raw_util, run.get(d3))]
            if sum(test_shared) > 0 or sum(test_raw) > 0:
                use_switches.append(switch) 


    # +1 for topology plot in the top left
    x=9999 # used with LAYOUTS; topology is placed here
    all_axes = []
    layout = LAYOUTS.get(len(use_switches)+1)
    cols = len(layout[0])
    rows = len(layout)
    figsize = (14,6)
    constrained_layout = True
    fig = plt.figure(constrained_layout=constrained_layout, figsize=figsize)
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

    # some cases require a plot without the topology
    if hide_topo:
        rows = -1
        cols = -1
        figsize = (12,9)  
        adjustheight = 9
        if len(use_switches)+1 == 4:
            rows = 1
            cols = 4
            figsize = (12,4) 
            adjustheight=3
        if len(use_switches)+1 == 11:
            rows = 3
            cols = 4  
            figsize = (12,9) 
            adjustheight=9
        if len(use_switches)+1 == 14:
            rows = 3
            cols = 5     
            figsize = (12,9) 
            adjustheight=12
        if rows > 0:
            all_axes = [None]
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(rows, cols, figure=fig)   
            cnt = 0
            for y in range(rows):
                for x in range(cols):
                    cnt += 1
                    if cnt <= len(use_switches)+1:
                        all_axes.append(plt.subplot(gs.new_subplotspec((y, x))))



    plotted_topo = False
    for idx, switch in enumerate(use_switches):
        #try:
        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt], sharey=axes[0])
        #except IndexError:
        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt])

        ax = all_axes[idx+1]
        axes.append(ax)
        
        ax.set_xlim(-1*XOFFSET, 400)
        

        thresh = run.get('scenario_table_capacity')

        d1 = 'dts_%d_table_datax' % (switch)
        d2 = 'dts_%d_table_datay_raw' % (switch)
        d3 = 'dts_%d_table_datay' % (switch)
        d4 = 'dts_%d_table_datay_shared' % (switch)

        datax = run.get(d1)
        if run.get(d2):
            raw_util = run.get(d2)
        else:
            raw_util = [0]*len(datax)
        if  max(raw_util) > maxy:
            maxy = max(raw_util)   

        # plot threshold
        t1 = ax.hlines(thresh, -1*XOFFSET, 400, color='blue', 
            label="Flow table capacity", linestyle='--', linewidth=1)

        if idx == len(use_switches)-1:
            ax.text(400, thresh, '%d' % (thresh), 
                fontsize=12, color='blue',
                verticalalignment='top', horizontalalignment='left', 
                alpha=1,
                bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='blue', alpha=1)
            )

        # red colored utilization over threshold
        fill_overutil = [True if x > thresh else False for x in raw_util]
        ax.fill_between(datax, raw_util, run.get(d3),
            where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
            label='Rules relocated')

        circled_number = str(switch)
        circled_color = 'black'
        ax.text(0.5, .95, circled_number, fontsize=14, 
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes, color='white', alpha=1,
            bbox=dict(boxstyle='circle', facecolor=circled_color, edgecolor='black')
        )

        ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
            interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')

        ax.plot(list(range(-1*XOFFSET,0)) + run.get(d1), [0]*XOFFSET + run.get(d3), color='black', linestyle='-', linewidth=0.75)

        fill_shared = [True if x < thresh else False for x in raw_util]
        ax.fill_between(datax, run.get(d3), raw_util,
            where=fill_shared, interpolate=True, color='green', alpha=0.5, 
            label='Rules stored in remote switch')


        # ----------------- plot topology if not yet done (first ax)
        if not hide_topo and not plotted_topo:
            plotted_topo = True

            #ax = fig.add_subplot(maingrid[rowcnt+1,axcnt])
            scenario_concentrated_switches = run.get('scenario_concentrated_switches')
            hosts_of_switch = {}
            edges = run.get('scenario_edges')
            for k, v in run.get('scenario_hosts_of_switch').items():
                hosts_of_switch[int(k)] = v
            plt_switches = list(range(0, run.get('scenario_switch_cnt')))
            utils.plot_topo_small(all_axes[0], hosts_of_switch, edges, plt_switches , scenario_concentrated_switches,
                switch_node_size=250, font_size=15)

            # ratio 
            all_axes[0].text(0.1, -0.1, 'Scenario id for reproducibility: %d\nFlow table utilization ratio: %.2f\\%%' % (run.get('param_topo_seed'),
                ratio_by_seed.get(run.get('param_topo_seed'))), transform=all_axes[0].transAxes,
                fontsize=14, fontweight='normal', color="darkgray", va='bottom', ha="left", 
            )



        axcnt += 1


    # -------------- virtual backup switch
    ax = all_axes[-1]
    percent_relocated = run.get('rsa_table_percent_relocated') / (run.get('rsa_table_percent_relocated')  + run.get('rsa_table_percent_shared')) * 100
    backup = run.get('rsa_table_backup_switch_util_over_time')   
    datax = list(range(-1*XOFFSET, len(backup)))
    ax.fill_between(datax, [0]*len(datax), [0]*XOFFSET + backup, 
        interpolate=True, color='red', alpha=0.5, label='Rules in virtual backup switch')
    ax.plot(datax, [0]*XOFFSET + backup, color="red", linestyle=':', linewidth=0.5)
    ax.set_xlim(-1*XOFFSET,400)
    ax.set_xticks([0,100,200,300,400])
    if maxy < max(backup):
        ax.set_ylim(-500,max(backup)+500)
    else:
        ax.set_ylim(-500,maxy+500)
    ax.text(1, 0.76, ('\\noindent %.2f\\%%\\\\(%.2f\\%%)' %  (percent_relocated, run.get('rsa_table_percent_relocated'))), transform=ax.transAxes,
        fontsize=15, fontweight='normal', color="red", va='bottom', ha="right", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )
    if percent_relocated < 0.001:
        ax.text(0.5, 0.5, 'Not used', transform=ax.transAxes,
            fontsize=20, fontweight='bold', color="darkgray", va='center', ha="center", 
        )

    axes.append(ax)
    circled_color = 'black'
    xalign = 0.5
    if len(use_switches) > 8:
        xalign = 0.3
    ax.text(xalign, .95, 'BS', fontsize=14, 
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, color='white', alpha=1,
        bbox=dict(boxstyle='round', facecolor=circled_color, edgecolor='black')
    )

    rowcnt += 1

    for ax in axes:
        ax.set_ylim(-120, maxy+500)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axes[-4:]:
        ax.set_xlabel('Time (s)')
    for ax in []:
        ax.set_ylabel('Flow table utilization')

    fig.suptitle('Capacity reduction factor: %d\\%%' % (100-run.get('param_topo_switch_capacity_overwrite')), fontsize=18)
    handles, labels = axes[1].get_legend_handles_labels()

    if hide_topo:
        # ratio 
        all_axes[1].text(0.02, 1.02, 'Scenario id for reproducibility: %d\nFlow table utilization ratio: %.2f\\%%' % (run.get('param_topo_seed'),
            ratio_by_seed.get(run.get('param_topo_seed'))), transform=all_axes[1].transAxes,
            fontsize=14, fontweight='normal', color="darkgray", va='bottom', ha="left", 
        )
        fig.legend(handles, labels, loc='lower right', ncol=2, fontsize=14)
    
        if adjustheight == 3:
            fig.subplots_adjust(top=0.8, bottom=0.3, left=0.04, right=0.98,)  
        else: 
            fig.subplots_adjust(top=0.93, bottom=0.13, left=0.04, right=0.98,)   
    else:     
        fig.legend(handles, labels, loc='upper left', ncol=1, fontsize=14)
    #fig.subplots_adjust(top=0.9, left=0, right=0.98, wspace=0.5, hspace=0.5) # padding top
    return fig

def plot_single_rsa_for_slides(run, ratio_by_seed):

    rowcnt = 0
    axes = []
    axcnt = 0
    maxy = 0
    all_axes = []

    use_switches = []
    for switch in range(0, run.get('scenario_switch_cnt')):
        use_switches.append(switch) 

    rows = -1
    cols = -1
    figsize = (12,9)  
    adjustheight = 9
    if len(use_switches)+1 == 4:
        rows = 1
        cols = 4
        figsize = (12,4) 
        adjustheight=3
    if len(use_switches)+1 == 11:
        rows = 3
        cols = 4  
        figsize = (12,9) 
        adjustheight=9
    if len(use_switches)+1 == 14:
        rows = 3
        cols = 5     
        figsize = (12,9) 
        adjustheight=12
    if rows > 0:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(rows, cols, figure=fig)   
        cnt = 0
        for y in range(rows):
            for x in range(cols):
                cnt += 1
                if cnt <= len(use_switches)+2:
                    all_axes.append(plt.subplot(gs.new_subplotspec((y, x))))

    ax = all_axes[0]                 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False) 

    ax.text(0.5, 0.6, ('%d' % (100-run.get('param_topo_switch_capacity_overwrite'))) + r'\%', fontsize=40, 
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, color='red', alpha=1,
        #bbox=dict(facecolor='white', edgecolor='black')
    )
    ax.text(0.5, 0.32, r'less', fontsize=25, 
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, color='red', alpha=1,
        #bbox=dict(facecolor='white', edgecolor='black')
    )
    ax.text(0.5, 0.15, r'capacity', fontsize=25, 
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, color='red', alpha=1,
        #bbox=dict(facecolor='white', edgecolor='black')
    )


    plotted_topo = False
    for idx, switch in enumerate(use_switches):
        #try:
        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt], sharey=axes[0])
        #except IndexError:
        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt])

        ax = all_axes[idx+1]
        axes.append(ax)
        
        ax.set_xlim(0, 250)
        

        thresh = run.get('scenario_table_capacity')

        d1 = 'dts_%d_table_datax' % (switch)
        d2 = 'dts_%d_table_datay_raw' % (switch)
        d3 = 'dts_%d_table_datay' % (switch)
        d4 = 'dts_%d_table_datay_shared' % (switch)

        datax = run.get(d1)
        if run.get(d2):
            raw_util = run.get(d2)
        else:
            raw_util = [0]*len(datax)
        if  max(raw_util) > maxy:
            maxy = max(raw_util)   

        # plot threshold
        t1 = ax.hlines(thresh, -1*XOFFSET, 400, color='blue', 
            label="Flow table capacity", linestyle='--', linewidth=1)

        if idx == len(use_switches)-1:
            ax.text(400, thresh, '%d' % (thresh), 
                fontsize=12, color='blue',
                verticalalignment='top', horizontalalignment='left', 
                alpha=1,
                bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='blue', alpha=1)
            )

        # red colored utilization over threshold
        fill_overutil = [True if x > thresh else False for x in raw_util]
        ax.fill_between(datax, raw_util, run.get(d3),
            where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
            label='Rules relocated')

        circled_number = str(switch)
        circled_color = 'black'
        ax.text(0.5, .95, circled_number, fontsize=14, 
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes, color='white', alpha=1,
            bbox=dict(boxstyle='circle', facecolor=circled_color, edgecolor='black')
        )

        ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
            interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')

        ax.plot(list(range(-1*XOFFSET,0)) + run.get(d1), [0]*XOFFSET + run.get(d3), color='black', linestyle='-', linewidth=0.75)

        fill_shared = [True if x < thresh else False for x in raw_util]
        ax.fill_between(datax, run.get(d3), raw_util,
            where=fill_shared, interpolate=True, color='green', alpha=0.5, 
            label='Rules stored in remote switch')

        axcnt += 1


    # -------------- virtual backup switch
    ax = all_axes[-1]
    percent_relocated = run.get('rsa_table_percent_relocated') / (run.get('rsa_table_percent_relocated')  + run.get('rsa_table_percent_shared')) * 100
    backup = run.get('rsa_table_backup_switch_util_over_time')   
    datax = list(range(-1*XOFFSET, len(backup)))
    ax.fill_between(datax, [0]*len(datax), [0]*XOFFSET + backup, 
        interpolate=True, color='red', alpha=0.5, label='Rules in virtual backup switch')
    ax.plot(datax, [0]*XOFFSET + backup, color="red", linestyle=':', linewidth=0.5)
    ax.set_xlim(-1*XOFFSET,400)
    ax.set_xticks([0,100,200,300,400])
    if maxy < max(backup):
        ax.set_ylim(-500,max(backup)+500)
    else:
        ax.set_ylim(-500,maxy+500)
    ax.text(1, 0.76, ('\\noindent %.2f\\%%' %  (run.get('rsa_table_percent_relocated'))), transform=ax.transAxes,
        fontsize=15, fontweight='normal', color="red", va='bottom', ha="right", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )
    if percent_relocated < 0.001:
        ax.text(0.5, 0.5, 'Not used', transform=ax.transAxes,
            fontsize=20, fontweight='bold', color="darkgray", va='center', ha="center", 
        )

    axes.append(ax)
    circled_color = 'black'
    xalign = 0.5
    if len(use_switches) > 8:
        xalign = 0.3
    ax.text(xalign, .95, 'BS', fontsize=14, 
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, color='white', alpha=1,
        bbox=dict(boxstyle='round', facecolor=circled_color, edgecolor='black')
    )

    rowcnt += 1

    for ax in axes:
        ax.set_ylim(-120, maxy+500)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axes[-4:]:
        ax.set_xlabel('Time (s)')
    for ax in []:
        ax.set_ylabel('Flow table utilization')

    #fig.suptitle('Capacity reduction factor: %d\\%%' % (100-run.get('param_topo_switch_capacity_overwrite')), fontsize=18)
    handles, labels = axes[1].get_legend_handles_labels()


    fig.legend(handles, labels, loc='lower right', ncol=4, fontsize=13)

    fig.subplots_adjust(top=0.98, bottom=0.13, left=0.04, right=0.98,)   

    return fig

def plot_scenario_raw(run, ratio_by_seed):

    rowcnt = 0
    axes = []
    axcnt = 0
    maxy = 0

    # +1 for topology plot in the top left
    x=9999 # used with LAYOUTS; topology is placed here
    all_axes = []
    layout = LAYOUTS.get(run.get('scenario_switch_cnt'))
    cols = len(layout[0])
    rows = len(layout)
    fig = plt.figure(constrained_layout=True, figsize=(14, 6))
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

    plotted_topo = False

    for switch in range(0, run.get('scenario_switch_cnt')):
        #try:
        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt], sharey=axes[0])
        #except IndexError:
        #    ax = fig.add_subplot(maingrid[rowcnt,axcnt])

        ax = all_axes[switch+1]
        axes.append(ax)
        
        ax.set_xlim(-1*XOFFSET, 400)
        

        datax = run.get('dts_%d_table_datax' % switch)
        datay = run.get('dts_%d_table_datay_raw' % switch)

        if max(datay) > maxy:
            maxy = max(datay)

        ax.plot(list(range(-1*XOFFSET,0)) + datax, [0]*XOFFSET + datay, color='black', linestyle='-', linewidth=0.75)

        ax.fill_between(datax, [0]*len(datay), datay,
            interpolate=True, color='orange', alpha=0.3, label='Rules in flow table')

        # show bottleneck parameters
        w1 = str(run.get('scenario_gen_param_topo_bottleneck_cnt'))
        w2 = str(run.get('scenario_gen_param_topo_bottleneck_duration')) + "s"
        w3 = str(run.get('scenario_gen_param_topo_bottleneck_intensity'))
        if run.get('scenario_gen_param_topo_bottleneck_cnt') == 0:
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


        # ----------------- plot topology if not yet done (first ax)
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


            # ratio 
            util_avg = run.get('scenario_table_util_avg_total')
            util_max = run.get('scenario_table_util_max_total')
            ratio = float(util_avg) / float(util_max)
            all_axes[0].text(0.1, -0.08, 'Scenario id for reproducibility: %d\nFlow table utilization ratio: %.2f\\%%' % (run.get('param_topo_seed'),
                ratio_by_seed.get(run.get('param_topo_seed'))), transform=all_axes[0].transAxes,
                fontsize=14, fontweight='normal', color="darkgray", va='bottom', ha="left", 
            )


        axcnt += 1

    rowcnt += 1

    for ax in axes:
        ax.set_ylim(-120, maxy+500)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axes[-4:]:
        ax.set_xlabel('Time (s)')
    for ax in []:
        ax.set_ylabel('Flow table utilization')

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=1, fontsize=16)
    return fig

def plot_scenario_rsa_multi_reduction_title(grid, fig, use_switches, alternate_rows=False):

    columns = len(use_switches)+3
    virtual_backup_switch_column = 2 # third column
    x = 1
    if columns > 8:
        x = 2
    width_ratios=[x,x,2] + [2]*len(use_switches)
    print("use_switches", use_switches, "columns", columns)
    subgrid = grid.subgridspec(1, columns, wspace=0, hspace=0, width_ratios=width_ratios)
    axes = []
    for switch in range(0, columns):
        axes.append(fig.add_subplot(subgrid[0,switch]))

    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        #ax.patch.set_facecolor('lightgray')
        #for _, spine in ax.spines.items():
        #    spine.set_color('lightgray')
        #    spine.set_linewidth(4)

    ax = axes[0]
    ax.text(0.5, 0.5, 'Capacity\nreduction', transform=ax.transAxes,
        fontsize=15, fontweight='normal', color="black", va='center', ha="center", 
    ) 
    ax = axes[1]
    ax.text(0.5, 0.5, 'Failure\nrate', transform=ax.transAxes,
        fontsize=15, fontweight='normal', color="black", va='center', ha="center", 
    )
    ax = axes[2]


    ax.text(0.5, 0.5, 'BS', fontsize=20, 
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, color='white', alpha=1,
        bbox=dict(boxstyle='round', facecolor='black', edgecolor='black')
    )

    for idx, switch in enumerate(use_switches):
        ax = axes[idx+3]
        ax.text(0.5, 0.5, str(switch), fontsize=20, 
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes, color='white', alpha=1,
            bbox=dict(boxstyle='circle', facecolor='black', edgecolor='black')
        )

def plot_scenario_rsa_multi_reduction(grid, fig, run, showX = False, alternate_rows=False, title_row=False, 
    all_switches=[], use_switches=[]):
    switch_cnt = run.get('scenario_switch_cnt')

    columns = len(all_switches)+3
    virtual_backup_switch_column = 2 # third column
    x = 1
    if columns > 8:
        x = 2
    width_ratios=[x,x,2] + [2]*len(all_switches)
    subgrid = grid.subgridspec(1, columns, wspace=0, hspace=0, width_ratios=width_ratios)
    axes = []
    for switch in range(0, columns):
        try:
            if switch == virtual_backup_switch_column:
                axes.append(fig.add_subplot(subgrid[0,switch]))
            else:
                axes.append(fig.add_subplot(subgrid[0,switch], sharey=axes[0]))
            ax = axes[-1]
            ax.get_yaxis().set_visible(False)
            if not showX:
                ax.get_xaxis().set_visible(False)   
        except:
            axes.append(fig.add_subplot(subgrid[0,switch]))

        ax = axes[-1]
        for _, spine in ax.spines.items():
            spine.set_color('lightgray')
            spine.set_linewidth(4)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if alternate_rows:
            ax.patch.set_facecolor('lightgray')
            ax.patch.set_alpha(0.2)
        else:
            ax.patch.set_facecolor('lightgray')
            ax.patch.set_alpha(0.4)         

    for ax in axes[0:2]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 

    # ------------- left-most column (recduction in percent)
    ax = axes[0]
    ax.text(0.5, 0.5, '-%d' % (100-run.get('param_topo_switch_capacity_overwrite')) + '\\%', transform=ax.transAxes,
        fontsize=18, fontweight='normal', color="black", va='center', ha="center", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    # ------------- second column (failure rate)
    ax = axes[1]
    ax.text(0.5, 0.5, '%.2f' % (run.get('rsa_table_percent_relocated')) + '\\%', transform=ax.transAxes,
        fontsize=18, fontweight='normal', color="black", va='center', ha="center", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    # ------------- fourth to last column (normal switches)
    maxy = 0
    for idx, switch in enumerate(all_switches):
        ax = axes[idx+3]
        thresh = run.get('scenario_table_capacity')

        d1 = 'dts_%d_table_datax' % (switch)
        d2 = 'dts_%d_table_datay_raw' % (switch)
        d3 = 'dts_%d_table_datay' % (switch)
        d4 = 'dts_%d_table_datay_shared' % (switch)

        datax = run.get(d1)
        if run.get(d2):
            raw_util = run.get(d2)
        else:
            raw_util = [0]*len(datax)
        if  max(raw_util) > maxy:
            maxy = max(raw_util)   

        # plot threshold
        t1 = ax.hlines(thresh, -1*XOFFSET, 400, color='blue', 
            label="Flow table capacity", linestyle='--', linewidth=1)

        if idx == len(all_switches)-1:
            ax.text(430, thresh, '%d' % (thresh), 
                fontsize=9, color='blue',
                verticalalignment='top', horizontalalignment='left', 
                alpha=1,
                bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='blue', alpha=1)
            )

        # red colored utilization over threshold
        fill_overutil = [True if x > thresh else False for x in raw_util]
        ax.fill_between(datax, raw_util, run.get(d3),
            where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
            label='Rules relocated')

        if switch in use_switches:
            ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
                interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')
        else:
            ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
                interpolate=True, color='gray', alpha=0.3, label='Not used as a remote switch')

        ax.plot(list(range(-1*XOFFSET,0)) + run.get(d1), [0]*XOFFSET + run.get(d3), color='black', linestyle='-', linewidth=0.75)

        fill_shared = [True if x < thresh else False for x in raw_util]
        ax.fill_between(datax, run.get(d3), raw_util,
            where=fill_shared, interpolate=True, color='green', alpha=0.5, 
            label='Rules stored in remote switch')
    
    # adjust height
    for ax in axes[3:]:
        ax.set_xlim(-1*XOFFSET,400)
        ax.set_xticks([0,100,200,300])
        ax.set_ylim(-500,maxy+500)

    # titles
    if title_row:
        ax = axes[0]
        ax.text(0.5, 1.2, 'Capacity\nreduction', transform=ax.transAxes,
            fontsize=15, fontweight='normal', color="black", va='bottom', ha="center", 
        ) 
        ax = axes[1]
        ax.text(0.5, 1.2, 'Failure\nrate', transform=ax.transAxes,
            fontsize=15, fontweight='normal', color="black", va='bottom', ha="center", 
        )
        ax = axes[2]
        ax.text(0.5, 1.2, 'Rules relocated to\nvirtual backup switch', transform=ax.transAxes,
            fontsize=15, fontweight='normal', color="black", va='bottom', ha="center", 
        ) 
        if not SHOW_TOPO:                            
            for idx, switch in enumerate(use_switches):
                ax = axes[idx+3]
                ax.text(0.5, 1.2, str(switch), fontsize=20, 
                    verticalalignment='center', horizontalalignment='center',
                    transform=ax.transAxes, color='white', alpha=1,
                    bbox=dict(boxstyle='circle', facecolor='black', edgecolor='black')
                )

                
      
    # -------------- virtual backup switch
    percent_relocated = run.get('rsa_table_percent_relocated') / (run.get('rsa_table_percent_relocated')  + run.get('rsa_table_percent_shared')) * 100
    ax = axes[virtual_backup_switch_column]
    backup = run.get('rsa_table_backup_switch_util_over_time')   
    datax = list(range(-1*XOFFSET, len(backup)))
    ax.fill_between(datax, [0]*len(datax), [0]*XOFFSET + backup, 
        interpolate=True, color='red', alpha=0.5, label='Rules in virtual backup switch')
    ax.plot(datax, [0]*XOFFSET + backup, color="red", linestyle=':', linewidth=0.5)
    ax.set_xlim(-1*XOFFSET,400)
    ax.set_xticks([0,100,200,300])
    if maxy < max(backup):
        ax.set_ylim(-500,max(backup)+500)
    else:
        ax.set_ylim(-500,maxy+500)
    ax.text(0.95, 0.8, ('%.2f' %  percent_relocated) + '\\%', transform=ax.transAxes,
        fontsize=15, fontweight='normal', color="red", va='center', ha="right", 
        #bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    return axes
    """
    # create a title with the most important parameters
    fig.suptitle("switches=%d hosts=%d m=%d hotspots=%d bottlenecks=%d capacity=%d%% seed=%d\n" % (
        self.param_topo_num_switches, self.param_topo_num_hosts, self.param_topo_scenario_ba_modelparam,
        self.param_topo_concentrate_demand, self.param_topo_bottleneck_cnt, self.factor,
        self.ctx.config.get('param_topo_seed')))
    """
    #gs.update(top=0.8)
    #fig.subplots_adjust(top=0.95) # padding top

def plot_topo_row(grid, fig, run):
    """ helper for the above function"""
    switch_cnt = run.get('scenario_switch_cnt')
    columns = switch_cnt+3
    width_ratios=[1,1,2] + [2]*switch_cnt
    subgrid = grid.subgridspec(1, columns, wspace=0, hspace=0, width_ratios=width_ratios)
    axes = []
    for switch in range(3, columns):
        axes.append(fig.add_subplot(subgrid[0,switch]))   
        ax = axes[-1]

        hosts_of_switch = {}
        edges = run.get('scenario_edges')
        for k, v in run.get('scenario_hosts_of_switch').items():
            hosts_of_switch[int(k)] = v
        plt_switches = list(range(0, run.get('scenario_switch_cnt')))
        concentrated_switches = [switch-3]
        utils.plot_topo_small(ax, hosts_of_switch, edges, plt_switches , concentrated_switches,
            switch_node_size=100, highlight_one=[switch-3],
            font_size=12)



def plot(blob, **kwargs):
    """
    Show example of flow delegation with reducing the threshold
    """

    # -----------------------
    # flow table utilization ratios
    # -----------------------

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

                #print(seed, util_avg, util_max, util_avg/float(util_max))


                #util_avg = run.get('scenario_table_util_avg_total')
                #util_max = run.get('scenario_table_util_max_total')
                ratio = (float(util_avg) / float(util_max))*100.0
                ratios.append(ratio)
                seed_ratio[ratio] = seed
                ratio_by_seed[seed] = ratio

    ratios = sorted(ratios)
    seed_ratio_sorted = sorted(seed_ratio.keys())


    includes = [
        'scenario_switch_cnt', 
        'scenario_table_capacity',
        'scenario_concentrated_switches', 
        'scenario_edges', 
        'scenario_bottlenecks', 
        'scenario_hosts_of_switch', 
        'scenario_link_util_mbit_max', 
        'scenario_table_util_avg_total',
        'scenario_table_util_max_total',
        'scenario_rules_per_switch_avg',

        'scenario_gen_param_topo_switch_capacity',
        'scenario_gen_param_topo_bottleneck_cnt',
        'scenario_gen_param_topo_bottleneck_duration',
        'scenario_gen_param_topo_bottleneck_intensity',
        'scenario_gen_param_topo_num_hosts',
        'scenario_gen_param_topo_num_flows',
        'scenario_gen_param_topo_concentrate_demand',
        'scenario_gen_param_topo_concentrate_demand_retries',
        'scenario_gen_param_topo_scenario_ba_modelparam',
        'scenario_gen_param_topo_traffic_scale',
        'scenario_gen_param_topo_iat_scale',
        'scenario_gen_param_topo_traffic_interswitch',
        'scenario_gen_param_topo_idle_timeout',

        'rsa_table_backup_switch_util_over_time',
        'rsa_table_fairness_avg',
        'rsa_link_util_delegated_mbit_max', 
        'rsa_ctrl_overhead_from_rsa',
        'rsa_table_percent_shared',
        'rsa_table_percent_relocated']


    keywords = ['hit_timelimit',
        'solver_cnt_feasable',
        'solver_cnt_infeasable',
        'solver_considered_ports',
        'solver_stats_time_solving',
        'table_datax',
        'table_datay_raw',
        'table_datay',
        'table_datay_shared',
        'ctrl_overhead',
        'ctrl_overhead_max',
        'ctrl_overhead_percent',
        'link_overhead',
        'link_overhead_max',
        'link_overhead_percent',
        'table_overhead',
        'table_overhead_max',
        'table_overhead_percent',
        'underutil',
        'underutil_max',
        'underutil_percent',
        'overutil',
        'overutil_percent',
        'overutil_max']


    PARAMS = [
        dict(v='$|H|$', d='Number of hosts', k='scenario_gen_param_topo_num_hosts'),
        dict(v='$|S|$', d='Number of switches', k='scenario_switch_cnt'),
        dict(v='m', d='Used for topology generation', k='scenario_gen_param_topo_scenario_ba_modelparam'),
        dict(v='$n_\\texttt{seed}$', d='Seed', k='param_topo_seed'),
        dict(v='$n_\\texttt{reduction}$', d='Capacity reduction factor', k='param_topo_switch_capacity_overwrite'),
        dict(v='$n_\\texttt{pairs}$ ', d='Number of host pairs', k='scenario_gen_param_topo_num_flows'),
        dict(v='$n_\\texttt{iat\\_scale}$', d='Global scale for $T_\\texttt{iat}$', k='scenario_gen_param_topo_iat_scale'),
        dict(v='$n_\\texttt{bneck}$', d='Number of temporal bottlenecks', k='scenario_gen_param_topo_bottleneck_cnt'),
        dict(v='$n_\\texttt{bneck\\_duration}$', d='Bottleneck duration', k='scenario_gen_param_topo_bottleneck_duration'),
        dict(v='$n_\\texttt{bneck\\_intensity}$', d='Bottleneck intensity', k='scenario_gen_param_topo_bottleneck_intensity'),
        dict(v='$n_\\texttt{isr}$', d='Inter switch ratio', k='scenario_gen_param_topo_traffic_interswitch'),
        dict(v='$n_\\texttt{hs}$', d='Number of hotspots', k='scenario_gen_param_topo_concentrate_demand'),
        dict(v='$n_\\texttt{hs\\_intensity}$', d='Hotspot intensity', k='scenario_gen_param_topo_concentrate_demand_retries'),
        dict(v='$n_\\texttt{traffic\\_scale}$', d='Global traffic scale', k='scenario_gen_param_topo_traffic_scale'),
        dict(v='$n_\\texttt{lifetime}$', d='Minimum flow rule lifetime', k='scenario_gen_param_topo_idle_timeout'),
    ]

    for keyword in keywords:
        includes += blob.find_columns(keyword)

    blob.include_parameters(**dict.fromkeys(includes, 1))





    # -----------------------
    # prepare data for plotting
    # -----------------------
    runs = []
    for use_seed in USE_SEEDS:
        subruns = blob.filter(param_topo_seed=use_seed)
        runs += subruns

    runs_by_seed = {}
    seeds = []
    ignored_seeds = []
    skipped = 0
    for run in runs:
        seed = run.get('param_topo_seed')
        if seed in SKIP_SEEDS:
            skipped += 1  
            if not seed in ignored_seeds:
                ignored_seeds.append(seed)
            continue        
        if run.get('hit_timelimit'):
            skipped += 1
            if not seed in ignored_seeds:
                ignored_seeds.append(seed)
            continue
        if run.get('rsa_solver_cnt_infeasable', 0) > 0:
            skipped += 1
            if not seed in ignored_seeds:
                ignored_seeds.append(seed)
            continue
        if not seed in seeds:
            seeds.append(seed)
        try:
            runs_by_seed[seed].append(run)
        except KeyError:
            runs_by_seed[seed] = [run]

        # the ratio between average and maximum table utilization for 
        # the case with "almost no reduction" is used as a reference value to
        # create different load sets
        #if run.get('scenario_gen_param_topo_switch_capacity') == 99:    
        #    util_avg = run.get('scenario_table_util_avg_total')
        #    util_max = run.get('scenario_table_util_max_total')
        #    ratio = util_avg / util_max
        #    ratio_by_seed[seed] = ratio


    print("runs", len(runs))
    print("seeds", len(seeds))
    print("ignored", len(ignored_seeds))
    #print("ratio", min(ratios), statistics.mean(ratios), max(ratios))



    #--------------------
    # print table with parameters
    #--------------------
    if 0:
        text = ''
        header = False
        for param in PARAMS:
            variable = param.get('v')
            desc = param.get('d')
            key = param.get('k')
            columns = []
            for seed in USE_SEEDS:
                run = runs_by_seed.get(seed)[0]
                columns.append('%d' % run.get(key))
            text += '%s & %s & %s \\\\\n' % (variable, desc, ' & '.join(columns))
        text = 'Parameter & Description & %s \\\\\n' % ' & '.join(['%d' % x for x in USE_SEEDS]) + text
        utils.export_textfile(text, 'scenario_table.txt', folder='99-dts')
    

    

    #--------------------
    # Failure rate for the example scenarios based on capacity reduction
    #--------------------
    if 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.tight_layout(pad=2.7) 
        colors = ['red', 'green', 'blue', 'm']
        markers = ['*', 'o', 'd', 's']
        for seed in seeds:
            runs = runs_by_seed.get(seed)
            datax = []
            datay = []
            datay2 = []
            for run in sorted(runs, key=lambda x: x.get('param_topo_switch_capacity_overwrite'), reverse=True):
                algo = run.get('param_dts_algo')
                if algo != 2: continue

                relative = run.get('rsa_table_percent_relocated') / (run.get('rsa_table_percent_relocated')  + run.get('rsa_table_percent_shared')) * 100
                absolute = run.get('rsa_table_percent_relocated')

                datax.append(100-run.get('param_topo_switch_capacity_overwrite'))
                datay.append(absolute)
                datay2.append(relative)
            color = colors.pop()
            marker = markers.pop()
            ratio = ratio_by_seed.get(seed)
            ax.plot(datax, datay, c=color, marker=marker, markevery=5, label=r'Scenario ID %d, flow table utilization ratio %.2f' % (seed, ratio) + '\\%%')
            #ax.plot(datax, datay2, c=color, linestyle='--', label=r'Scenario ID %d, flow table utilization ratio %.2f' % (seed, ratio) + '\\%%')
            ax.set_xlabel(r'Capacity reduction factor (\%)', fontsize=16)
            ax.set_ylabel(r'Failure rate (\%)', fontsize=16)

            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
            ax.legend(fontsize=16)

            for x, y in zip(datax, datay):
                if y > 0:
                    tt = 10
                    if x == 49: 
                        tt += 10
                    ax.vlines(x-1, 0, tt, color=color, linestyle=':', linewidth=2, alpha=1)
                    ax.text(x-1, tt, r'\textbf{' + str(int(x-1)) + r'\%}',
                        fontsize=16, 
                        verticalalignment='bottom', horizontalalignment='center',
                        color=color, alpha=1,
                        bbox=dict(facecolor='white', edgecolor=color)
                    )
                    break;


        utils.export(fig, 'failure_rate.pdf', folder='99-dts')
        plt.close()



    #--------------------
    # now for the different plots which will be stored in separate folders by seed
    #--------------------
    for seed in seeds:

        number_of_rows = len(THREHSOLDS_INCLUDED)
        thresholds = list(sorted(THREHSOLDS_INCLUDED, reverse=True))

        if len(USE_SEEDS) > 0:
            if not seed in USE_SEEDS:
                continue

        runs = runs_by_seed.get(seed)
        first_row_done = False

        folder = '99-dts/%2.2f_%d' % (ratio_by_seed.get(seed), seed)

        #--------------------
        # raw scenario
        #--------------------
        fig = plot_scenario_raw(runs[0], ratio_by_seed)
        utils.export(fig, 'scenario_%d.pdf' % seed, folder=folder)
        plt.close()
        
        #--------------------
        # single plots DTS
        #--------------------
        if 0:
            for run in sorted(runs, key=lambda x: x.get('param_topo_switch_capacity_overwrite'), reverse=True):
                thresh = run.get('param_topo_switch_capacity_overwrite')
                algo = run.get('param_dts_algo')
                if thresh in thresholds:

                    # determine the switch with highest utilization
                    max_overutil = 0
                    use_switch = None
                    switches = run.get('scenario_switch_cnt')
                    for switch in range(0, switches):
                        overutil = run.get('dts_%d_overutil_max' % switch)
                        if overutil > max_overutil:
                            max_overutil = overutil
                            use_switch = switch     

                    #util_avg = run.get('scenario_table_util_avg_total')
                    #util_max = run.get('scenario_table_util_max_total')
                    #print(seed, thresh, use_switch, util_avg, util_max)

                    fig, (ax, ax2) = plt.subplots(1,2, figsize=(12, 5), gridspec_kw = {'width_ratios':[3, 1]})
                    fig.tight_layout(h_pad=-1.5, pad=2.9) 

                    plot_single_dts(ax, run, use_switch)

                    plot_single_desc(ax2, run, use_switch)

                    # export
                    filename = "switch_%d_reduction_%d_algo_%d.pdf" % (use_switch, 100-thresh, algo)
                    utils.export(fig, filename, folder=folder)
                    plt.close()
                    #exit()
            
        #--------------------
        # all DTS runs in one plot, arranged in rows per reduction factor
        #--------------------
        if 0:
            fig = plt.figure(figsize=FIGSIZE)
            maingrid = gridspec.GridSpec(number_of_rows, 1, figure=fig, 
                wspace=0, hspace=0, left=0, right=0.95, top=0.9)      
            axcnt = 0

            # identify the switch with highest maximum table utilization
            switches = runs[0].get('scenario_switch_cnt')
            max_overutil = 0
            use_switch = None
            for switch in range(0, switches):
                overutil = runs[0].get('dts_%d_overutil_max' % switch)
                if overutil > max_overutil:
                    max_overutil = overutil
                    use_switch = switch

            sets = []
            current_thresh = 0
            for run in sorted(runs, key=lambda x: x.get('param_topo_switch_capacity_overwrite'), reverse=True):
                thresh = run.get('param_topo_switch_capacity_overwrite')
                if thresh in thresholds:
                    if thresh != current_thresh:
                        sets.append([run])
                    else:
                        sets[-1].append(run)
                    current_thresh = thresh

            alternate = 0
            for set in sets:

                if len(set) != 3:
                    logger.info("warning for seed: %d (only %d results, expected 3)" % (seed, len(set)))

                # print one dts scenario
                axes = plot_scenario_dts_multi_reduction(maingrid[axcnt], fig, set, use_switch,
                    showX=thresholds[-1]==thresh,
                    alternate_rows=alternate % 2 == 0,
                    title_row=not first_row_done
                )
                alternate += 1
                axcnt += 1
                first_row_done = True
                if axcnt == number_of_rows:
                    ax = axes[0]
                    ax.text(0.2, -0.5, 'Scenario id for reproducibility: %d  (switch %d)\nTable capacity without flow delegation: %d' % (
                        run.get('param_topo_seed'), use_switch,
                        run.get('scenario_table_util_max_total')), transform=ax.transAxes,
                        fontsize=14, fontweight='normal', color="darkgray", va='bottom', ha="left", 
                    )

            # legend in the bottom
            h1, l1 = axes[-1].get_legend_handles_labels()
            fig.legend(h1, l1, loc='lower right', ncol=3, fontsize=14)

            # export
            filename = "scenario_%d_dts_reduction.pdf" % (seed)
            utils.export(fig, filename, folder=folder)
            plt.close()

        #--------------------
        # all RSA runs in one plot, arranged in rows per reduction factor
        #--------------------
        if 0:


            thresholds = list(sorted(THREHSOLDS_INCLUDED_RSA, reverse=True))
            if THRESHOLD_BY_SEED.get(seed):
                thresholds = list(sorted( THRESHOLD_BY_SEED.get(seed), reverse=True))
            number_of_rows_rsaplot = len(thresholds)

            # precalculate the number of "extra" rows needed because switch count changes
            use_switches_by_thresh = {}
            use_switches_all = []
            for run in sorted(runs, key=lambda x: x.get('param_topo_switch_capacity_overwrite'), reverse=True):
                algo = run.get('param_dts_algo')
                if algo != 2: continue
                if run.get('param_topo_switch_capacity_overwrite') in thresholds:
                    use_switches = []
                    for switch in range(0, run.get('scenario_switch_cnt')):
                        d2 = 'dts_%d_table_datay_raw' % (switch)
                        d3 = 'dts_%d_table_datay' % (switch)
                        d4 = 'dts_%d_table_datay_shared' % (switch)
                        thresh = run.get('scenario_table_capacity')
                        raw_util = run.get(d2)
                        test_raw = [1 if x > thresh else 0 for x in raw_util]
                        test_shared = [abs(x1-x2) for x1, x2 in zip(raw_util, run.get(d3))]
                        if sum(test_shared) > 0 or sum(test_raw) > 0:
                            use_switches.append(switch) 
                            if not switch in use_switches_all:
                                use_switches_all.append(switch)
                    use_switches_by_thresh[run.get('param_topo_switch_capacity_overwrite')] = use_switches

            fig = plt.figure(figsize=FIGSIZE)

            maingrid = gridspec.GridSpec(number_of_rows_rsaplot+1, 1, figure=fig, 
                wspace=0, hspace=0, left=0, right=0.95, top=0.98, height_ratios=[1] + [2]*number_of_rows_rsaplot)      
          

            axcnt = 0

            use_switches_all = sorted(use_switches_all)

            # title row
            plot_scenario_rsa_multi_reduction_title(maingrid[axcnt], fig, use_switches_all,
                alternate_rows=True)
            axcnt+=1

            alternate  = False
            for run in sorted(runs, key=lambda x: x.get('param_topo_switch_capacity_overwrite'), reverse=True):

                thresh = run.get('param_topo_switch_capacity_overwrite')
                algo = run.get('param_dts_algo')
                if algo != 2: continue
                if thresh in thresholds:
                    axes = plot_scenario_rsa_multi_reduction(maingrid[axcnt], fig, run, 
                        showX=thresholds[-1]==thresh,
                        alternate_rows=alternate,
                        title_row=False,
                        all_switches=use_switches_all,
                        use_switches=use_switches_by_thresh.get(thresh)
                    )
                    alternate = not alternate
                    axcnt += 1
                    if axcnt == number_of_rows_rsaplot+1:
                        ax = axes[0]
                        ax.text(0.2, -0.5, 'Scenario id for reproducibility: %d\nCapacity without flow delegation: %d' % (run.get('param_topo_seed'),
                            run.get('scenario_table_util_max_total')), transform=ax.transAxes,
                            fontsize=14, fontweight='normal', color="darkgray", va='bottom', ha="left", 
                        )

            # legend in the bottom
            h1, l1 = axes[-1].get_legend_handles_labels()
            h2, l2 = axes[2].get_legend_handles_labels()
            fig.legend(h1+h2, l1+l2, loc='lower right', ncol=3, fontsize=14)

            # export
            filename = "scenario_%d_rsa_reduction.pdf" % (seed)
            utils.export(fig, filename, folder=folder)
            plt.close()

    
        #--------------------
        # all 99 for slides
        #--------------------
        if 1:
            folder = '99-dts/%2.2f_%d/99' % (ratio_by_seed.get(seed), seed)
            thresholds = list(sorted(THRESHOLDS_RSA_SINGLE, reverse=True))  
            for run in sorted(runs, key=lambda x: x.get('param_topo_switch_capacity_overwrite'), reverse=True):
                thresh = run.get('param_topo_switch_capacity_overwrite')
                algo = run.get('param_dts_algo')
                if not algo == 2: continue
                fig = plot_single_rsa_for_slides(run, ratio_by_seed)   
                filename = "example_rsa_%d_notopo.pdf" % (100-thresh)
                utils.export(fig, filename, folder=folder)
                plt.close()
                #exit()

  
        #--------------------
        # single plots RSA
        #--------------------
        if 0:
            thresholds = list(sorted(THRESHOLDS_RSA_SINGLE, reverse=True))  
            for run in sorted(runs, key=lambda x: x.get('param_topo_switch_capacity_overwrite'), reverse=True):
                thresh = run.get('param_topo_switch_capacity_overwrite')
                algo = run.get('param_dts_algo')
                if not algo == 2: continue
                if thresh in thresholds:
                    # default (with topology)
                    fig = plot_single_rsa(run, ratio_by_seed)   
                    # export
                    filename = "rsa_%d.pdf" % (100-thresh)
                    utils.export(fig, filename, folder=folder)
                    plt.close()

                    fig = plot_single_rsa(run, ratio_by_seed, hide_topo=True)   
                    # export
                    filename = "rsa_%d_notopo.pdf" % (100-thresh)
                    utils.export(fig, filename, folder=folder)
                    plt.close()
