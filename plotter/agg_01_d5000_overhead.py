import logging, math, json, pickle, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from topo.static import LAYOUTS
import statistics

logger = logging.getLogger(__name__)

from . import agg_2_utils as utils

XOFFSET = 60

def plot_scenario_raw(run):

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
        
        ax.set_xlim(0, 400)
        ax.set_ylabel('Flow table utilization')
        
        thresh = run.get('scenario_table_capacity')
        datax = run.get('dts_%d_table_datax' % switch)
        datay = run.get('dts_%d_table_datay_raw' % switch)

        if max(datay) > maxy:
            maxy = max(datay)

        ax.plot(list(range(-1*XOFFSET,0)) + datax, [0]*XOFFSET + datay, color='black', linestyle='-', linewidth=0.75)

        ax.fill_between(datax, [0]*len(datay), [min(thresh, x) for x in datay],
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

        ax.hlines(thresh, 0, 400, color='blue', 
                label="Flow table capacity", linestyle='--', linewidth=1)  

        d2 = 'dts_%d_table_datay_raw' % (switch)
        d3 = 'dts_%d_table_datay' % (switch)
        fill_overutil = [True if x > thresh else False for x in datay]
        ax.fill_between(datax, [thresh]*len(datax), datay,
            where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
            label='Bottleneck')

        # plot bottlenecks
        ax.hlines(-1*XOFFSET, -1*XOFFSET, 400, color='gray', linestyle='-', alpha=0.3, linewidth=7)
        bottleneck_data = run.get('scenario_bottlenecks')
        set_label = 0
        for start, end, details in bottleneck_data:
            if set_label == 0:
                ax.hlines(-1*XOFFSET, start, end, color='red', linestyle='-', alpha=0.3, linewidth=7)
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


def plot(blob, **kwargs):
    "Plot dts scores (overutil, underutil, overheads)"

    utils.EXPORT_BLOB = blob
   

    includes = ['hit_timelimit', 'scenario_switch_cnt', 'scenario_table_capacity', 'scenario_table_capacity_reduction',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch']

    includes += blob.find_columns('ctrl_overhead_percent')
    includes += blob.find_columns('ctrl_overhead')
    includes += blob.find_columns('link_overhead')
    includes += blob.find_columns('link_overhead_percent')
    includes += blob.find_columns('table_overhead_percent')
    includes += blob.find_columns('table_overhead')
    includes += blob.find_columns('underutil_percent')
    includes += blob.find_columns('overutil_percent')
    includes += blob.find_columns('table_datax')
    includes += blob.find_columns('table_datay_raw')
    includes += blob.find_columns('table_datay')
    includes += blob.find_columns('table_datay_shared')
    includes += blob.find_columns('solver_cnt_infeasable')
    
    

    blob.include_parameters(**dict.fromkeys(includes, 1))

    runs = blob.filter(**dict())

    # -----------------------
    # prepare data for plotting
    # -----------------------
    DATA = {}
    seeds = []
    ignore_seeds = []
    infeasible = 0
    timelimit = 0
    switchcnt = 0
    switchcnt_ds = 0
    for run in runs:
        seed = run.get('param_topo_seed')
        if run.get('hit_timelimit'):
            timelimit += 1
            ignore_seeds.append(seed)
            continue
        param_dts_algo = run.get('param_dts_algo')
        param_dts_look_ahead = run.get('param_dts_look_ahead')
        if not seed in seeds:
            seeds.append(seed)
        if not DATA.get(param_dts_algo):
            DATA[param_dts_algo] = {}
        for switch in range(0, run.get('scenario_switch_cnt')):
            DATA[param_dts_algo][(seed, switch)] = run
            if param_dts_algo == 3:
                switchcnt += 1
                if run.get('dts_%d_table_overhead_percent' % (switch)) > 0:
                    switchcnt_ds += 1
            if run.get('dts_%d_solver_cnt_infeasable' % (switch), 0) > 0:
                #ignore_seeds.append(seed)
                infeasible += 1

        #if seed == 79859:
        #    fig = plot_scenario_raw(run)
        #    utils.export(fig, 'scenario_%d.pdf' % seed, folder='dts_compare/underutil')
        #    plt.close()

    print("infeasible", infeasible)
    print("timelimit", timelimit)
    print("switchcnt", switchcnt)
    print("switchcnt_ds", switchcnt_ds)

    # -----------------------
    # Table Overhead
    # -----------------------
    if 0:

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        fig3.tight_layout(pad=2.7)
        ax3.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax3.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel(r'Table overhead (\%)', fontsize=15)
        ax3.set_ylabel(r'CDF', fontsize=15)
        #ax3.set_xlim(0,60)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.tight_layout(pad=2.7)
        ax2.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax2.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel(r'Capacity reduction factor (\%)', fontsize=15)
        ax2.set_ylabel(r'Table overhead (rules)', fontsize=15)
        #ax2.set_xlim(0,60)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.tight_layout(pad=2.7)
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(r'Table overhead (rules)', fontsize=15)
        ax.set_ylabel('CDF', fontsize=15)

        for param_dts_algo, DATA1 in DATA.items():
            color = 'red'
            label = 'Select-Opt'
            marker = 'd'
            table_overhead = [] 
            result_table_overhead_percent = []

            if param_dts_algo == 2:
                color = 'blue'
                label = 'Select-CopyFirst'
                marker = 'o'
                axcnt = 1
            if param_dts_algo == 3:
                color = 'green'
                label = 'Select-Greedy'
                marker = 'X'
                axcnt = 1

            underutil_skipped = 0
            underutil_arr = []

            overhead_per_reduction = {}
            for seedswitch, run in DATA1.items():
                seed, switch = seedswitch
                #if run.get('scenario_switch_cnt') == 2: continue
                if seed in ignore_seeds:
                    underutil_skipped += 1
                    continue
                if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                    continue



                thresh = run.get('scenario_table_capacity')
                d2 = 'dts_%d_table_datay_raw' % (switch)
                raw_util = run.get(d2)
                fill_overutil = [1 if x > thresh else 0 for x in raw_util]
                new_table_overhead = (run.get('dts_%d_table_overhead' % switch) / float(sum(fill_overutil)))
                table_overhead_percent = run.get('dts_%d_table_overhead_percent' % (switch))

                overutil = run.get('dts_%d_overutil_percent' % (switch))
                if overutil == 100:
                    # this can only happen in extremely rare situation where the threshold and the
                    # actual capacity are identical; is ignored here; An example for this can be
                    # reproced using scenario id 117841 (switch 11)
                    print("100", seed, switch)
                    continue;

                result_table_overhead_percent.append(table_overhead_percent)
                table_overhead.append(new_table_overhead)

                factor = 100-run.get('scenario_table_capacity_reduction')
                if factor <= 60:
                    try:
                        overhead_per_reduction[factor].append(new_table_overhead)
                    except:
                        overhead_per_reduction[factor] = [new_table_overhead]


            utils.plotcdf(ax3, result_table_overhead_percent, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
            
            utils.plotcdf(ax, table_overhead, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
            
            print("")
            for perc in [25,50,75,80,90,99,100]:
                p = np.percentile(table_overhead, perc)
                print("table overhead", param_dts_algo, perc, "   ", p)

            average = []
            boxvalues = []
            for _, values in sorted(overhead_per_reduction.items()):
                boxvalues.append(values)
                if len(values) > 0:
                    average.append(np.percentile(values, 90))
                else:
                    average.append(0) 

            ax2.plot([x+1 for x in np.arange(len(average))], average, label=label, marker=marker, linewidth=1.5, color=color)

        handles, labels = ax3.get_legend_handles_labels()
        fig3.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
        fig3.subplots_adjust(left=0.1, top=0.85) # padding top
        utils.export(fig3, 'dts_compare_table3.pdf', folder='dts_compare')      

        handles, labels = ax2.get_legend_handles_labels()
        fig2.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
        fig2.subplots_adjust(left=0.1, top=0.85) # padding top
        utils.export(fig2, 'dts_compare_table2.pdf', folder='dts_compare')      

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
        fig.subplots_adjust(left=0.1, top=0.85) # padding top
        utils.export(fig, 'dts_compare_table.pdf', folder='dts_compare')
        plt.close()

    # -----------------------
    # Link Overhead
    # -----------------------
    if 0:

        for uselabel, use_algos in zip(['use123_', 'use2_', 'use1_', 'use12_'], [[1,2,3], [2], [1], [1,2]]):

            fig3, ax3 = plt.subplots(figsize=(8, 4))
            fig3.tight_layout(pad=2.7)
            ax3.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax3.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax3.set_xlabel(r'Link overhead (\%)', fontsize=15)
            ax3.set_ylabel(r'CDF', fontsize=15)
            #ax3.set_xlim(0,60)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            fig2.tight_layout(pad=2.7)
            ax2.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax2.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel(r'Capacity reduction factor (\%)', fontsize=15)
            ax2.set_ylabel(r'Link overhead (Mbit/s)', fontsize=15)
            #ax2.set_xlim(0,60)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.tight_layout(pad=2.7)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel(r'Link overhead (Mbit/s)', fontsize=15)
            ax.set_ylabel('CDF', fontsize=15)
            ax.set_xscale('log')
            #from matplotlib.ticker import ScalarFormatter
            #ax.xaxis.set_major_formatter(ScalarFormatter())
            from matplotlib.ticker import FuncFormatter
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            ax.xaxis.set_major_formatter(formatter)

            for param_dts_algo, DATA1 in DATA.items():
                if param_dts_algo not in use_algos: 
                    continue;
                color = 'red'
                label = 'Select-Opt'
                marker = 'd'
                link_overhead = [] 
                result_link_overhead_percent = []

                if param_dts_algo == 2:
                    color = 'blue'
                    label = 'Select-CopyFirst'
                    marker = 'o'
                    axcnt = 1
                if param_dts_algo == 3:
                    color = 'green'
                    label = 'Select-Greedy'
                    marker = 'X'
                    axcnt = 1

                underutil_skipped = 0
                underutil_arr = []

                overhead_per_reduction = {}
                for seedswitch, run in DATA1.items():
                    seed, switch = seedswitch
                    #if run.get('scenario_switch_cnt') == 2: continue
                    if seed in ignore_seeds:
                        underutil_skipped += 1
                        continue
                    if run.get('dts_%d_link_overhead_percent' % (switch)) == 0:
                        continue



                    thresh = run.get('scenario_table_capacity')
                    d2 = 'dts_%d_table_datay_raw' % (switch)
                    raw_util = run.get(d2)
                    fill_overutil = [1 if x > thresh else 0 for x in raw_util]
                    new_link_overhead = ((run.get('dts_%d_link_overhead' % switch) / 1000000) / float(sum(fill_overutil)))
                    link_overhead_percent = run.get('dts_%d_link_overhead_percent' % (switch))

                    overutil = run.get('dts_%d_overutil_percent' % (switch))
                    if overutil == 100:
                        # this can only happen in extremely rare situation where the threshold and the
                        # actual capacity are identical; is ignored here; An example for this can be
                        # reproced using scenario id 117841 (switch 11)
                        print("100", seed, switch)
                        continue;

                    result_link_overhead_percent.append(link_overhead_percent)
                    link_overhead.append(new_link_overhead)

                    factor = 100-run.get('scenario_table_capacity_reduction')
                    if factor <= 50:
                        try:
                            overhead_per_reduction[factor].append(new_link_overhead)
                        except:
                            overhead_per_reduction[factor] = [new_link_overhead]


                utils.plotcdf(ax3, result_link_overhead_percent, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
                
                utils.plotcdf(ax, link_overhead, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
                
                print("len", len(link_overhead), underutil_skipped)
                for perc in [25,50,75,80,90,99,100]:
                    p = np.percentile(link_overhead, perc)
                    print("link overhead", param_dts_algo, perc, "   ", p)

                average = []
                boxvalues = []
                for _, values in sorted(overhead_per_reduction.items()):
                    boxvalues.append(values)
                    if len(values) > 0:
                        average.append(np.percentile(values, 90))
                    else:
                        average.append(0) 

                ax2.plot([x+1 for x in np.arange(len(average))], average, label=label, marker=marker, linewidth=1.5, color=color)

            handles, labels = ax3.get_legend_handles_labels()
            fig3.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
            fig3.subplots_adjust(left=0.1, top=0.85) # padding top
            utils.export(fig3, '%sdts_compare_link3.pdf' % uselabel, folder='dts_compare')      

            handles, labels = ax2.get_legend_handles_labels()
            fig2.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
            fig2.subplots_adjust(left=0.1, top=0.85) # padding top
            utils.export(fig2, '%sdts_compare_link2.pdf' % uselabel, folder='dts_compare')      

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
            fig.subplots_adjust(left=0.1, top=0.85) # padding top
            utils.export(fig, '%sdts_compare_link.pdf' % uselabel, folder='dts_compare')
            plt.close()

    # -----------------------
    # Ctrl Overhead
    # -----------------------
    if 1:

        for uselabel, use_algos in zip(['use123_', 'use2_', 'use1_', 'use12_'], [[1,2,3], [2], [1], [1,2]]):

            fig3, ax3 = plt.subplots(figsize=(8, 4))
            fig3.tight_layout(pad=2.7)
            ax3.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax3.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax3.set_xlabel(r'Control overhead (control messages/s)', fontsize=15)
            ax3.set_ylabel(r'CDF', fontsize=15)
            #ax3.set_xlim(0,60)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            fig2.tight_layout(pad=2.7)
            ax2.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax2.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel(r'Capacity reduction factor (\%)', fontsize=15)
            ax2.set_ylabel(r'Control overhead (control messages/s)', fontsize=15)
            #ax2.set_xlim(0,60)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.tight_layout(pad=2.7)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel(r'Control overhead (control messages/s)', fontsize=15)
            ax.set_ylabel('CDF', fontsize=15)
            #ax.set_xscale('log')
            #from matplotlib.ticker import ScalarFormatter
            #ax.xaxis.set_major_formatter(ScalarFormatter())
            from matplotlib.ticker import FuncFormatter
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            ax.xaxis.set_major_formatter(formatter)

            for param_dts_algo, DATA1 in DATA.items():
                if param_dts_algo not in use_algos: 
                    continue;

                color = 'red'
                label = 'Select-Opt'
                marker = 'd'
                ctrl_overhead = [] 
                result_ctrl_overhead_percent = []

                if param_dts_algo == 2:
                    color = 'blue'
                    label = 'Select-CopyFirst'
                    marker = 'o'
                    axcnt = 1
                if param_dts_algo == 3:
                    color = 'green'
                    label = 'Select-Greedy'
                    marker = 'X'
                    axcnt = 1

                underutil_skipped = 0
                underutil_arr = []

                overhead_per_reduction = {}
                for seedswitch, run in DATA1.items():
                    seed, switch = seedswitch
                    #if run.get('scenario_switch_cnt') == 2: continue
                    if seed in ignore_seeds:
                        underutil_skipped += 1
                        continue
                    if run.get('dts_%d_ctrl_overhead_percent' % (switch)) == 0:
                        continue



                    thresh = run.get('scenario_table_capacity')
                    d2 = 'dts_%d_table_datay_raw' % (switch)
                    raw_util = run.get(d2)
                    fill_overutil = [1 if x > thresh else 0 for x in raw_util]
                    new_ctrl_overhead = ((run.get('dts_%d_ctrl_overhead' % switch)) / float(sum(fill_overutil)))
                    ctrl_overhead_percent = run.get('dts_%d_ctrl_overhead_percent' % (switch))

                    overutil = run.get('dts_%d_overutil_percent' % (switch))
                    if overutil == 100:
                        # this can only happen in extremely rare situation where the threshold and the
                        # actual capacity are identical; is ignored here; An example for this can be
                        # reproced using scenario id 117841 (switch 11)
                        print("100", seed, switch)
                        continue;

                    result_ctrl_overhead_percent.append(ctrl_overhead_percent)
                    ctrl_overhead.append(new_ctrl_overhead)

                    factor = 100-run.get('scenario_table_capacity_reduction')
                    if factor <= 50:
                        try:
                            overhead_per_reduction[factor].append(new_ctrl_overhead)
                        except:
                            overhead_per_reduction[factor] = [new_ctrl_overhead]


                utils.plotcdf(ax3, result_ctrl_overhead_percent, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
                
                utils.plotcdf(ax, ctrl_overhead, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
                
                print("len", len(ctrl_overhead), underutil_skipped)
                for perc in [25,50,75,80,90,99,100]:
                    p = np.percentile(ctrl_overhead, perc)
                    print("ctrl overhead", param_dts_algo, perc, "   ", p)

                average = []
                boxvalues = []
                for _, values in sorted(overhead_per_reduction.items()):
                    boxvalues.append(values)
                    if len(values) > 0:
                        average.append(np.percentile(values, 90))
                    else:
                        average.append(0) 

                ax2.plot([x+1 for x in np.arange(len(average))], average, label=label, marker=marker, linewidth=1.5, color=color)

            handles, labels = ax3.get_legend_handles_labels()
            fig3.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
            fig3.subplots_adjust(left=0.1, top=0.85) # padding top
            utils.export(fig3, '%sdts_compare_ctrl3.pdf' % uselabel, folder='dts_compare')      

            handles, labels = ax2.get_legend_handles_labels()
            fig2.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
            fig2.subplots_adjust(left=0.1, top=0.85) # padding top
            utils.export(fig2, '%sdts_compare_ctrl2.pdf' % uselabel, folder='dts_compare')      

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
            fig.subplots_adjust(left=0.1, top=0.85) # padding top
            utils.export(fig, '%sdts_compare_ctrl.pdf' % uselabel, folder='dts_compare')
            plt.close()

    # -----------------------
    # Figure: over and underutilization
    # -----------------------
    if 0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.tight_layout(pad=2.7)
        for ax, xlabel in zip(fig.axes, 
            [r'Overutilization (\%)', r'Underutilization (\%)']):
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel('CDF', fontsize=15)

        for param_dts_algo, DATA1 in DATA.items():
            color = 'red'
            label = 'Select-Opt'
            marker = 'd'
            result_underutil_percent = [] 
            result_overutil_percent = []

            if param_dts_algo == 2:
                color = 'blue'
                label = 'Select-CopyFirst'
                marker = 'o'
                axcnt = 1
            if param_dts_algo == 3:
                color = 'green'
                label = 'Select-Greedy'
                marker = 'X'
                axcnt = 1

            for seedswitch, run in DATA1.items():
                seed, switch = seedswitch
                if seed in ignore_seeds:
                    continue
                if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                    continue

                underutil = run.get('dts_%d_underutil_percent' % (switch))
                overutil = run.get('dts_%d_overutil_percent' % (switch))

                if overutil == 100:
                    # this can only happen in extremely rare situation where the threshold and the
                    # actual capacity are identical; is ignored here; An example for this can be
                    # reproced using scenario id 117841 (switch 11)
                    print("100", seed, switch)
                    continue;
                result_underutil_percent.append(underutil)
                result_overutil_percent.append(overutil)

            utils.plotcdf(axes[0], result_overutil_percent, label=label, marker=marker, markevery=20, linewidth=1.5, color=color)
            utils.plotcdf(axes[1], result_underutil_percent, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)

        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
        fig.subplots_adjust(left=0.1, top=0.85) # padding top
        utils.export(fig, 'dts_compare_overutil_and_underutil.pdf', folder='dts_compare')
        plt.close()

    # -----------------------
    # Figure: cdfs for the other three metrics
    # -----------------------
    if 0:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout(pad=2.7)
        for ax, xlabel in zip(fig.axes, 
            [r'Table overhead (\%)', r'Link overhead (\%)', r'Control overhead (\%)']):
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel('CDF', fontsize=15)

        for param_dts_algo, DATA1 in DATA.items():

            result_ctrl_overhead_percent = []
            result_link_overhead_percent = [] 
            result_table_overhead_percent = [] 

            color = 'red'
            label = 'Select-Opt'
            marker = 'd'
            if param_dts_algo == 2:
                color = 'blue'
                label = 'Select-CopyFirst'
                marker = 'o'
                axcnt = 1
            if param_dts_algo == 3:
                color = 'green'
                label = 'Select-Greedy'
                marker = 'X'
                axcnt = 1

            for seedswitch, run in DATA1.items():
                seed, switch = seedswitch
                if seed in ignore_seeds:
                    continue
                if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                    continue

                ctrl_overhead_percent = run.get('dts_%d_ctrl_overhead_percent' % (switch))
                link_overhead_percent = run.get('dts_%d_link_overhead_percent' % (switch))
                table_overhead_percent = run.get('dts_%d_table_overhead_percent' % (switch))

                result_ctrl_overhead_percent.append(ctrl_overhead_percent)
                result_link_overhead_percent.append(link_overhead_percent)
                result_table_overhead_percent.append(table_overhead_percent)

            utils.plotcdf(axes[0], result_table_overhead_percent, 
                label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
            utils.plotcdf(axes[1], result_link_overhead_percent, 
                label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
            utils.plotcdf(axes[2], result_ctrl_overhead_percent, 
                label=label, marker=marker, markevery=500, linewidth=1.5, color=color)

        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
        fig.subplots_adjust(left=0.1, top=0.85) # padding top
        utils.export(fig, 'dts_compare_other_metrics.pdf', folder='dts_compare')
        plt.close()
        exit(1)

