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

def plot_overutil_examples(fig, ax, ax2, run, switch, scale, **kwargs):
    thresh = run.get('scenario_table_capacity')

    d1 = 'dts_%d_table_datax' % (switch)
    d2 = 'dts_%d_table_datay_raw' % (switch)
    d3 = 'dts_%d_table_datay' % (switch)
    d4 = 'dts_%d_table_datay_shared' % (switch)

    datax = run.get(d1)
    raw_util = run.get(d2)
    actual_util = run.get(d3)

    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_ylim(0,max(raw_util)*1.5)
    ax.set_xlim(-60,460)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.set_ylabel('Flow table utilization', fontsize=15)

    # plot threshold

    ax.text(400, thresh, '%d' % (thresh), 
        fontsize=12, color='blue',
        verticalalignment='top', horizontalalignment='left', 
        alpha=1,
        bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='blue', alpha=1)
    )

    zoomx = 0
    zoomy = 0

    drawn = False
    cnt = 0
    for y in actual_util:
        if y > thresh:
            if not drawn:
                drawn = True
                ra = (max(raw_util)*1.5) / (460+60)
                print(cnt, y, ra)
                circle = patches.Ellipse((datax[cnt], y), 20, 20*ra, lw=5, fill=False,edgecolor='blue', color='red', zorder=10)
                ax.add_artist(circle)
                zoomx = datax[cnt]
                zoomy = y

        cnt += 1


    # red colored utilization over threshold
    fill_overutil = [True if x > thresh else False for x in raw_util]
    ax.hlines(thresh, 0, 400, color='blue', 
            label="Flow table capacity", linestyle='--', linewidth=1)  
    ax.fill_between(datax, raw_util, run.get(d3),
        where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
        label='Rules relocated')
    ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
        interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')
    ax.plot(run.get(d1), run.get(d3), color='black', linestyle='-', linewidth=0.75)
    ax.legend(loc='upper left', fontsize=14)

    overutil = run.get('dts_%d_overutil_percent' % (switch))
    ax2.text(0.1, 0.8, ('Scenario id: %d (switch %d)\nOverutilization: %.2f' % (run.get('param_topo_seed'), switch, overutil)) + r'\%',
        transform=ax2.transAxes,
        fontsize=14, fontweight='normal', color="black", va='bottom', ha="left", 
        bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    ax2.fill_between(datax, raw_util, run.get(d3),
        where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
        label='Rules relocated')
    ax2.hlines(thresh, 0, 400, color='blue', 
            label="Flow table capacity", linestyle='--', linewidth=1)  
    ax2.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
        interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')
    ax2.plot(run.get(d1), run.get(d3), color='black', linestyle='-', linewidth=0.75)
    ax2.set_xlim(zoomx-10, zoomx+10)
    ax2.set_ylim(zoomy-30-scale, zoomy+30+scale)
    ax2.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax2.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    return ax

def plot_underutil_examples(fig, ax, run, switch, **kwargs):
    thresh = run.get('scenario_table_capacity')

    d1 = 'dts_%d_table_datax' % (switch)
    d2 = 'dts_%d_table_datay_raw' % (switch)
    d3 = 'dts_%d_table_datay' % (switch)
    d4 = 'dts_%d_table_datay_shared' % (switch)

    datax = run.get(d1)
    raw_util = run.get(d2)
    actual_util = run.get(d3)

    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_ylim(0,max(raw_util)*1.5)
    ax.set_xlim(-60,460)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.set_ylabel('Flow table utilization', fontsize=15)

    # plot threshold

    ax.text(400, thresh, '%d' % (thresh), 
        fontsize=12, color='blue',
        verticalalignment='top', horizontalalignment='left', 
        alpha=1,
        bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='blue', alpha=1)
    )

    zoomx = 0
    zoomy = 0

    drawn = False
    cnt = 0
    for y in actual_util:
        if y > thresh:
            if not drawn:
                drawn = True
                ra = (max(raw_util)*1.5) / (460+60)
                print(cnt, y, ra)
                circle = patches.Ellipse((datax[cnt], y), 20, 20*ra, lw=5, fill=False,edgecolor='blue', color='red', zorder=10)
                ax.add_artist(circle)
                zoomx = datax[cnt]
                zoomy = y

        cnt += 1

    fill_underutil = [True if x < thresh and y > thresh else False for x, y in zip(run.get(d3), raw_util)]
    ax.fill_between(datax, run.get(d3), [thresh]*len(datax), 
        where=fill_underutil, interpolate=True, color='red', alpha=1, label='Underutilization')


    under = run.get('dts_%d_underutil_percent' % (switch))
    ax.text(0.9, 0.8, ('Scenario id: %d (switch %d)\nOverutilization: %.2f' % (run.get('param_topo_seed'), switch, under)) + r'\%',
        transform=ax.transAxes,
        fontsize=16, fontweight='normal', color="black", va='bottom', ha="right", 
        bbox=dict(boxstyle="square", ec='white', fc='white',)
    )

    # red colored utilization over threshold
    fill_overutil = [True if x > thresh else False for x in raw_util]
    ax.hlines(thresh, 0, 400, color='blue', 
            label="Flow table capacity", linestyle='--', linewidth=1)  
    ax.fill_between(datax, raw_util, run.get(d3),
        where=fill_overutil, interpolate=True, color='red', alpha=0.2, 
        label='Rules relocated')
    ax.fill_between(datax, [0]*len(run.get(d3)), run.get(d3),
        interpolate=True, color='orange', alpha=0.3, label='Rules not touched by flow delegation')
    ax.plot(run.get(d1), run.get(d3), color='black', linestyle='-', linewidth=0.75)
    ax.legend(loc='upper left', fontsize=14)

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
   
    # -----------------------
    # Overutilization examples
    # -----------------------
    if 0:
        for algo in [3]:
            filepath = os.path.join(utils.EXPORT_FOLDER, 'dts_compare/overutil', 'overutil_%d.json' % algo)
            with open(filepath, 'r') as file:
                data = json.loads(file.read())
                for ov, seed, switch, run in data:
                    fig, (ax, ax2) = plt.subplots(1,2, figsize=(12, 5), gridspec_kw = {'width_ratios':[2, 1]})
                    fig.tight_layout(h_pad=-1.5, pad=2.9) 
                    plot_overutil_examples(fig, ax, ax2, run, switch, 0)
                    filename = "overutil_details_%d_%.2f.pdf" % (algo, ov)
                    utils.export(fig, filename, folder='dts_compare/overutil')
                    plt.close()
            filepath = os.path.join(utils.EXPORT_FOLDER, 'dts_compare/overutil', 'overutil_sum_%d.json' % algo)
            with open(filepath, 'r') as file:
                data = json.loads(file.read())
                for ov, seed, switch, run in data:
                    fig, (ax, ax2) = plt.subplots(1,2, figsize=(12, 5), gridspec_kw = {'width_ratios':[2, 1]})
                    fig.tight_layout(h_pad=-1.5, pad=2.9) 
                    plot_overutil_examples(fig, ax, ax2, run, switch, 20)
                    filename = "overutil_sum_details_%d_%.2f.pdf" % (algo, ov)
                    utils.export(fig, filename, folder='dts_compare/overutil')
                    plt.close()

    # -----------------------
    # Underutilization examples
    # -----------------------
    if 0:
        for algo in [1,3]:
            filepath = os.path.join(utils.EXPORT_FOLDER, 'dts_compare/underutil', 'underutil_%d.json' % algo)
            with open(filepath, 'r') as file:
                data = json.loads(file.read())
                print(len(data))
                for ov, seed, switch, run in data:
                    fig, ax, = plt.subplots(figsize=(12, 5))
                    fig.tight_layout(h_pad=-1.5, pad=2.9) 
                    plot_underutil_examples(fig, ax, run, switch)
                    filename = "underutil_details_%d_%d_%.2f.pdf" % (algo, ov, seed)
                    utils.export(fig, filename, folder='dts_compare/underutil')
                    plt.close()
    #exit()


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
    # Overutilization
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.tight_layout(pad=2.7)
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(r'Overutilization (\%)', fontsize=15)
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

            overutil_arr_sum = []
            overutil_arr = []
            for seedswitch, run in DATA1.items():
                seed, switch = seedswitch
                if seed in ignore_seeds:
                    continue
                if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                    continue

                thresh = run.get('scenario_table_capacity')
                d2 = 'dts_%d_table_datay_raw' % (switch)
                d3 = 'dts_%d_table_datay' % (switch)
                if param_dts_algo == 3:
                    if run.get(d3):
                        data = [x for x in run.get(d3) if x > thresh]
                        if len(data) > 0:
                            overutil_arr_sum.append((sum(data), seed, switch, run))

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

                overutil_arr.append((overutil, seed, switch, run))

            if len(overutil_arr) > 0:
                overutil_arr = sorted(overutil_arr)
                data = json.dumps(overutil_arr[-10:])
                utils.export_textfile(data, 'overutil_%d.json' % param_dts_algo, folder='dts_compare/overutil')  

            if len(overutil_arr_sum) > 0:
                overutil_arr_sum = sorted(overutil_arr_sum)
                data = json.dumps(overutil_arr_sum[-10:])
                utils.export_textfile(data, 'overutil_sum_%d.json' % param_dts_algo, folder='dts_compare/overutil')        

            utils.plotcdf(ax, result_overutil_percent, label=label, marker=marker, markevery=20, linewidth=1.5, color=color)
            #utils.plotcdf(axes[1], result_underutil_percent, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)

            if param_dts_algo == 3:
                x = overutil_arr[-1][0]
                y = 1
                ax.vlines(x, y, y-0.01, color='black', linestyle=':', linewidth=2, alpha=1)
                ax.text(x, y-0.01, r'\noindent Scenario $z_{122314}$ \\with  overutilization ' + ('%.2f' % x) + '\\%%',
                    fontsize=18, fontweight='normal', color="black", va='center', ha="right", 
                    bbox=dict(boxstyle="square", ec='white', fc='white',)
                )

                x = overutil_arr_sum[-1][0]
                for _overutil, _seed, _switch, _run in sorted(overutil_arr):
                    if _seed == 170154:
                        x = _overutil
                        break;

                ax.hlines(0.968, x, x+1, color='black', linestyle=':', linewidth=2, alpha=1)
                ax.text(x+1, 0.968, r'\noindent Scenario $z_{170154}$ \\with  overutilization ' + ('0.24') + '\\%%',
                    fontsize=18, fontweight='normal', color="black", va='center', ha="left", 
                    bbox=dict(boxstyle="square", ec='white', fc='white',)
                )

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
        fig.subplots_adjust(left=0.1, top=0.9) # padding top
        utils.export(fig, 'dts_compare_overutil.pdf', folder='dts_compare')
        plt.close()

    # -----------------------
    # Underutilization
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.tight_layout(pad=2.7)
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(r'Underutilization (\%)', fontsize=15)
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

            underutil_skipped = 0
            underutil_arr = []
            for seedswitch, run in DATA1.items():
                seed, switch = seedswitch
                #if run.get('scenario_switch_cnt') == 2: continue
                if seed in ignore_seeds:
                    underutil_skipped += 1
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
                if run.get('scenario_table_capacity_reduction') <= 70:
                    underutil_arr.append((underutil, seed, switch, run))


            if len(underutil_arr) > 0:
                underutil_arr = sorted(underutil_arr)
                mid = int(len(underutil_arr)/2)
                data = json.dumps(underutil_arr[-5:] + underutil_arr[:5] + underutil_arr[mid-5:mid+5])
                utils.export_textfile(data, 'underutil_%d.json' % param_dts_algo, folder='dts_compare/underutil')  

            utils.plotcdf(ax, result_underutil_percent, label=label, marker=marker, markevery=500, linewidth=1.5, color=color)
            print("len", len(result_underutil_percent), underutil_skipped)
            for perc in [50,80,90,99]:
                p = np.percentile(result_underutil_percent, perc)
                print("underutil", param_dts_algo, perc, "   ", p)

            """
            if param_dts_algo == 3:
                x = underutil_arr[-1][0]
                y = 1
                ax.vlines(x, y, y-0.01, color='black', linestyle=':', linewidth=2, alpha=1)
                ax.text(x, y-0.01, r'\noindent Scenario $z_{122314}$ \\with  overutilization ' + ('%.2f' % x) + '\\%%',
                    fontsize=18, fontweight='normal', color="black", va='center', ha="right", 
                    bbox=dict(boxstyle="square", ec='white', fc='white',)
                )

                x = underutil_arr_sum[-1][0]
                for _overutil, _seed, _switch, _run in sorted(underutil_arr):
                    if _seed == 170154:
                        x = _overutil
                        break;

                ax.hlines(0.968, x, x+1, color='black', linestyle=':', linewidth=2, alpha=1)
                ax.text(x+1, 0.968, r'\noindent Scenario $z_{170154}$ \\with  overutilization ' + ('0.24') + '\\%%',
                    fontsize=18, fontweight='normal', color="black", va='center', ha="left", 
                    bbox=dict(boxstyle="square", ec='white', fc='white',)
                )
            """

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14)
        fig.subplots_adjust(left=0.1, top=0.85) # padding top
        utils.export(fig, 'dts_compare_underutil.pdf', folder='dts_compare')
        plt.close()
