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

NUMBER_OF_CLASSES = 3

SETS = [(0,20), (20,30), (30,40), (40,50), (50,60), (60, 70), (70,100)]

ALGO = 2

DATASET_TYPE = 0

def plot(blob, **kwargs):
    "Plot dts scores (overutil, underutil, overheads)"

    utils.EXPORT_BLOB = blob
   
    if 'D5000' in blob.path:
        logger.info('set DATASET_TYPE to D5000')
        DATASET_TYPE = 5000
 
    if 'D5050' in blob.path:
        logger.info('set DATASET_TYPE to D5050_runtime')
        DATASET_TYPE = 1
        assert(ALGO == 2)

    includes = ['scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch', 'scenario_table_util_avg_total', 'scenario_table_util_max_total',
        'rsa_solver_jobs', 'rsa_solver_stats_itercount', 'rsa_solver_stats_nodecount']

    includes += blob.find_columns('hit_timelimit')
    includes += blob.find_columns('solver_stats_time_modeling')
    includes += blob.find_columns('solver_stats_time_solving')
    includes += blob.find_columns('solver_cnt_feasable')
    includes += blob.find_columns('solver_stats_itercount')
    includes += blob.find_columns('solver_cnt_infeasable')
    includes += blob.find_columns('solver_considered_ports')
    includes += blob.find_columns('table_datay_raw')
    includes += blob.find_columns('overutil_max')
    includes += blob.find_columns('solver_stats_time_modeling_other')
    includes += blob.find_columns('solver_stats_time_modeling_demand')


    blob.include_parameters(**dict.fromkeys(includes, 1))

    runs = blob.filter(**dict())

    # -----------------------
    # prepare data for plotting
    # -----------------------
    DATA = {}
    seeds = []
    ratios= []
    runs_by_seed = {}
    ignore_seeds = []
    infeasible = 0
    switch_cnt_total = 0
    time_modeling_other = 0
    switches_with_utilization = 0
    switches_with_bottlenecks = 0
    bottleneck_sizes = []
    for run in runs:
        seed = run.get('param_topo_seed')
        if run.get('hit_timelimit'):
            ignore_seeds.append(seed)
            continue
        param_dts_algo = run.get('param_dts_algo')
        param_dts_look_ahead = run.get('param_dts_look_ahead')

        if param_dts_algo != ALGO: continue
        if not seed in seeds:
            seeds.append(seed)
        if not DATA.get(param_dts_algo):
            DATA[param_dts_algo] = {}


        #util_avg = run.get('scenario_table_util_avg_total')
        #util_max = run.get('scenario_table_util_max_total')
        #ratio = util_max
        #ratios_solving_time.append((ratio, seed)) 
        switch_cnt_total += run.get('scenario_switch_cnt')
        try:
            runs_by_seed[seed].append(run)
        except KeyError:
            runs_by_seed[seed] = [run]


        util_values = []
        util_max = []
        use_switches = []
        for switch in range(0, run.get('scenario_switch_cnt')):
            d2 = 'dts_%d_table_datay_raw' % (switch)
            thresh = run.get('scenario_table_capacity')
            if run.get(d2):
                data = [x for x in run.get(d2) if x > 0]
                if len(data) > 0:
                    switches_with_utilization += 1
                    util_values.append(sum(data)/float(len(data)))
                    util_max.append(max(data))
                    bottlenecks = [1 for y in data if y > thresh]
                    if len(bottlenecks) > 0:
                        switches_with_bottlenecks += 1
                        bottleneck_sizes.append(sum(bottlenecks))
                        use_switches.append(switch)


        if run.get('dts_%d_solver_stats_time_modeling_other' % (switch)):
            time_modeling_other += 1

        if len(util_values) > 0:
            util_avg = sum(util_values)/float(len(util_values))
            util_max = max(util_max)
            ratio = (float(util_avg) / float(util_max))*100.0
            for switch in use_switches:
                ratios.append((ratio, param_dts_algo, switch, run))


    print("ignore_seeds", len(ignore_seeds))
    print("infeasible", infeasible)
    print("switch_cnt_total", switch_cnt_total)
    print("switches_with_utilization", switches_with_utilization)
    print("switches_with_bottlenecks", switches_with_bottlenecks)
    print("bottleneck_sizes_sum", sum(bottleneck_sizes))
    print("time_modeling_other", time_modeling_other)



    def set_barriers(ax, ranges, alldata, cat=0, xlim=None):
        RANGES = ranges

        max_x = max(alldata)
        if max_x > max(ranges):
            ranges.append(math.floor(max_x)+1)
        else:
            ranges = list(filter(lambda e: e/2 < max_x, ranges))

        xticks = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1] + [2**x for x in range(10) if 2**x < ranges[-1]]

        xlow = 0.02
        if cat == 1:
            xlow = 0.5
            xticks = [0.5, 1] + [2**x for x in range(10) if 2**x < ranges[-1]]
        if cat == 2:
            xlow = 0.25
            xticks = [0.25,0.5, 1] + [2**x for x in range(10) if 2**x < ranges[-1]]    
        if cat == 20:
            xlow = 2
            xticks = [2**(x+1) for x in range(12) if 2**x < ranges[-1]]   
        if cat == 21:
            xlow = 0.125/8.
            xticks = [0.125, 0.5, 1] + [2**x for x in range(10) if 2**x < ranges[-1]]         
        if cat == 4:
            xlow = 0.125
            xticks = [0.125,0.5, 1] + [2**x for x in range(10) if 2**x < ranges[-1]]         
        if cat == 5:
            xlow = 0.125/8.
            xticks = [0.125/8.,0.5, 1] + [2**x for x in range(10) if 2**x < ranges[-1]]         
        if cat == 30:
            xlow = 0.125/8.
            xticks = [0.125/8.,0.125/4,0.125,0.5, 1,2,8,32]        
        if cat == 31:
            xlow = 0.125/4.
            xticks = [0.125/4., 0.125,0.5, 1,2,8,32,128]        

        if xlim:
            if xticks[-1] > xlim[-1]:
                xticks[-1] =  xlim[-1]
            ax.set_xlim(xlow,ranges[-1]+sum(ranges[:-1]))
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(xlow,ranges[-1]+sum(ranges[:-1]))
        ax.set_ylim(0,1.15)
        ax.set_xlim(xlim)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_ylabel('CDF', fontsize=15)
        ax.set_xscale('log')
        ax.set_xticks(xticks, [])
        ax.minorticks_off()
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax.legend(loc='lower right', fontsize=15)

        limits = []
        for k, upper in enumerate(ranges):
            if xlim:
                if upper > xlim[-1]: break;
            lower = 0
            if k > 0: lower = sum(ranges[:k])
            limit = lower+upper
            if k == len(ranges)-1:
                limit = max_x
            limits.append(limit)
            limittext = ('%1.1f' % limit)
            if limit == 0.125:
                limittext = '%1.3f' % limit
            if limit >= 1:
                limittext = '%d' % int(limit)
            ax.text(limit/2, 1.09, r'\textbf{' + r'$<$' + limittext + r'ms}', color='black', fontsize=14,
                horizontalalignment='center', bbox=dict(facecolor='white', edgecolor='white'))  
            text = r'\noindent '
            """
            for id, data in sorted(setdata):
                selected = list(filter(lambda x: x < limit, data))
                percent = (len(selected)/float(len(data)))*100.0
                text += ('Set %d: ' % id) + ('%2.2f' % (percent)) + r'\% \\'
            """
            selected = list(filter(lambda x: x < limit, alldata))
            percent = (len(selected)/float(len(alldata)))*100.0
            text += ('%2.2f' % (percent)) + r'\%'
            ax.text(limit/2, 1.03, text,
                horizontalalignment='center',  #bbox=dict(facecolor='white', edgecolor='white')
            )

        for x in xticks:
            if x in limits:
                ax.vlines(x, 0, 1.25, color='black', linestyle='--', linewidth=1.5, alpha=1)
            else:
                ax.vlines(x, 0, 1, color='grey', linestyle='--', linewidth=1, alpha=0.3)

    def plot_set(set, param_dts_algo):
        solver2 = []
        result_solver_stats_time_solving = []
        result_solver_stats_time_modeling = []
        result_solver_stats_time_modeling_other = []
        counter = 0
        for ratio, param_dts_algo, switch, run in set:
            if run.get('dts_%d_solver_cnt_infeasable' % (switch)) == 0:
                solver_stats_time_solving = run.get('dts_%d_solver_stats_time_solving' % (switch))
                solver_stats_time_modeling = run.get('dts_%d_solver_stats_time_modeling' % (switch))

                solver_stats_time_modeling_other = run.get('dts_%d_solver_stats_time_modeling_other' % (switch))
                

                if solver_stats_time_solving == None:
                    continue
                if solver_stats_time_modeling == None:
                    continue

                if ALGO < 3:
                    solver2 += [x * 1000.0 for x, y in zip(run.get('rsa_solver_stats_time_solving'), 
                        run.get('dts_%d_solver_stats_itercount' % (switch))) if x > 0 and y > 0]


                result_solver_stats_time_solving += [x * 1000.0 for x in solver_stats_time_solving if x > 0]
                result_solver_stats_time_modeling += [x * 1000.0 for x in solver_stats_time_modeling if x > 0]   
                if solver_stats_time_modeling_other != None:
                    result_solver_stats_time_modeling_other += [x * 1000.0 for x in solver_stats_time_modeling_other if x > 0]  
                counter += 1

        return result_solver_stats_time_modeling, result_solver_stats_time_modeling_other, result_solver_stats_time_solving, solver2


    #--------------------
    # calculate sets with different flow table utilization ratios
    #-------------------- 
    sets = {}
    for x, y in SETS:
        sets[(x,y)] = []

    set10 = 0
    for ratio, param_dts_algo, switch, run in ratios:
        assigned = 0
        for x, y in SETS:
            if ratio >= x and ratio < y:
                sets[(x,y)].append((ratio, param_dts_algo, switch, run))
                assigned += 1
        assert(assigned == 1)
        if ratio >= 0 and ratio < 10:
            set10 += 1

    print("set10", set10)


    if DATASET_TYPE == 5000:
        folder = 'runtime/%d-%d' % (DATASET_TYPE, ALGO)

    if DATASET_TYPE == 1:
        folder = 'runtime'  
    # -----------------------
    # Solving time
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.tight_layout(pad=3)
        alldata = []
        setcnt = 1
        for key, set in sorted(sets.items()):
            modeling, modeling2, solving, solving2= plot_set(set, 2)
            ratio1, _, _, _ = set[0]
            ratio2, _, _, _ = set[-1]
            utils.plotcdf(ax, solving, alpha=0.8, linewidth=1, label='c%d (%d samples)' % (
                setcnt, len(solving)))
            alldata += solving   
            setcnt += 1
        utils.plotcdf(ax, alldata, color="black", linewidth=3, label='all %d samples' % (len(alldata)))
        if ALGO == 3:
            set_barriers(ax, [0.125], alldata, cat=30)
            ax.set_xlabel('Solving time Select-Greedy (ms)', fontsize=15)
        if ALGO == 2:
            set_barriers(ax, [0.125, 0.5-0.125, 1.5, 6], alldata, cat=0, xlim=(0.03, 32))
            ax.set_xlabel('Solving time Select-CopyFirst (ms)', fontsize=15)
        if ALGO == 1:
            set_barriers(ax, [2,6,24, 128-32], alldata, cat=1)
            ax.set_xlabel('Solving time Select-Opt (ms)', fontsize=15)
        ax.legend(loc='lower right', fontsize=15)
        utils.export(fig, 'runtime_%d_solving.pdf' % ALGO, folder = folder)
        plt.close()

    # -----------------------
    # Solving time 2
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.tight_layout(pad=3)
        alldata = []
        setcnt = 1
        for key, set in sorted(sets.items()):
            modeling, modeling2, solving, solving2= plot_set(set, 2)
            ratio1, _, _, _ = set[0]
            ratio2, _, _, _ = set[-1]
            utils.plotcdf(ax, solving2, alpha=0.8, linewidth=1, label='c%d (%d samples)' % (
                setcnt, len(solving2)))
            alldata += solving2   
            setcnt += 1
        utils.plotcdf(ax, alldata, color="black", linewidth=3, label='all %d samples' % (len(alldata)))
        if ALGO == 2:
            set_barriers(ax, [0.125, 0.5-0.125, 1.5, 6], alldata, cat=0, xlim=(0.03, 32))
            ax.set_xlabel('Solving time Select-CopyFirst (ms)', fontsize=15)
        if ALGO == 1:
            set_barriers(ax, [0.125, 0.5-0.125, 1.5,6,24, 128-32], alldata, cat=21)
            ax.set_xlabel('Solving time Select-Opt (ms)', fontsize=15)
        ax.legend(loc='lower right', fontsize=15)
        utils.export(fig, 'runtime_%d_solving2.pdf' % ALGO, folder = folder)
        plt.close()

    # -----------------------
    # Modeling time
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.tight_layout(pad=3)
        alldata = []
        setcnt = 1
        for key, set in sorted(sets.items()):
            modeling, modeling2, solving, solving2 = plot_set(set, 2)
            ratio1, _, _, _ = set[0]
            ratio2, _, _, _ = set[-1]
            utils.plotcdf(ax, modeling, alpha=0.8, linewidth=1, label='c%d (%d samples)' % (
                setcnt, len(modeling)))
            alldata += modeling   
            setcnt += 1
        utils.plotcdf(ax, alldata, color="black", linewidth=3, label='all %d samples' % (len(alldata)))
        if ALGO == 3:
            set_barriers(ax, [0.125, 0.5-0.125, 1.5, 6, 24], alldata, cat=31)
            ax.set_xlabel('Modeling time Select-Greedy (ms)', fontsize=15)
        if ALGO == 2:
            set_barriers(ax, [2,6,24, 128-32], alldata, cat=1)
            ax.set_xlabel('Modeling time Select-CopyFirst (ms)', fontsize=15)
        if ALGO == 1:
            set_barriers(ax, [8, 32-8, 128-32, 512-128,2048-512], alldata, cat=20)
            ax.set_xlabel('Modeling time Select-Opt (ms)', fontsize=15)
        ax.legend(loc='lower right', fontsize=15)
        utils.export(fig, 'runtime_%d_modeling.pdf' % ALGO, folder = folder)
        plt.close()

    # -----------------------
    # Modeling time 2
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.tight_layout(pad=3)
        alldata = []
        setcnt = 1
        for key, set in sorted(sets.items()):
            modeling, modeling2, solving, solving2 = plot_set(set, 2)
            ratio1, _, _, _ = set[0]
            ratio2, _, _, _ = set[-1]
            utils.plotcdf(ax, modeling2, alpha=0.8, linewidth=1, label='c%d (%d samples)' % (
                setcnt, len(modeling2)))
            alldata += modeling2   
            setcnt += 1
        if len(alldata) > 0:
            utils.plotcdf(ax, alldata, color="black", linewidth=3, label='all %d samples' % (len(alldata)))
            set_barriers(ax, [2,6,24], alldata, cat=1)
            ax.legend(loc='lower right', fontsize=15)
            ax.set_xlabel('Modeling time Select-CopyFirst without link overhead coefficients (ms)', fontsize=15)
            utils.export(fig, 'runtime_2_modeling2.pdf', folder = folder)
            plt.close()

    # -----------------------
    # Solving time RSA
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.tight_layout(pad=3)
        alldata = []
        setcnt = 1
        
        for key, set in sorted(sets.items()):
            modeling = []
            solving = []
            used_seeds = []
            for ratio, param_dts_algo, switch, run in set:
                seed = run.get('param_topo_seed')
                if seed in used_seeds:
                    # RSA part is not separate per switch 
                    continue
                used_seeds.append(seed)
                solving += [x * 1000.0 for x, y in zip(run.get('rsa_solver_stats_time_solving'), 
                    run.get('rsa_solver_stats_itercount')) if x > 0 and y > 0]
                modeling += [x * 1000.0 for x in run.get('rsa_solver_stats_time_modeling') if x > 0]   
            utils.plotcdf(ax, solving, alpha=0.8, linewidth=1, label='c%d (%d samples)' % (
                setcnt, len(solving)))
            alldata += solving   
            setcnt += 1
        utils.plotcdf(ax, alldata, color="black", linewidth=3, label='all %d samples' % (len(alldata)))
        set_barriers(ax, [2,6,24, 128-32,512-128], alldata, cat=2)
        ax.legend(loc='lower right', fontsize=15)
        ax.set_xlabel('Solving time RS-Alloc (ms)', fontsize=15)
        utils.export(fig, 'rsa_runtime_solving.pdf', folder = folder)
        plt.close()

    # -----------------------
    # Modeling time RSA
    # -----------------------
    if 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.tight_layout(pad=3)
        alldata = []
        setcnt = 1
        
        for key, set in sorted(sets.items()):
            modeling = []
            used_seeds = []
            for ratio, param_dts_algo, switch, run in set:
                seed = run.get('param_topo_seed')
                if seed in used_seeds:
                    # RSA part is not separate per switch 
                    continue
                used_seeds.append(seed)
                modeling += [x * 1000.0 for x, y in zip(run.get('rsa_solver_stats_time_modeling'), 
                    run.get('rsa_solver_stats_itercount')) if x > 0 and y > 0]
            utils.plotcdf(ax, modeling, alpha=0.8, linewidth=1, label='c%d (%d samples)' % (
                setcnt, len(modeling)))
            alldata += modeling   
            setcnt += 1
        utils.plotcdf(ax, alldata, color="black", linewidth=3, label='all %d samples' % (len(alldata)))
        set_barriers(ax, [2,6,24, 128-32], alldata, cat=4)
        ax.legend(loc='lower right', fontsize=15)
        ax.set_xlabel('Modeling time RS-Alloc (ms)', fontsize=15)
        utils.export(fig, 'rsa_runtime_modeling.pdf', folder = folder)
        plt.close()

    exit()



















    # -----------------------
    # old
    # -----------------------
    if True:
        for param_dts_algo, DATA1 in DATA.items():
            color = 'red'
            label = 'Select-Opt'
            if param_dts_algo == 2:
                color = 'blue'
                label = 'Select-CopyFirst'

            maxv = []
            result_solver_stats_time_solving = []
            result_solver_stats_time_modeling = []
            for seedswitch, run in DATA1.items():
                seed, switch = seedswitch
                if run.get('dts_%d_solver_cnt_infeasable' % (switch)) == 0:

                    solver_stats_time_solving = run.get('dts_%d_solver_stats_time_solving' % (switch))
                    solver_stats_time_modeling = run.get('dts_%d_solver_stats_time_modeling' % (switch))

                    if solver_stats_time_solving == None:
                        continue
                    if solver_stats_time_modeling == None:
                        continue

                    maxv.append((max(solver_stats_time_modeling), seed))
                    
                    result_solver_stats_time_solving += [x * 1000.0 for x in solver_stats_time_solving]
                    result_solver_stats_time_modeling += [x * 1000.0 for x in solver_stats_time_modeling]
                    """
                    solver_stats_time_solving = statistics.mean(solver_stats_time_solving) * 1000
                    solver_stats_time_modeling = statistics.mean(solver_stats_time_modeling) * 1000

                    try:
                        result_solver_stats_time_solving.append(solver_stats_time_solving)
                    except KeyError:
                        result_solver_stats_time_solving = [solver_stats_time_solving]

                    try:
                        result_solver_stats_time_modeling.append(solver_stats_time_modeling)
                    except KeyError:
                        result_solver_stats_time_modeling = [solver_stats_time_modeling]
                    """


            for cat, sets, ranges in zip([0,1], [sets_solving, sets_modeling], [[0.125, 0.5-0.125, 1.5, 6], [2,6,24, 128-32]]):
                plt.close()
                fig, ax = plt.subplots(figsize=(12, 7))
                fig.tight_layout(pad=3)
                #ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                #ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5) 

                alldata = []
                setdata = []
                for setcnt, set in enumerate(sets):
                    modeling, modeling2, solving, solving2 = plot_set(set, param_dts_algo)
                    ratio1, _, _, _ = set[0]
                    ratio2, _, _, _ = set[-1]
                    if cat == 0:
                        utils.plotcdf(ax, solving, alpha=0.8, linewidth=1, label='[Set %d] max. overutilization: %d - %d (%d samples)' % (
                            setcnt+1, ratio1, ratio2, len(solving)))
                        alldata += solving   
                        setdata.append((setcnt+1, solving))                     
                    if cat == 1:
                        utils.plotcdf(ax, modeling, alpha=0.8, linewidth=1, label='[Set %d] max. capacity: %d - %d (%d samples)' % (
                            setcnt+1, ratio1, ratio2, len(solving))) 
                        alldata += modeling   
                        setdata.append((setcnt+1, modeling))     

                print(">", cat, len(alldata))

                print(">> ", sum([1 for x in alldata if x < 0.12])) 

                utils.plotcdf(ax, alldata, color="black", linewidth=3, label='all %d samples' % (len(alldata)))

                RANGES = ranges

                max_x = max(alldata)
                if max_x > max(RANGES):
                    RANGES.append(math.floor(max_x)+1)
                else:
                    RANGES = list(filter(lambda e: e/2 < max_x, RANGES))

                xticks = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1] + [2**x for x in range(10) if 2**x < RANGES[-1]]

                xlow = 0.02
                if cat == 1:
                    xlow = 0.5
                    xticks = [0.5, 1] + [2**x for x in range(10) if 2**x < RANGES[-1]]
                ax.set_xlim(xlow,RANGES[-1]+sum(RANGES[:-1]))

                ax.set_ylim(0,1.25)
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.set_ylabel('CDF', fontsize=15)
                ax.set_xscale('log')
                ax.set_xticks(xticks, [])
                ax.minorticks_off()
                if cat == 0:
                    ax.set_xlabel('Solving Time (ms)', fontsize=15)
                else:
                    ax.set_xlabel('Modeling Time (ms)', fontsize=15)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
                ax.legend(loc='lower right', fontsize=15)

                limits = []
                for k, upper in enumerate(RANGES):
                    lower = 0
                    if k > 0: lower = sum(RANGES[:k])
                    limit = lower+upper
                    limits.append(limit)
                    ax.text(limit/2, 1.2, r'\textbf{' + r'$<$' + ('%1.1f' % limit) + r'ms}', color='black', fontsize=13,
                        horizontalalignment='center', bbox=dict(facecolor='white', edgecolor='white'))  
                    text = r'\noindent '
                    for id, data in sorted(setdata):
                        selected = list(filter(lambda x: x < limit, data))
                        percent = (len(selected)/float(len(data)))*100.0
                        text += ('Set %d: ' % id) + ('%2.2f' % (percent)) + r'\% \\'
                    selected = list(filter(lambda x: x < limit, alldata))
                    percent = (len(selected)/float(len(alldata)))*100.0
                    text += r'\textbf{' + ('all: %2.2f' % (percent)) + r'\%}'
                    ax.text(limit/2, 1.03, text,
                        horizontalalignment='center',
                        bbox=dict(facecolor='white', edgecolor='white'))

                for x in xticks:
                    if x in limits:
                        ax.vlines(x, 0, 1.25, color='black', linestyle='--', linewidth=1.5, alpha=1)
                    else:
                        ax.vlines(x, 0, 1, color='grey', linestyle='--', linewidth=1, alpha=0.3)

                #h1, l1 = ax.get_legend_handles_labels()
                #fig.legend(h1, l1, loc='upper center', ncol=2, fontsize=16)
                #fig.subplots_adjust(top=0.90) 
                utils.export(fig, 'runtime_dts_%d.pdf' % cat, folder='runtime')


            """
            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_solver_stats_time_solving.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[param_dts_algo-1][0].boxplot(result_solver_stats_time_solving.values(), positions=datax, notch=True, showfliers=False)
            axes[param_dts_algo-1][0].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_solver_stats_time_modeling.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[param_dts_algo-1][1].boxplot(result_solver_stats_time_modeling.values(), positions=datax, notch=True, showfliers=False)
            axes[param_dts_algo-1][1].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)
            """


        exit(1)



