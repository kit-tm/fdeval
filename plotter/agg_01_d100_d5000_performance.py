import logging, math, json, pickle, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import time
import importlib  
import statistics
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
matplotlib.rcParams.update(params)

from . import agg_2_utils as utils


logger = logging.getLogger(__name__)

# change these 
NUMBER_OF_CLASSES = 5
RELATIVE_FAILURE = False
BOXPLOT = False

# more or less static parameters
QUANTILES_EXTENDED = [0.5, 0.8, 0.9, 0.95, 0.99, 1]
QUANTILES = [0.5, 0.9]
HIGHLIGHT_QUANTILE_RED = 0.9
HIGHLIGHT_QUANTILE_BLUE = 0.5
DATASET_TYPE = -1 # 5000 or 100; determined by folder name
CUTOFF_THRESHOLD = 0 # exclude all scenarios with higher threshold / capacity reduction (used with 5000er only)
FIGSIZE = (10, 14)
if NUMBER_OF_CLASSES == 1:
    FIGSIZE = (10, 6)
if BOXPLOT:
    FIGSIZE = (10, 10)  
    if NUMBER_OF_CLASSES == 1:
        FIGSIZE = (10, 4)


# classes for flow table utilization
SETS = [(0,20), (20,30), (30,40), (40,50), (50,60), (60, 70), (70,100)]
#SETS = [(0,100)]
# x-values for the "Achieved capacity reduction" plot
EXPECTED_REDUCTION = [0, 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
TITLE_UTIL_RATIO = False
TITLE_ALLOWED_FAILURE_RATE = True



def plot(blob, **kwargs):
    """
    RSA box plots 
        x = reduced capacity in %
        y = failure rate (rules that couldn't be handled by flow delegation)

    """
    if 'D100' in blob.path:
        logger.info('set DATASET_TYPE to D100')
        DATASET_TYPE = 100

    if 'D5000' in blob.path:
        logger.info('set DATASET_TYPE to D5000')
        DATASET_TYPE = 5000


    utils.EXPORT_BLOB = blob
   

    assert(DATASET_TYPE == 100 or DATASET_TYPE == 5000)

    includes = ['scenario_switch_cnt', 
        'scenario_table_capacity',
        'scenario_gen_param_topo_switch_capacity', 
        'scenario_table_util_avg_total', 
        'scenario_table_util_max_total',   
        'rsa_table_percent_relocated', 
        'rsa_table_percent_shared',
        'rsa_table_percent_over_capacity'
    ]

    includes += blob.find_columns('hit_timelimit')
    includes += blob.find_columns('table_datay_raw')


    blob.include_parameters(**dict.fromkeys(includes, 1))

    runs = blob.filter(**dict())

    use_runs = []
    runs_by_seed = {}
    ratios = []
    seeds = []
    ignored_seeds = []
    skipped = 0
    timelimit = 0
    infeasible = 0
    for run in runs:
        seed = run.get('param_topo_seed')
        if run.get('param_dts_algo') != 2:
            skipped += 1
            continue 
        if run.get('hit_timelimit'):
            skipped += 1
            timelimit += 1
            if not seed in ignored_seeds:
                ignored_seeds.append(seed)
            continue
        if run.get('rsa_solver_cnt_infeasable', 0) > 0:
            skipped += 1
            infeasible += 1
            if not seed in ignored_seeds:
                ignored_seeds.append(seed)
            continue

        #if DATASET_TYPE == 5000:
        #    if CUTOFF_THRESHOLD > 0:
        #        if 100-run.get('scenario_gen_param_topo_switch_capacity') > CUTOFF_THRESHOLD:
        #            skipped += 1
        #            continue     

        if not seed in seeds:
            seeds.append(seed)

        try:
            runs_by_seed[seed].append(run)
        except KeyError:
            runs_by_seed[seed] = [run]

        # the ratio between average and maximum table utilization for 
        # the case with "almost no reduction" is used as a reference value to
        # create different load sets

        if DATASET_TYPE == 5000:
            # in this dataset, there is only a single run per seed
            #util_avg = run.get('scenario_table_util_avg_total')
            #util_max = run.get('scenario_table_util_max_total')
            #ratio = util_avg / util_max

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
                ratio = (float(util_avg) / float(util_max))*100.0

                ratios.append((ratio, seed))        

        if DATASET_TYPE == 100:
            # in this dataset, each seed was executed 80 times with different threshold
            if run.get('scenario_gen_param_topo_switch_capacity') == 99:    
                #util_avg = run.get('scenario_table_util_avg_total')
                #util_max = run.get('scenario_table_util_max_total')
                #ratio = util_avg / util_max

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
                    ratio = (float(util_avg) / float(util_max))*100.0


                    ratios.append((ratio, seed))


    print("runs", len(runs))
    print("timelimit", timelimit)
    print("infeasible", infeasible)
    print("skipped", skipped)
    print("seeds", len(seeds))
    print("ignored", len(ignored_seeds))
    print("ratio", min([x for x, _ in ratios]), statistics.mean([x for x, _ in ratios]), max([x for x, _ in ratios]))
    print("ratio_cnt", len(ratios))


    if DATASET_TYPE == 100:
        assert(len(seeds) == len(ratios))


    ratios = sorted(ratios, key=lambda x: x[0])


    #--------------------
    # print operational range plot from json file
    #-------------------- 

    fig, axes = plt.subplots(2,2, figsize=(10, 10), sharex=False, sharey=False)
    fig.tight_layout(pad=3) 

    fig.axes[3].get_xaxis().set_visible(False)
    fig.axes[3].get_yaxis().set_visible(False)
    fig.axes[3].spines['top'].set_visible(False)
    fig.axes[3].spines['right'].set_visible(False)
    fig.axes[3].spines['bottom'].set_visible(False)
    fig.axes[3].spines['left'].set_visible(False) 


    #--------------------
    # Flow delegation performance
    #-------------------- 
    if 0:
        cnt = 0
        for rate, f in zip([0, 0.1, 1], ['json_operational_range0', 'json_operational_range01', 'json_operational_range1']):
            ax = fig.axes[cnt]
            filepath = os.path.join(utils.EXPORT_FOLDER, 'flow-delegation-performance', f)
            data0q90 = {0:0, 20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
            data0q50 = {0:0, 20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
            with open(filepath, 'r') as file:
                data = json.loads(file.read())
                for k, row in sorted(data.items()):
                    x, y, q, operational_range = row
                    print(x, y, q, operational_range)
                    if q == 0.9:
                        data0q90[x] = operational_range
                    if q == 0.5:
                        data0q50[x] = operational_range

            ax.text(0.9, 0.9, r'\textbf{Failure rate = ' + '%.2f' % rate + r'\%}',
                fontsize=16,  transform=ax.transAxes,
                verticalalignment='center', horizontalalignment='right',
                color='black', alpha=1,
                bbox=dict(facecolor='white', edgecolor='white')
            )
            ax.plot(np.arange(len(data0q90)), data0q90.values(), c='red', marker='*', label='90th percentile')
            ax.plot(np.arange(len(data0q50)), data0q50.values(), c='blue', marker='s', label='50th percentile')
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel(r'Utilization ratio class', fontsize=16, labelpad=10)
            ax.set_xticks([0,1,2,3,4,5,6])
            ax.set_ylim(0,60)
            ax.set_xticklabels(['c1', 'c2','c3','c4','c5','c6','c7'])
            #ax.set_xlim(0,6)
            if cnt == 0 or cnt == 2:
                ax.set_ylabel(r'Operational range (reduction)', fontsize=16, labelpad=10)
            cnt += 1

        handles, labels = ax.get_legend_handles_labels()
        fig.axes[3].legend(handles, labels, loc='upper left', ncol=1, fontsize=20)


        filename = "ds_%d_operational_range.pdf" % (DATASET_TYPE)   
        utils.export(fig, filename, folder='flow-delegation-performance')
        plt.close()



    #--------------------
    # calculate sets with different flow table utilization ratios
    #-------------------- 
    sets = {}
    for x, y in SETS:
        sets[(x,y)] = []

    set10 = 0
    for ratio, seed in ratios:
        assigned = 0
        for x, y in SETS:
            if ratio >= x and ratio < y:
                sets[(x,y)].append((ratio, seed))
                assigned += 1
        assert(assigned == 1)
        if ratio >= 0 and ratio < 10:
            set10 += 1
    print("set10", set10)
    

    #--------------------
    # Reduction per ratio
    #--------------------
    text2 = ""
    for allowed_failures in [0, 0.1, 1]:
        text2 += "\n====================== allowed_failures = %.2f\n\n" % allowed_failures 
        markers = ['*', 'o', 'd', 's', '^', 'P', 'X']
        fig, ax = plt.subplots(figsize=(10, 6), sharex=False, sharey=True)
        fig.tight_layout(pad=3)
        text = ''
        total = 0
        cnt = 0
        for x, y in SETS:
            print(x, y, len(sets[(x,y)]))
            if len(sets[(x,y)]) == 0:
                continue
            data_x = []
            failure_rate = []
            for ratio, seed in sets[(x,y)]:
                runs = runs_by_seed.get(seed)
                for run in runs:
                    failed = run.get('rsa_table_percent_relocated')
                    percent_relocated = run.get('rsa_table_percent_relocated') / (run.get('rsa_table_percent_relocated')  + run.get('rsa_table_percent_shared')) * 100
                    if RELATIVE_FAILURE:
                        failed = percent_relocated   
                    fr = 100-run.get('scenario_gen_param_topo_switch_capacity') 
                    failure_rate.append((fr, failed))


            expected_cnt = {}
            expected_ratios = {}
            expected_0 = {}
            expected_01 = {}
            expected_1 = {}
            for ex in EXPECTED_REDUCTION:
                expected_cnt[ex] = 0
                expected_0[ex] = 0
                expected_01[ex] = 0
                expected_1[ex] = 0
            for v, fr in sorted(failure_rate):
                for ex in EXPECTED_REDUCTION:
                    if v >= ex:
                        expected_cnt[ex] += 1
                        try:
                            expected_ratios[ex].append(fr)
                        except:
                            expected_ratios[ex] = [fr]
                        if fr == 0:
                            expected_0[ex] += 1
                        if fr < 0.1:
                            expected_01[ex] += 1
                        if fr < 1:
                            expected_1[ex] += 1


            datax = []
            datay = []

            text2 += "\n====" + str((x,y)) + "\n"
            for ex in sorted(EXPECTED_REDUCTION):
                try:
                    #if expected_ratios.get(ex):
                    #    exr_0 = np.quantile(expected_ratios.get(ex), 0.9)
                    #else:
                    #    exr_0 = 0
                    exr_0 = expected_0[ex] / float(expected_cnt[ex]) * 100
                except ZeroDivisionError:
                    exr_0 = 0
                try:
                    exr_01 = expected_01[ex] / float(expected_cnt[ex]) * 100
                except ZeroDivisionError:
                    exr_01 = 0
                try:
                    exr_1 = expected_1[ex] / float(expected_cnt[ex]) * 100
                except ZeroDivisionError:
                    exr_1 = 0  

                datax.append(ex)
                if allowed_failures == 0:
                    datay.append(exr_0)
                    text2 += "[%.2f][%d] %.2f\n" % (allowed_failures, ex, exr_0)
                if allowed_failures == 0.1:
                    datay.append(exr_01)
                    text2 += "[%.2f][%d] %.2f\n" % (allowed_failures, ex,exr_01)
                if allowed_failures == 1:
                    datay.append(exr_1)
                    text2 += "[%.2f][%d] %.2f\n" % (allowed_failures,ex, exr_1)

            ax.plot(datax, datay, marker=markers.pop(), label=(r'c%d -- ' % (cnt+1)) + 'utilization ratio %d-%d' % (x,y))


            ratio_0 = sum([1 for x, failed in failure_rate if failed == 0]) / float(len(failure_rate)) * 100
            ratio_01 = sum([1 for x, failed in failure_rate if failed < 0.1]) / float(len(failure_rate)) * 100
            ratio_1 = sum([1 for x, failed in failure_rate if failed < 1]) / float(len(failure_rate)) * 100

            total += len(failure_rate)

            mean_ratio = statistics.mean([ratio for ratio, seed in sets[(x,y)]])
            #print("-->", x,y, ratio_0, ratio_01, ratio_1)
            text += '%d\\%%-%d\\%% & %.2f & %d & %.2f\\%% & %.2f\\%% & %.2f\\%%\\\\\n' % (x, y, 
                mean_ratio, len(failure_rate), ratio_0, ratio_01, ratio_1)

            cnt += 1
            
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        #ax.set_ylim(0,89)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel(r'Achieved capacity reduction (\%) ', fontsize=16,  labelpad=10)
        if allowed_failures == 0:
            ax.set_ylabel(r'Fraction of scenarios with 0\% failure rate', fontsize=16, labelpad=10)  
        else:
            ax.set_ylabel(('Fraction of scenarios with failure rate below %.2f' % allowed_failures) + r'\%', fontsize=16, labelpad=10) 
        ax.legend(fontsize=15)
        
        ax.set_title(('Allowed failure rate: %.2f' % allowed_failures) + r'\%', fontsize=18)
        #handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='center left',  bbox_to_anchor=(0.8,0.5), ncol=1, fontsize=12)
        #fig.subplots_adjust(hspace=0.3, right=0.8) # no gap
        filename = "ds_%d_%d.pdf" % (DATASET_TYPE, int(allowed_failures*100))   
        utils.export(fig, filename, folder='flow-delegation-performance')
        filename = "ds_%d_%d_table.txt" % (DATASET_TYPE, int(allowed_failures*100))   
        utils.export_textfile(text, filename, folder='flow-delegation-performance')
        plt.close()

    filename = "ds_%d_records.txt" % (DATASET_TYPE)  
    utils.export_textfile(text2, filename, folder='flow-delegation-performance')
    

    fig, ax = plt.subplots(figsize=(6, 4), sharex=False, sharey=True)
    fig.tight_layout(pad=3)
    boxratios = []
    ticklabels = []
    for i, key in enumerate(SETS):
        ticklabels.append('c%d' % (i+1))
        boxratios.append([r for r, s in sets[key]])

    ax.boxplot(boxratios)
    ax.set_xticklabels(ticklabels)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yticks([10,20,30,40,50,60,70,80,90])
    ax.set_xlabel('Considered classes of flow table utilization ratios', fontsize=16,  labelpad=10)
    ax.set_ylabel(r'Flow table utilization ratio (\%)', fontsize=16,  labelpad=10)
    ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)

    filename = "ds_%d_ratios.pdf" % (DATASET_TYPE)   
    utils.export(fig, filename, folder='flow-delegation-performance')   

    #exit(1)

    #--------------------
    # 2-stock plot
    #--------------------
    operational_range0 = {}
    operational_range01 = {}
    operational_range1 = {}
    for quantile_extension, use_quantiles in zip(['', '_extended'], [QUANTILES, QUANTILES_EXTENDED]):
        for allowed_failures in [0, 0.1, 1]:
            for key, set in sets.items():
                if len(set) == 0:
                    continue
                ratio_low, ratio_high = key
                plt.close()
                fig, ax = plt.subplots(figsize=(10, 5), sharex=False, sharey=True)
                fig.tight_layout(pad=2.7)

                alldata = {}
                used_ratios = []
                data_x = []
                data_y = []

                for ratio, seed in set:
                    used_ratios.append(ratio)
                    runs = runs_by_seed.get(seed)
                    for run in runs:
                        failed = run.get('rsa_table_percent_relocated')
                        percent_relocated = run.get('rsa_table_percent_relocated') / (run.get('rsa_table_percent_relocated')  + run.get('rsa_table_percent_shared')) * 100
                        if RELATIVE_FAILURE:
                            failed = percent_relocated   
                        x = 100-run.get('scenario_gen_param_topo_switch_capacity')
                        if not alldata.get(x):
                            alldata[x] = []   
                        alldata[x].append(failed)
                        data_y.append(failed)

                total = 0
                xvals = []
                boxdata = []
                histdata = []
                quantiles = {}
                for x, data in sorted(alldata.items()):
                    #print(x, len(data))
                    boxdata.append(data)
                    xvals.append(x)
                    histdata.append(len(data))
                    total += len(data)
                    for q in use_quantiles:
                        try:
                            quantiles[q].append(np.quantile(data, q)) # or np.percentile(data, q*100) which is the same
                        except KeyError:
                            quantiles[q] = [np.quantile(data, q)]

                ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_ylabel(r'Failure rate (in \%)', fontsize=16)
                ax.set_xlabel(r'Capacity reduction (in \%)', fontsize=16)

                # ----------------------- regular plot with percentiles
                for q in use_quantiles:
                    if q == HIGHLIGHT_QUANTILE_RED:     
                        if allowed_failures  == 0:                   
                            ax.plot(xvals, quantiles.get(q), color="red", marker='*', 
                                markevery=5, linewidth=1.5, label='%dth percentile' % (int(q*100)))
                            barrier_1 = allowed_failures
                            barrier_1_x = 1000
                            barrier_1_y = 0
                            for x, failed in zip(xvals, quantiles.get(q)):
                                if x < barrier_1_x and failed > barrier_1:
                                    barrier_1_x = x
                                    barrier_1_y = failed
                            if barrier_1_x > 1:
                                ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+20, color='red', linestyle=':', linewidth=2, alpha=1)
                                ax.text(barrier_1_x, barrier_1_y+20, r'\noindent \textbf{' + str(barrier_1_x) + r'\%}',
                                    fontsize=17, 
                                    verticalalignment='center', horizontalalignment='center',
                                    color='red', alpha=1,
                                    bbox=dict(facecolor='white', edgecolor='red')
                                )
                                operational_range0[str((ratio_low, ratio_high, q))] = [ratio_low, ratio_high, q, barrier_1_x]


                        if allowed_failures > 0:
                            ax.plot(xvals, quantiles.get(q), color="red", marker='*', 
                                markevery=5, linewidth=1.5, label='%dth percentile' % (int(q*100)))
                            barrier_1 = 0.1
                            barrier_1_x = 1000
                            barrier_1_y = 0
                            for x, failed in zip(xvals, quantiles.get(q)):
                                if x < barrier_1_x and failed > barrier_1:
                                    barrier_1_x = x
                                    barrier_1_y = failed
                            if barrier_1_x > 1:
                                ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+20, color='red', linestyle=':', linewidth=2, alpha=1)
                                ax.text(barrier_1_x, barrier_1_y+20, r'\noindent \textbf{' + str(barrier_1_x) + r'\%} if failure rate $\leq$ 0.1\%',
                                    fontsize=17, 
                                    verticalalignment='center', horizontalalignment='center',
                                    color='red', alpha=1,
                                    bbox=dict(facecolor='white', edgecolor='red')
                                )
                                operational_range01[str((ratio_low, ratio_high, q))] = [ratio_low, ratio_high, q, barrier_1_x]
                            ax.plot(xvals, quantiles.get(q), color="red", marker='*', 
                                markevery=5, linewidth=1.5)
                            barrier_1 = 1
                            barrier_1_x = 1000
                            barrier_1_y = 0
                            for x, failed in zip(xvals, quantiles.get(q)):
                                if x < barrier_1_x and failed > barrier_1:
                                    barrier_1_x = x
                                    barrier_1_y = failed
                            if barrier_1_x > 1:
                                ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+50, color='red', linestyle=':', linewidth=2, alpha=1)
                                ax.text(barrier_1_x, barrier_1_y+50, r'\noindent \textbf{' + str(barrier_1_x) + r'\%} if failure rate $\leq$ 1\%',
                                    fontsize=17, 
                                    verticalalignment='center', horizontalalignment='center',
                                    color='red', alpha=1,
                                    bbox=dict(facecolor='white', edgecolor='red')
                                )
                                operational_range1[str((ratio_low, ratio_high, q))] = [ratio_low, ratio_high, q, barrier_1_x]
                    elif q == HIGHLIGHT_QUANTILE_BLUE:
                        if allowed_failures == 0:
                            ax.plot(xvals, quantiles.get(q), color="blue", marker='X', 
                                markevery=5, linewidth=1.5, label='%dth percentile' % (int(q*100)))
                            barrier_1 = allowed_failures
                            barrier_1_x = 1000
                            barrier_1_y = 0
                            for x, failed in zip(xvals, quantiles.get(q)):
                                if x < barrier_1_x and failed > barrier_1:
                                    barrier_1_x = x
                                    barrier_1_y = failed 
                            if barrier_1_x > 1:
                                ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+30, color='blue', linestyle=':', linewidth=2, alpha=1)
                                ax.text(barrier_1_x, barrier_1_y+30, r'\noindent \textbf{' + str(barrier_1_x) + r'\%}',
                                    fontsize=17, 
                                    verticalalignment='center', horizontalalignment='center',
                                    color='blue', alpha=1,
                                    bbox=dict(facecolor='white', edgecolor='blue')
                                )
                                operational_range0[str((ratio_low, ratio_high, q))] = [ratio_low, ratio_high, q, barrier_1_x]
                        if allowed_failures > 0:
                            ax.plot(xvals, quantiles.get(q), color="blue", marker='*', 
                                markevery=5, linewidth=1.5, label='%dth percentile' % (int(q*100)))
                            barrier_1 = 0.1
                            barrier_1_x = 1000
                            barrier_1_y = 0
                            for x, failed in zip(xvals, quantiles.get(q)):
                                if x < barrier_1_x and failed > barrier_1:
                                    barrier_1_x = x
                                    barrier_1_y = failed
                            if barrier_1_x > 1:
                                ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+35, color='blue', linestyle=':', linewidth=2, alpha=1)
                                ax.text(barrier_1_x, barrier_1_y+35, r'\noindent \textbf{' + str(barrier_1_x) + r'\%} if failure rate $\leq$ 0.1\%',
                                    fontsize=17, 
                                    verticalalignment='center', horizontalalignment='center',
                                    color='blue', alpha=1,
                                    bbox=dict(facecolor='white', edgecolor='blue')
                                )
                                operational_range01[str((ratio_low, ratio_high, q))] = [ratio_low, ratio_high, q, barrier_1_x]
                            ax.plot(xvals, quantiles.get(q), color="blue", marker='*', 
                                markevery=5, linewidth=1.5)
                            barrier_1 = 1
                            barrier_1_x = 1000
                            barrier_1_y = 0
                            for x, failed in zip(xvals, quantiles.get(q)):
                                if x < barrier_1_x and failed > barrier_1:
                                    barrier_1_x = x
                                    barrier_1_y = failed
                            if barrier_1_x > 1:
                                ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+65, color='blue', linestyle=':', linewidth=2, alpha=1)
                                ax.text(barrier_1_x, barrier_1_y+65, r'\noindent \textbf{' + str(barrier_1_x) + r'\%} if failure rate $\leq$ 1\%',
                                    fontsize=17, 
                                    verticalalignment='center', horizontalalignment='center',
                                    color='blue', alpha=1,
                                    bbox=dict(facecolor='white', edgecolor='blue')
                                )
                                operational_range1[str((ratio_low, ratio_high, q))] = [ratio_low, ratio_high, q, barrier_1_x]


                    else:
                        ax.plot(xvals, quantiles.get(q), alpha=0.3, label='%dth percentile' % (int(q*100))) 


                subax = inset_axes(ax, width="100%", height='20%', loc='upper center')
                subax.bar(xvals, histdata, color='lightgray')
                #subax.get_xaxis().set_visible(False)
                subax.patch.set_alpha(0)
                #subax.get_yaxis().set_visible(False)
                subax.spines['top'].set_visible(False)
                subax.spines['right'].set_visible(False)
                #subax.spines['bottom'].set_visible(False)
                subax.spines['left'].set_visible(False) 
                subax.set_xlim(-5,85)
                if CUTOFF_THRESHOLD > 0:
                    subax.set_xlim(-5,CUTOFF_THRESHOLD+5)    
                subax.set_xticks([])
                subax.yaxis.tick_right()
                subax.text(1.05, 0.5, 'Number of\nscenarios\n(%d in total)' % sum(histdata), fontsize=12, 
                    verticalalignment='center', horizontalalignment='left',
                    transform=subax.transAxes, color='gray', alpha=1,
                    #bbox=dict(facecolor='white', edgecolor='white')
                )

                ax.set_xlim(-5,85)
                ax.set_xticks([1,10,20,30,40,50,60,70,80])
                if CUTOFF_THRESHOLD > 0:
                    ax.set_xlim(-5,CUTOFF_THRESHOLD+5)  
                    ax.set_xticks([1] + list(range(10, CUTOFF_THRESHOLD+1, 10)))
                ax.set_ylim(-5,120)
                ax.set_yticks([0,20,40,60,80])
                if RELATIVE_FAILURE:
                    ax.set_ylim(-5,140)
                    ax.set_yticks([0,20,40,60,80,100])       

                if TITLE_UTIL_RATIO:
                    ax.set_title(r'\textbf{Flow table utilization ratio between ' +
                        ('%.2f' % (min(used_ratios))) + r'\% and ' +
                        ('%.2f' % (max(used_ratios))) + r'\%}')

                if TITLE_ALLOWED_FAILURE_RATE:
                    if allowed_failures == 0:
                        ax.set_title(r'\textbf{Allowed failure rate: ' +
                            ('%.2f' % (allowed_failures)) + r'\%}')        

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='center left',  bbox_to_anchor=(0.8,0.5), ncol=1, fontsize=13)
                fig.subplots_adjust(hspace=0.3, right=0.8) # no gap

                filename = "ds_%d_class_%d_%d_%d%s.pdf" % (DATASET_TYPE, 
                    ratio_low, ratio_high, int(allowed_failures*100), quantile_extension)   
                utils.export(fig, filename, folder='flow-delegation-performance')

                #plt.show()
                #exit()  


    #--------------------
    # operational range create json files
    #--------------------
    data = json.dumps(operational_range0)
    utils.export_textfile(data, 'json_operational_range0', folder='flow-delegation-performance')  
    data = json.dumps(operational_range01)
    utils.export_textfile(data, 'json_operational_range01', folder='flow-delegation-performance')  
    data = json.dumps(operational_range1)
    utils.export_textfile(data, 'json_operational_range1', folder='flow-delegation-performance')  


    """
    sets = np.array_split(ratios, NUMBER_OF_CLASSES)
    for i, set in enumerate(sets):
        print(i, len(set), statistics.mean([x for x, _ in set]))
    """

    exit()

    # ----------------------------------

    fig, axes = plt.subplots(NUMBER_OF_CLASSES, figsize=FIGSIZE, sharex=False, sharey=True)
    fig.tight_layout(pad=2.7)

    
    for i, set in enumerate(sets):
        try:
            ax = axes[i]
        except TypeError:
            ax = axes

        alldata = {}
        used_ratios = []
        data_x = []
        data_y = []

        for ratio, seed in set:
            used_ratios.append(ratio)
            runs = runs_by_seed.get(seed)
            for run in runs:
                failed = run.get('rsa_table_percent_relocated')
                percent_relocated = run.get('rsa_table_percent_relocated') / (run.get('rsa_table_percent_relocated')  + run.get('rsa_table_percent_shared')) * 100
                if RELATIVE_FAILURE:
                    failed = percent_relocated   
                x = 100-run.get('scenario_gen_param_topo_switch_capacity')
                if not alldata.get(x):
                    alldata[x] = []   
                alldata[x].append(failed)
                data_y.append(failed)


        total = 0
        xvals = []
        boxdata = []
        histdata = []
        quantiles = {}
        for x, data in sorted(alldata.items()):
            #print(x, len(data))
            boxdata.append(data)
            xvals.append(x)
            histdata.append(len(data))
            total += len(data)
            for q in QUANTILES:
                try:
                    quantiles[q].append(np.quantile(data, q)) # or np.percentile(data, q*100) which is the same
                except KeyError:
                    quantiles[q] = [np.quantile(data, q)]


        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel(r'Failure rate (in \%)')
        ax.set_xlabel(r'Flow table capacity reduction (in \%)')
        if NUMBER_OF_CLASSES > 1:
            ax.set_title(r'\textbf{Flow table utilization between ' +
                ('%.2f' % (min(used_ratios)*100)) + r'\% and ' +
                ('%.2f' % (max(used_ratios)*100)) + r'\%}')

        # ----------------------- Boxplot
        if BOXPLOT:
            ax.boxplot(boxdata, notch=True)
            ax.set_xlim(-5,85)
            ax.set_xticks([1,10,20,30,40,50,60,70,80])
            if CUTOFF_THRESHOLD > 0:
                ax.set_xlim(-5,CUTOFF_THRESHOLD+5)  
                print("use", [1] + list(range(10, CUTOFF_THRESHOLD+1, 10)))
                ax.set_xticklabels([1] + list(range(10, CUTOFF_THRESHOLD+1, 10)))    
            continue

        # ----------------------- regular plot with percentiles
        for q in QUANTILES:
            if q == HIGHLIGHT_QUANTILE_RED:
                ax.plot(xvals, quantiles.get(q), color="black", marker='*', 
                    markevery=5, linewidth=1.5, label='%dth percentile' % (int(q*100)))
                barrier_1 = 0.1
                barrier_1_x = 1000
                barrier_1_y = 0
                for x, failed in zip(xvals, quantiles.get(q)):
                    if x < barrier_1_x and failed > barrier_1:
                        barrier_1_x = x
                        barrier_1_y = failed
                if barrier_1_x > 1:
                    ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+20, color='black', linestyle=':', linewidth=2, alpha=1)
                    ax.text(barrier_1_x, barrier_1_y+20, r'\noindent \textbf{' + str(int(q*100)) + 
                        r'\%} of the scenarios\\have a failure rate\\ of \textbf{below ' + str(barrier_1)  + r'\%} if flow table\\' +
                        r'capacity is \textbf{reduced by ' + str(barrier_1_x) + r'\%}',
                        fontsize=11, 
                        verticalalignment='bottom', horizontalalignment='center',
                        color='gray', alpha=1,
                        bbox=dict(facecolor='white', edgecolor='black')
                    )
            elif q == HIGHLIGHT_QUANTILE_BLUE:
                ax.plot(xvals, quantiles.get(q), color="blue", marker='X', 
                    markevery=5, linewidth=1.5, label='%dth percentile' % (int(q*100)))
                barrier_1 = 0.1
                barrier_1_x = 1000
                barrier_1_y = 0
                for x, failed in zip(xvals, quantiles.get(q)):
                    if x < barrier_1_x and failed > barrier_1:
                        barrier_1_x = x
                        barrier_1_y = failed 
                if barrier_1_x > 1:
                    ax.vlines(barrier_1_x, barrier_1_y, barrier_1_y+55, color='blue', linestyle=':', linewidth=2, alpha=1)
                    ax.text(barrier_1_x, barrier_1_y+55, r'\noindent \textbf{' + str(int(q*100)) + 
                        r'\%} of the scenarios\\have a failure rate\\ of \textbf{below ' + str(barrier_1)  + r'\%} if flow table\\' +
                        r'capacity is \textbf{reduced by ' + str(barrier_1_x) + r'\%}',
                        fontsize=11, 
                        verticalalignment='bottom', horizontalalignment='center',
                        color='gray', alpha=1,
                        bbox=dict(facecolor='white', edgecolor='blue')
                    )
            else:
                ax.plot(xvals, quantiles.get(q), alpha=0.3, label='%dth percentile' % (int(q*100))) 


        subax = inset_axes(ax, width="100%", height='20%', loc='upper center')
        print(histdata)
        subax.bar(xvals, histdata, color='lightgray')
        #subax.get_xaxis().set_visible(False)
        subax.patch.set_alpha(0)
        #subax.get_yaxis().set_visible(False)
        subax.spines['top'].set_visible(False)
        subax.spines['right'].set_visible(False)
        #subax.spines['bottom'].set_visible(False)
        subax.spines['left'].set_visible(False) 
        subax.set_xlim(-5,85)
        if CUTOFF_THRESHOLD > 0:
            subax.set_xlim(-5,CUTOFF_THRESHOLD+5)    
        subax.set_xticks([])
        subax.yaxis.tick_right()
        subax.text(1.05, 0.5, 'Number of\nscenarios\n(%d in total)' % sum(histdata), fontsize=10, 
            verticalalignment='center', horizontalalignment='left',
            transform=subax.transAxes, color='gray', alpha=1,
            #bbox=dict(facecolor='white', edgecolor='white')
        )


        ax.set_xlim(-5,85)
        ax.set_xticks([1,10,20,30,40,50,60,70,80])
        if CUTOFF_THRESHOLD > 0:
            ax.set_xlim(-5,CUTOFF_THRESHOLD+5)  
            ax.set_xticks([1] + list(range(10, CUTOFF_THRESHOLD+1, 10)))
        ax.set_ylim(-5,120)
        ax.set_yticks([0,20,40,60,80])
        if RELATIVE_FAILURE:
            ax.set_ylim(-5,140)
            ax.set_yticks([0,20,40,60,80,100])         
        #ax.set_xticks(xvals)
        #ax.set_xticklabels(xvals)

    if not BOXPLOT:
        handles, labels = ax.get_legend_handles_labels()
        if NUMBER_OF_CLASSES == 1:
            fig.legend(handles, labels, loc='center left',  bbox_to_anchor=(0.8,0.5), ncol=1, fontsize=12)
        else:
            fig.legend(handles, labels, loc='upper left',  bbox_to_anchor=(0.8,0.9), ncol=1, fontsize=12)
        fig.subplots_adjust(hspace=0.3, right=0.8) # no gap

    filename = "flow-delegation-performance_ds%d_class_%d.pdf" % (DATASET_TYPE, NUMBER_OF_CLASSES)
    if RELATIVE_FAILURE:
        filename = "flow-delegation-performance_relative_ds%d_class_%d.pdf" % (DATASET_TYPE, NUMBER_OF_CLASSES) 
    if BOXPLOT:
        filename = 'boxplot_' + filename   
    utils.export(fig, filename, folder='flow-delegation-performance')
    plt.show()
    plt.close()   






