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

LOOKAHEAD =  [1,2,3,4,5,6,7,8,9]

DATASET_TYPE = 0

def plot(blob, **kwargs):
    "Plot look-ahead DTS"

    utils.EXPORT_BLOB = blob
  
    
    if 'lookahead12' in blob.path:
        logger.info('set DATASET_TYPE to 5000')
        DATASET_TYPE = 5000

        includes = ['scenario_switch_cnt']
        includes += blob.find_columns('hit_timelimit')
        includes += blob.find_columns('solver_cnt_feasable')
        includes += blob.find_columns('solver_cnt_infeasable')

        blob.include_parameters(**dict.fromkeys(includes, 1))

        for param_dts_look_ahead in [1,2]:
            print("param_dts_look_ahead", param_dts_look_ahead)
            runs = blob.filter(param_dts_look_ahead=param_dts_look_ahead)
            cnt_timeout = 0
            cnt_feasible = 0
            cnt_infeasible = 0
            for run in runs:
                if run.get('hit_timelimit'):
                    cnt_timeout += 1
                    continue
                for switch in range(0, run.get('scenario_switch_cnt')):

                    f1 = run.get('dts_%d_solver_cnt_feasable' % (switch))
                    f2 = run.get('dts_%d_solver_cnt_infeasable' % (switch))
                    if f1 and f1 > 0:
                        cnt_feasible += 1
                        if f2 and f2 > 0:
                            cnt_infeasible += 1
            print("cnt_timeout", cnt_timeout)
            print("cnt_feasible", cnt_feasible)
            print("cnt_infeasible", cnt_infeasible)
        exit()

    macros_ports = []
    macros_total = []
    total = {}
    total_infeasable = {}

    data_per_port = {}

    includes = ['scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch']

    includes += blob.find_columns('solver_stats_time_modeling')
    includes += blob.find_columns('solver_stats_time_solving')
    includes += blob.find_columns('timer')
    includes += blob.find_columns('hit_timelimit')
    

    # 15 is the maximum number if switches in current experiments; (id 0-14)
    for switch_cnt in range(0, 15):
        includes.append('dts_%d_solver_cnt_feasable' % (switch_cnt))
        includes.append('dts_%d_solver_cnt_infeasable' % (switch_cnt))
        includes.append('dts_%d_solver_considered_ports' % (switch_cnt))

        includes.append('dts_%d_ctrl_overhead' % (switch_cnt))
        includes.append('dts_%d_link_overhead' % (switch_cnt))
        includes.append('dts_%d_table_overhead' % (switch_cnt))
        includes.append('dts_%d_table_datay_raw' % (switch_cnt))



        includes.append('dts_%d_ctrl_overhead_percent' % (switch_cnt))
        includes.append('dts_%d_link_overhead_percent' % (switch_cnt))
        includes.append('dts_%d_table_overhead_percent' % (switch_cnt))
        includes.append('dts_%d_underutil_percent' % (switch_cnt))

    blob.include_parameters(**dict.fromkeys(includes, 1))


    runs = blob.filter(**dict())

    # -----------------------
    # prepare data for plotting
    # -----------------------
    DATA = {}
    seeds = []
    timercnt = 0
    for run in runs:
        seed = run.get('param_topo_seed')
        param_dts_algo = run.get('param_dts_algo')
        param_dts_look_ahead = run.get('param_dts_look_ahead')
        if not seed in seeds:
            seeds.append(seed)
        if not DATA.get(param_dts_algo):
            DATA[param_dts_algo] = {}


        for switch in range(0, run.get('scenario_switch_cnt')):
            if not DATA[param_dts_algo].get((seed, switch)):
                DATA[param_dts_algo][(seed, switch)] = {} 
            DATA[param_dts_algo][(seed, switch)][param_dts_look_ahead] = run

    # -----------------------
    # Figure: modeling and solving time dts based on look-ahead 1..9 or 1..30
    # -----------------------
    if 1:
        for factor_limit in [9,30]:
            for param_dts_algo, DATA1 in DATA.items():
                if param_dts_algo== 1 and factor_limit == 30:
                    continue
                plt.close()
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.tight_layout(pad=3)
                for ax, label in zip(fig.axes, [r'Modeling time (ms)', r'Solving time (ms)']*2):
                    ax.set_xlabel('Look-ahead factor L', fontsize=15)
                    ax.set_ylabel('%s' % label, fontsize=15)

                for ax in fig.axes:
                    ax.set_xlim(1,9)
                    ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)  

                color = 'red'
                label = 'Select-Opt'
                if param_dts_algo == 2:
                    color = 'blue'
                    label = 'Select-CopyFirst'

                # for this plot, we can only use data where all look-ahead values 
                # were calculated without timeouts!
                use_data = {}
                ignore_seeds = []
                for seedswitch, DATA2 in DATA1.items():
                    use = True
                    seed, switch = seedswitch
                    for param_dts_look_ahead, run in sorted(DATA2.items()):
                        if run.get('hit_timelimit'):
                            use = False
                        if run.get('dts_%d_ctrl_overhead_percent' % (switch)) == None:
                            use = False
                    if use == False:
                        if not seed in ignore_seeds:
                            ignore_seeds.append(seed)
                for seedswitch, DATA2 in DATA1.items():
                    seed, switch = seedswitch
                    if seed not in ignore_seeds:
                        use_data[seedswitch] = DATA2


                result_solver_stats_time_solving = {}
                result_solver_stats_time_modeling = {}
                for seedswitch, DATA2 in use_data.items():
                    seed, switch = seedswitch
                    datax = []
                    datay = []
                    for param_dts_look_ahead, run in sorted(DATA2.items()):
                        if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                            continue

                        if run.get('dts_%d_solver_cnt_infeasable' % (switch)) == 0:

                            solver_stats_time_solving = run.get('dts_%d_solver_stats_time_solving' % (switch))
                            solver_stats_time_modeling = run.get('dts_%d_solver_stats_time_modeling' % (switch))

                            solver_stats_time_solving = statistics.mean(solver_stats_time_solving) * 1000
                            solver_stats_time_modeling = statistics.mean(solver_stats_time_modeling) * 1000

                            try:
                                result_solver_stats_time_solving[param_dts_look_ahead].append(solver_stats_time_solving)
                            except KeyError:
                                result_solver_stats_time_solving[param_dts_look_ahead] = [solver_stats_time_solving]

                            try:
                                result_solver_stats_time_modeling[param_dts_look_ahead].append(solver_stats_time_modeling)
                            except KeyError:
                                result_solver_stats_time_modeling[param_dts_look_ahead] = [solver_stats_time_modeling]

                print("")
                print("solving time", param_dts_algo)
                datax = []
                datay = []
                databox = []
                for param_dts_look_ahead, data in sorted(result_solver_stats_time_solving.items()):
                    if param_dts_look_ahead <= factor_limit:
                        print(" ", param_dts_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))
                        datax.append(param_dts_look_ahead)
                        datay.append(statistics.median(data))
                        databox.append(data)
                axes[1].boxplot(databox, positions=datax, notch=False, showfliers=False)
                axes[1].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

                print("")
                print("modeling time", param_dts_algo)
                datax = []
                datay = []
                databox = []
                for param_dts_look_ahead, data in sorted(result_solver_stats_time_modeling.items()):
                    if param_dts_look_ahead <= factor_limit:
                        print(" ", param_dts_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))
                        datax.append(param_dts_look_ahead)
                        datay.append(statistics.median(data))
                        databox.append(data)
                axes[0].boxplot(databox, positions=datax, notch=False, showfliers=False)
                axes[0].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)
                
                if factor_limit > 9:
                    for ax in fig.axes:      
                        labels = []
                        for x in range(1,factor_limit):
                            if x % 2 == 1:
                                labels.append(''+str(x))
                            else:
                                labels.append('')
                        ax.set_xticklabels(labels)
                #h1, l1 = fig.axes[-1].get_legend_handles_labels()
                h2, l2 = fig.axes[0].get_legend_handles_labels()
                fig.legend(h2, l2, loc='upper center', ncol=2, fontsize=16)
                fig.subplots_adjust(top=0.88) 
                utils.export(fig, 'lookahead_%d_dts_%d_solving.pdf' % (factor_limit, param_dts_algo), folder='lookahead')


    # -----------------------
    # Figure: rsa times for L=1  to L=30
    # -----------------------
    if 0:
        use_data = {}
        for run in runs:
            if run.get('param_dts_algo') != 2:
                continue  
            seed = run.get('param_topo_seed')
            param_dts_look_ahead = run.get('param_dts_look_ahead')
            if not use_data.get((param_dts_look_ahead, seed)):
                use_data[(param_dts_look_ahead, seed)] = []
            use_data[(param_dts_look_ahead, seed)].append(run)

        if len(use_data) > 0:
            plt.close()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.tight_layout(pad=3)
            for ax, label in zip(fig.axes, [r'Modeling time (ms)', r'Solving time (ms)']*2):
                ax.set_xlabel('Look-ahead factor L', fontsize=15)
                ax.set_ylabel('%s' % label, fontsize=15)

            for ax in fig.axes:
                ax.set_xlim(1,9)
                ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)  


            color = 'blue'
            label = 'Select-CopyFirst'

            result_solver_stats_time_solving = {}
            result_solver_stats_time_modeling = {}
            for key, runs in sorted(use_data.items()):
                param_dts_look_ahead, seed = key

                for run in runs:
                    solver_stats_time_solving = run.get('rsa_solver_stats_time_solving')
                    solver_stats_time_modeling = run.get('rsa_solver_stats_time_modeling')
                    solver_stats_time_solving = statistics.mean(solver_stats_time_solving) * 1000
                    solver_stats_time_modeling = statistics.mean(solver_stats_time_modeling) * 1000
                    try:
                        result_solver_stats_time_solving[param_dts_look_ahead].append(solver_stats_time_solving)
                    except KeyError:
                        result_solver_stats_time_solving[param_dts_look_ahead] = [solver_stats_time_solving]
                    try:
                        result_solver_stats_time_modeling[param_dts_look_ahead].append(solver_stats_time_modeling)
                    except KeyError:
                        result_solver_stats_time_modeling[param_dts_look_ahead] = [solver_stats_time_modeling]

            print("")
            print("solving time", param_dts_algo)
            datax = []
            datay = []
            databox = []
            for param_dts_look_ahead, data in sorted(result_solver_stats_time_solving.items()):
                print(" ", param_dts_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
                databox.append(data)
            axes[1].boxplot(databox, positions=datax, notch=False, showfliers=False)
            axes[1].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

            print("")
            print("modeling time", param_dts_algo)
            datax = []
            datay = []
            databox = []
            for param_dts_look_ahead, data in sorted(result_solver_stats_time_modeling.items()):
                print(" ", param_dts_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
                databox.append(data)
            axes[0].boxplot(databox, positions=datax, notch=False, showfliers=False)
            axes[0].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)
            
            for ax in fig.axes:      
                labels = []
                for x in range(1,30):
                    if x % 2 == 1:
                        labels.append(''+str(x))
                    else:
                        labels.append('')
                ax.set_xticklabels(labels)
            #h1, l1 = fig.axes[-1].get_legend_handles_labels()
            h2, l2 = fig.axes[0].get_legend_handles_labels()
            fig.legend(h2, l2, loc='upper center', ncol=2, fontsize=16)
            fig.subplots_adjust(top=0.88) 
            utils.export(fig, 'lookahead_rsa30.pdf', folder='lookahead')

    # -----------------------
    # Figure: box plots for all three metrics, ordered by look-ahead
    # -----------------------
    if 1:

        plt.close()
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.tight_layout(pad=2.7)
        for ax, label in zip(fig.axes, [r'Table overhead (rules)',  r'Link overhead (Mbit/s)', r'Control overhead (messages/s)']*2):
            ax.set_xlabel('Look-ahead factor L', fontsize=15)
            ax.set_ylabel('%s' % label, fontsize=15)

        for ax in fig.axes:
            #ax.set_xlim(1,9)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)       
            ax.set_xticks([[x*2+1 for x in np.arange(14)]])
        for param_dts_algo, DATA1 in DATA.items():
            color = 'red'
            label = 'Select-Opt'
            if param_dts_algo == 2:
                color = 'blue'
                label = 'Select-CopyFirst'

            # for this plot, we can only use data where all look-ahead values 
            # were calculated without timeouts!
            use_data = {}
            ignore_seeds = []
            for seedswitch, DATA2 in DATA1.items():
                use = True
                seed, switch = seedswitch
                for param_dts_look_ahead, run in sorted(DATA2.items()):
                    if run.get('hit_timelimit'):
                        use = False
                    if run.get('dts_%d_ctrl_overhead_percent' % (switch)) == None:
                        use = False
                if use == False:
                    if not seed in ignore_seeds:
                        ignore_seeds.append(seed)
            for seedswitch, DATA2 in DATA1.items():
                seed, switch = seedswitch
                if seed not in ignore_seeds:
                    use_data[seedswitch] = DATA2

            # create data for the boxplots
            result_ctrl_overhead_percent = {}
            result_link_overhead_percent = {}
            result_table_overhead_percent = {}
            for seedswitch, DATA2 in use_data.items():
                seed, switch = seedswitch

                for param_dts_look_ahead in range(1,31):
                    run = DATA2.get(param_dts_look_ahead)
                    if not run:
                        continue

                    if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                        continue

                    if run.get('dts_%d_solver_cnt_infeasable' % (switch)) == 0:


                        thresh = run.get('scenario_table_capacity')
                        d2 = 'dts_%d_table_datay_raw' % (switch)
                        raw_util = run.get(d2)
                        fill_overutil = [1 if x > thresh else 0 for x in raw_util]

                        new_table_overhead = (run.get('dts_%d_table_overhead' % switch) / float(sum(fill_overutil)))
                        new_link_overhead = ((run.get('dts_%d_link_overhead' % switch) / 1000000) / float(sum(fill_overutil)))
                        new_ctrl_overhead = ((run.get('dts_%d_ctrl_overhead' % switch)) / float(sum(fill_overutil)))



                        ctrl_overhead_percent = run.get('dts_%d_ctrl_overhead_percent' % (switch))
                        link_overhead_percent = run.get('dts_%d_link_overhead_percent' % (switch))
                        table_overhead_percent = run.get('dts_%d_table_overhead_percent' % (switch))

                        table_overhead_percent = new_table_overhead
                        link_overhead_percent = new_link_overhead
                        ctrl_overhead_percent = new_ctrl_overhead
                    else:
                        print("!!!!!!!", param_dts_algo, seedswitch, param_dts_look_ahead)
                        # worst case score if infeasable
                        #ctrl_overhead_percent = 500 # approx. value (can be > 100%)
                        #link_overhead_percent = 100
                        #table_overhead_percent = 100           

                    try:
                        result_ctrl_overhead_percent[param_dts_look_ahead].append(ctrl_overhead_percent)
                    except KeyError:
                        result_ctrl_overhead_percent[param_dts_look_ahead] = [ctrl_overhead_percent]
                    try:
                        result_link_overhead_percent[param_dts_look_ahead].append(link_overhead_percent)
                    except KeyError:
                        result_link_overhead_percent[param_dts_look_ahead] = [link_overhead_percent]
                    try:
                        result_table_overhead_percent[param_dts_look_ahead].append(table_overhead_percent)
                    except KeyError:
                        result_table_overhead_percent[param_dts_look_ahead] = [table_overhead_percent]
            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_table_overhead_percent.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[0].boxplot(result_table_overhead_percent.values(), positions=datax, notch=False, showfliers=SHOW_OUTLIERS)
            axes[0].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_link_overhead_percent.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[1].boxplot(result_link_overhead_percent.values(), positions=datax, notch=False, showfliers=SHOW_OUTLIERS)
            axes[1].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_ctrl_overhead_percent.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[2].boxplot(result_ctrl_overhead_percent.values(), positions=datax, notch=False, showfliers=SHOW_OUTLIERS)
            axes[2].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

        for ax in fig.axes:      
            labels = []
            for x in range(1,30):
                if x % 2 == 1:
                    labels.append(''+str(x))
                else:
                    labels.append('')
            ax.set_xticklabels(labels)
        h2, l2 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h2, l2, loc='upper center', ncol=2, fontsize=16)
        fig.subplots_adjust(top=0.88) # padding top
        if not SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_dts_metrics_above_9.pdf', folder='lookahead')
        else:
            utils.export(fig, 'lookahead_dts_metrics_with_outliers_above_9.pdf', folder='lookahead')

        #plt.show()




    # -----------------------
    # Figure: score for each scenario (shows that Select-Opt runs into timeouts for most scenario)
    # -----------------------
    if 0:
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, sharey=True)
        fig.tight_layout(pad=1)
        axes[0][0].set_ylabel(r'Table overhead in \%', fontsize=15)
        axes[1][0].set_ylabel(r'Link overhead in \%', fontsize=15)
        axes[2][0].set_ylabel(r'Control overhead in \%', fontsize=15)
        axes[2][0].set_xlabel(r'Look-ahead factor L', fontsize=15)
        axes[2][1].set_xlabel(r'Look-ahead factor L', fontsize=15)

        # helper function
        def getxy(key, switch, DATA2, counts):
            datax = []
            datay = []
            for param_dts_look_ahead, run in sorted(DATA2.items()):
                if run.get('hit_timelimit'):
                    if run.get('param_dts_look_ahead') == 2:
                        print("Failed L=2 timeout", run.get('param_dts_algo'), run.get('param_topo_seed'))
                    continue
                if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                    continue
                if run.get('dts_%d_solver_cnt_infeasable' % (switch)) == 0:
                    val = run.get('dts_%d_%s' % (switch, key))
                    if val < 50:
                        # val < 50 --> there are two outliers that destroy the figure layout
                        # (are hidden here, not important)
                        datax.append(param_dts_look_ahead)
                        datay.append(val)
                    try:
                        counts[param_dts_look_ahead].append(val)
                    except KeyError:
                        counts[param_dts_look_ahead] = [val]
                else:
                    if run.get('param_dts_look_ahead') == 1 and run.get('param_dts_algo') == 2:
                        print("Failed L=1 infeasible",  run.get('param_dts_algo'), run.get('param_topo_seed'))

                    if run.get('param_dts_look_ahead') == 2:
                        print("Failed L=2 infeasible",  run.get('param_dts_algo'), run.get('param_topo_seed'))

            return datax, datay

        rowcnt = 0
        for key in ['table_overhead_percent', 'link_overhead_percent', 'ctrl_overhead_percent']:
            for param_dts_algo, DATA1 in DATA.items():
                counts = {}
                color = 'red'
                label = 'Select-Opt'
                axcnt = 0
                if param_dts_algo == 2:
                    color = 'blue'
                    label = 'Select-CopyFirst'
                    axcnt = 1

                ax = axes[rowcnt][axcnt]
                ax.set_ylim(0,60)
                ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)    
                for seedswitch, DATA2 in DATA1.items():
                    seed, switch = seedswitch
                    datax, datay = getxy(key, switch, DATA2, counts)
                    ax.plot(datax, datay, alpha=0.2, linestyle='-', color=color)  

                # number of used experiments
                for x in LOOKAHEAD:
                    count_data = counts.get(x, [])
                    if x == 1:
                        # for legend
                        ax.vlines(x, 0, 50, linestyle = '--', color=color, alpha=1, label=label)
                    else:
                        ax.vlines(x, 0, 50, linestyle = '--', color=color, alpha=1)
                    ax.text(x, 50, '%d' % len(count_data), fontsize=12, 
                        verticalalignment='center', horizontalalignment='center',
                        color='black', alpha=1,
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor=color)
                    )

            rowcnt += 1

        h1, l1 = fig.axes[-1].get_legend_handles_labels()
        h2, l2 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h2+h1, l2+l1, loc='upper center', ncol=2, fontsize=16)
        fig.subplots_adjust(left=0.05, top=0.9, bottom=0.06) # padding top
        utils.export(fig, 'lookahead_dts_all_scenarios.pdf', folder='lookahead')

    # -----------------------
    # Figure: box plots for all three metrics, ordered by look-ahead
    # -----------------------
    if 0:

        plt.close()
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.tight_layout(pad=2.7)
        for ax, label in zip(fig.axes, [r'Table overhead (rules)',  r'Link overhead (Mbit/s)', r'Control overhead (messages/s)']*2):
            ax.set_xlabel('Look-ahead factor L', fontsize=15)
            ax.set_ylabel('%s' % label, fontsize=15)

        for ax in fig.axes:
            ax.set_xlim(1,9)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)       

        for param_dts_algo, DATA1 in DATA.items():
            color = 'red'
            label = 'Select-Opt'
            if param_dts_algo == 2:
                color = 'blue'
                label = 'Select-CopyFirst'

            # for this plot, we can only use data where all look-ahead values 
            # were calculated without timeouts!
            use_data = {}
            ignore_seeds = []
            for seedswitch, DATA2 in DATA1.items():
                use = True
                seed, switch = seedswitch
                for param_dts_look_ahead, run in sorted(DATA2.items()):
                    if run.get('hit_timelimit'):
                        use = False
                    if run.get('dts_%d_ctrl_overhead_percent' % (switch)) == None:
                        use = False
                if use == False:
                    if not seed in ignore_seeds:
                        ignore_seeds.append(seed)
            for seedswitch, DATA2 in DATA1.items():
                seed, switch = seedswitch
                if seed not in ignore_seeds:
                    use_data[seedswitch] = DATA2

            # some debugging (not important)
            print("ignore_seeds", len(ignore_seeds))
            print(param_dts_algo, "old", len(DATA1))
            print(param_dts_algo, "new", len(use_data))
            for seedswitch, DATA2 in use_data.items():
                seed, switch = seedswitch
                for x in range(1,10):
                    run = DATA[1][seedswitch][x]
                    val = run.get('dts_%d_table_overhead_percent' % (switch))
                    run2 = DATA[2][seedswitch][x]
                    val2 = run2.get('dts_%d_table_overhead_percent' % (switch))
                    print("> ", 2, seedswitch, x, val, " -- ", val2)
                break

            # create data for the boxplots
            result_ctrl_overhead_percent = {}
            result_link_overhead_percent = {}
            result_table_overhead_percent = {}
            for seedswitch, DATA2 in use_data.items():
                seed, switch = seedswitch

                for param_dts_look_ahead in LOOKAHEAD:
                    run = DATA2.get(param_dts_look_ahead)

                    if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                        continue

                    if run.get('dts_%d_solver_cnt_infeasable' % (switch)) == 0:


                        thresh = run.get('scenario_table_capacity')
                        d2 = 'dts_%d_table_datay_raw' % (switch)
                        raw_util = run.get(d2)
                        fill_overutil = [1 if x > thresh else 0 for x in raw_util]

                        new_table_overhead = (run.get('dts_%d_table_overhead' % switch) / float(sum(fill_overutil)))
                        new_link_overhead = ((run.get('dts_%d_link_overhead' % switch) / 1000000) / float(sum(fill_overutil)))
                        new_ctrl_overhead = ((run.get('dts_%d_ctrl_overhead' % switch)) / float(sum(fill_overutil)))



                        ctrl_overhead_percent = run.get('dts_%d_ctrl_overhead_percent' % (switch))
                        link_overhead_percent = run.get('dts_%d_link_overhead_percent' % (switch))
                        table_overhead_percent = run.get('dts_%d_table_overhead_percent' % (switch))

                        table_overhead_percent = new_table_overhead
                        link_overhead_percent = new_link_overhead
                        ctrl_overhead_percent = new_ctrl_overhead
                    else:
                        print("!!!!!!!", param_dts_algo, seedswitch, param_dts_look_ahead)
                        # worst case score if infeasable
                        #ctrl_overhead_percent = 500 # approx. value (can be > 100%)
                        #link_overhead_percent = 100
                        #table_overhead_percent = 100           

                    try:
                        result_ctrl_overhead_percent[param_dts_look_ahead].append(ctrl_overhead_percent)
                    except KeyError:
                        result_ctrl_overhead_percent[param_dts_look_ahead] = [ctrl_overhead_percent]
                    try:
                        result_link_overhead_percent[param_dts_look_ahead].append(link_overhead_percent)
                    except KeyError:
                        result_link_overhead_percent[param_dts_look_ahead] = [link_overhead_percent]
                    try:
                        result_table_overhead_percent[param_dts_look_ahead].append(table_overhead_percent)
                    except KeyError:
                        result_table_overhead_percent[param_dts_look_ahead] = [table_overhead_percent]
            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_table_overhead_percent.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[param_dts_algo-1][0].boxplot(result_table_overhead_percent.values(), positions=datax, notch=False, showfliers=SHOW_OUTLIERS)
            axes[param_dts_algo-1][0].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_link_overhead_percent.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[param_dts_algo-1][1].boxplot(result_link_overhead_percent.values(), positions=datax, notch=False, showfliers=SHOW_OUTLIERS)
            axes[param_dts_algo-1][1].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_ctrl_overhead_percent.items()):
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[param_dts_algo-1][2].boxplot(result_ctrl_overhead_percent.values(), positions=datax, notch=False, showfliers=SHOW_OUTLIERS)
            axes[param_dts_algo-1][2].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)


        h1, l1 = fig.axes[-1].get_legend_handles_labels()
        h2, l2 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h2+h1, l2+l1, loc='upper center', ncol=2, fontsize=16)
        fig.subplots_adjust(top=0.90) # padding top
        if not SHOW_OUTLIERS:
            utils.export(fig, 'lookahead_dts_metrics.pdf', folder='lookahead')
        else:
            utils.export(fig, 'lookahead_dts_metrics_with_outliers.pdf', folder='lookahead')

        #plt.show()

    # -----------------------
    # Figure: modeling and solving time based on look-ahead
    # -----------------------
    if 0:

        plt.close()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.tight_layout(pad=3)
        for ax, label in zip(fig.axes, [r'Modeling time (ms)', r'Solving time (ms)']*2):
            ax.set_xlabel('Look-ahead factor L', fontsize=15)
            ax.set_ylabel('%s' % label, fontsize=15)

        for ax in fig.axes:
            ax.set_xlim(1,9)
            ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)  

        for param_dts_algo, DATA1 in DATA.items():
            color = 'red'
            label = 'Select-Opt'
            if param_dts_algo == 2:
                color = 'blue'
                label = 'Select-CopyFirst'

            # for this plot, we can only use data where all look-ahead values 
            # were calculated without timeouts!
            use_data = {}
            ignore_seeds = []
            for seedswitch, DATA2 in DATA1.items():
                use = True
                seed, switch = seedswitch
                for param_dts_look_ahead, run in sorted(DATA2.items()):
                    if run.get('hit_timelimit'):
                        use = False
                    if run.get('dts_%d_ctrl_overhead_percent' % (switch)) == None:
                        use = False
                if use == False:
                    if not seed in ignore_seeds:
                        ignore_seeds.append(seed)
            for seedswitch, DATA2 in DATA1.items():
                seed, switch = seedswitch
                if seed not in ignore_seeds:
                    use_data[seedswitch] = DATA2


            result_solver_stats_time_solving = {}
            result_solver_stats_time_modeling = {}
            for seedswitch, DATA2 in use_data.items():
                seed, switch = seedswitch
                datax = []
                datay = []
                for param_dts_look_ahead, run in sorted(DATA2.items()):
                    if run.get('dts_%d_table_overhead_percent' % (switch)) == 0:
                        continue

                    if run.get('dts_%d_solver_cnt_infeasable' % (switch)) == 0:

                        solver_stats_time_solving = run.get('dts_%d_solver_stats_time_solving' % (switch))
                        solver_stats_time_modeling = run.get('dts_%d_solver_stats_time_modeling' % (switch))

                        solver_stats_time_solving = statistics.mean(solver_stats_time_solving) * 1000
                        solver_stats_time_modeling = statistics.mean(solver_stats_time_modeling) * 1000

                        try:
                            result_solver_stats_time_solving[param_dts_look_ahead].append(solver_stats_time_solving)
                        except KeyError:
                            result_solver_stats_time_solving[param_dts_look_ahead] = [solver_stats_time_solving]

                        try:
                            result_solver_stats_time_modeling[param_dts_look_ahead].append(solver_stats_time_modeling)
                        except KeyError:
                            result_solver_stats_time_modeling[param_dts_look_ahead] = [solver_stats_time_modeling]

            print("")
            print("solving time", param_dts_algo)
            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_solver_stats_time_solving.items()):
                print(" ", param_dts_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[param_dts_algo-1][1].boxplot(result_solver_stats_time_solving.values(), positions=datax, notch=False, showfliers=False)
            axes[param_dts_algo-1][1].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

            print("")
            print("modeling time", param_dts_algo)
            datax = []
            datay = []
            for param_dts_look_ahead, data in sorted(result_solver_stats_time_modeling.items()):
                print(" ", param_dts_look_ahead, np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 99))
                datax.append(param_dts_look_ahead)
                datay.append(statistics.median(data))
            axes[param_dts_algo-1][0].boxplot(result_solver_stats_time_modeling.values(), positions=datax, notch=False, showfliers=False)
            axes[param_dts_algo-1][0].plot(datax, datay, color=color, marker='*', linestyle="--", linewidth=2, label=label)

        h1, l1 = fig.axes[-1].get_legend_handles_labels()
        h2, l2 = fig.axes[0].get_legend_handles_labels()
        fig.legend(h2+h1, l2+l1, loc='upper center', ncol=2, fontsize=16)
        fig.subplots_adjust(top=0.90) 
        utils.export(fig, 'lookahead_dts_solving_times.pdf', folder='lookahead')

        exit(1)


    # -----------------------
    # Figure: infeasible solutions based on look-ahead
    # -----------------------
    if 0:
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
        exit(1)


