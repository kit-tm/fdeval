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
    # Figure: Experiment execution times for DT-Select look-ahead analysis
    # -----------------------
    if 1:
        fig, axes = plt.subplots(2,1,figsize=(10, 8))
        fig.tight_layout(pad=2.7)


        def compare(r1, r2):
            v1 = 0
            v2 = 0
            if r1.get('hit_timelimit'):
                v1 += 200000000
            if r2.get('hit_timelimit'):
                v2 += 200000000

            for switch in range(0, r1.get('scenario_switch_cnt')):
                v = r1.get('timer_rsa_solver_dts_%d' % switch)
                if v:
                    v1 += v
            for switch in range(0, r2.get('scenario_switch_cnt')):
                v = r2.get('timer_rsa_solver_dts_%d' % switch)    
                if v:
                    v2 += v           

            return v1 - v2

        import functools

        axes[0].set_title(r'\textbf{Experiments with Select-Opt}', fontsize=15)
        axes[1].set_title(r'\textbf{Experiments with Select-CopyFirst}', fontsize=15)
        axes[1].set_xlabel('Experiment', fontsize=15)

        for algo in [1,2]:

            time_prepare_rsa = []
            time_rsa = []
            time_dts = []
            time_dts_single =  []
            time_scenario = []
            time_precalc = []
            time_statistics = []
            timeout = []
            cnttimeout = 0
            for run in sorted(runs, key=functools.cmp_to_key(compare)):  
                if run.get('param_dts_algo') != algo: continue 
                if run.get('hit_timelimit'):
                    cnttimeout += 1
                    timeout.append(1600)
                    time_prepare_rsa.append(0)
                    time_rsa.append(0)
                    time_dts.append(0)
                    time_dts_single.append(0)
                    time_scenario.append(0)
                    time_precalc.append(0)
                    time_statistics.append(0)        
                    continue
                dts = []
                for switch in range(0, run.get('scenario_switch_cnt')):
                    v = run.get('timer_rsa_solver_dts_%d' % switch)
                    if v:
                        dts.append(v)

                prepare_rsa = run.get('timer_rsa_solver_prepare_rsa')
                rsa = run.get('timer_rsa_solver_run_rsa')
                if rsa > 400000:
                    print(rsa, run.get('param_topo_seed'), run.get('param_dts_look_ahead'))

                stats = run.get('timer_rsa_solver_update_statistics_dts') + run.get('timer_rsa_solver_update_statistics_rsa')

                t1 = run.get('timer_scenario_generator_create_flow_arrival_times')
                t2 = run.get('timer_scenario_generator_create_flow_demands')
                t3 = run.get('timer_scenario_generator_create_flows')
                t4 = run.get('timer_scenario_generator_create_topology')
                scenario = t1+t2+t3+t4
                precalc = run.get('timer_scenario_generator_precalculate_data_dts') +  run.get('timer_scenario_generator_precalculate_data_rts')
                precalc += prepare_rsa
                #print(dts, rsa, statistics, scenario, precalc)

                time_prepare_rsa.append(prepare_rsa)
                time_rsa.append(rsa)
                time_dts.append(sum(dts))
                time_dts_single.append(statistics.mean(dts))
                time_scenario.append(scenario)
                time_precalc.append(precalc)
                time_statistics.append(stats)
                timeout.append(0)

            time_dts = [x/1000. for x in  time_dts]
            time_rsa = [x/1000. for x in  time_rsa]
            time_dts_single = [x/1000. for x in  time_dts_single]
            time_statistics = [x/1000. for x in  time_statistics]
            time_scenario = [x/1000. for x in  time_scenario]
            time_prepare_rsa = [x/1000. for x in  time_prepare_rsa]
            time_precalc = [x/1000. for x in  time_precalc]

            if cnttimeout>0:
                axes[algo-1].vlines(900-cnttimeout, 0,1600,linestyle='--', color='black', linewidth=2)   
                print("cnttimeout", 900-cnttimeout)
            axes[algo-1].stackplot(np.arange(len(time_dts)),
                [time_dts, time_rsa, time_statistics, time_scenario, time_precalc, timeout], 
                labels=['DT-Select','RS-Alloc','Metrics', 'Scenario Generation', 'Preprocessing', 'Timeout'], 
                colors=['tab:blue', 'black', 'tab:orange', 'tab:green', 'tab:olive', 'lightgray'],
                alpha=0.7)
            axes[algo-1].legend(fontsize=15, loc='upper left')
        for ax in fig.axes:
            ax.set_xticks([0,100,200,300,400,500,600,700,800,899])
            ax.set_ylabel('Experiment execution time (s)', fontsize=15)
        axes[0].text(750, 800, 'Timeout', 
            fontsize=22, color='gray',
            verticalalignment='center', horizontalalignment='center', 
            alpha=1,
            #bbox=dict(boxstyle='square,pad=0.2',facecolor='white', edgecolor='red', alpha=1)
        )

        utils.export(fig, 'timings.pdf', folder='lookahead')
        print("timercnt", timercnt)

