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
import random


import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
matplotlib.rcParams.update(params)

from . import agg_2_utils as utils


logger = logging.getLogger(__name__)


SAMPLES = 5000
SAMPLE_TRIES = 100
SAMPLE_SEED = 2


def plot(blob, **kwargs):
    """
    Select a subset out of 500.000 scenarios using certain criteria
    """

    utils.EXPORT_BLOB = blob
   
   
    blob.include_parameters(
        scenario_switch_cnt=1,
        hit_timelimit=1,
        scenario_table_capacity=1,
        scenario_table_capacity_reduction=1,
        scenario_table_util_max_total=1,
        #scenario_table_util_max_total_per_port=1,
        scenario_link_util_mbit_avg=1,
        scenario_link_util_mbit_max=1,
        scenario_bottlenecks=1,
        scenario_rules_per_switch_max=1,
        scenario_rules_total=1,
        timer_scenario_generator_precalculate_data_dts=1,
        scenario_hosts_per_switch_avg=1,
        scenario_hosts_per_switch_max=1,
        scenario_gen_param_topo_num_hosts=1,
        scenario_gen_param_topo_bottleneck_duration=1,
        scenario_gen_param_topo_bottleneck_cnt=1,
        scenario_gen_param_topo_bottleneck_intensity=1,
        scenario_gen_param_topo_concentrate_demand=1,
        scenario_gen_param_topo_concentrate_demand_retries=1,
        scenario_gen_param_topo_scenario_ba_modelparam=1,
        scenario_gen_param_topo_traffic_scale=1,
        scenario_gen_param_topo_traffic_interswitch=1,
        scenario_gen_param_topo_idle_timeout=1,
    )


    def plot_cdf(ax, runs, key, title=None, scale=None, **kwargs):
        datax = []
        datay = []
        total = len(runs)
        try:
            for i, run in enumerate(sorted(runs, key=lambda x: x.get(key))):
                if key == 'scenario_table_capacity_reduction':
                    datax.append(100-run.get(key))
                else:
                    datax.append(run.get(key))
                datay.append(i/total)
        except TypeError as e:
            for run in runs:
                if run.get(key) == None:
                    print(key, run.get(key), run)
                    raise e
            raise e

        ax.plot(datax, datay, **kwargs)
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        if title:
            ax.set_xlabel(title)
            ax.set_ylabel('CDF')
        else:
            ax.set_xlabel(key.replace('_', '\\_'))
        return max(datax)


    def get_quantils(runs, key):
        xvals = sorted([x.get(key) for x in runs])
        result = []
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            result.append(numpy.quantile(xvals, q))
        return result


    runs = blob.filter(**dict())
    
    timelimit = 0
    use_orig = []
    use_orig_timelimit = []
    use_runs = []
    use_seeds = []

    for run in runs:

        #if len(use_runs) == 10000:
        #    continue

        if run.get('hit_timelimit') and run.get('hit_timelimit')  > 0:
            timelimit += 1
            #run['hit_timelimit'] = 1
            use_orig_timelimit.append(run)
            continue
        else:
            run['hit_timelimit'] = run['timer_scenario_generator_precalculate_data_dts'] / 1000
            use_orig.append(run)
            

        if run.get('timer_scenario_generator_precalculate_data_dts') > 30000:
            continue;


        if run.get('scenario_table_capacity') < 1000 or run.get('scenario_table_capacity') > 3000:
            continue;

        if run.get('scenario_link_util_mbit_avg') > 1000:
            continue;

        if run.get('scenario_table_util_max_total') > 6000:
            continue;

        if run.get('scenario_rules_per_switch_max') > 200000:
            continue;


        run['timer_scenario_generator_precalculate_data_dts'] /= 1000
        use_runs.append(run)
        use_seeds.append(run.get('param_topo_seed'))

    print("scenarios", len(runs))

    # important parameters (those used to reduce the scenario set)
    ORIG = [
        dict(key='scenario_rules_per_switch_max', title=r'Maximum rules/switch'),
        dict(key='scenario_table_capacity', title=r'Flow table capacity'),
        dict(key='scenario_table_util_max_total', title=r'Maximum flow table utilization'),
        dict(key='scenario_link_util_mbit_avg', title=r'Average link utilization (Mbit/s)'),
    ]

    # -----------------------
    # Figure: show what was left out (CDF with red/green highlights)
    # -----------------------
    fig, axes = plt.subplots(3,2, figsize=(8,7), sharex=False, sharey=False)
    fig.tight_layout(pad=2.7)
    axcnt = 0
    for i, pdata in enumerate(ORIG):
        ax = fig.axes[i]
        key = pdata.get('key')
        maxval = plot_cdf(ax, use_orig, key , title=pdata.get('title'), scale=pdata.get('scale'),
            color='red', linewidth=2)  
        ax.set_xlim(0,maxval)
        for key2, barrier in [('scenario_rules_per_switch_max', 200000),
            ('scenario_table_util_max_total', 6000),
            ('scenario_link_util_mbit_avg', 1000),
            ('timer_scenario_generator_precalculate_data_dts', 30000)]:
            if key == key2:
                ax.vlines(barrier, 0, 1, linestyle = ':', color='black', alpha=1)
                ax.axvspan(0,barrier,0, 1, color="green", alpha=0.1)
                ax.axvspan(barrier,maxval,0, 1, color="red", alpha=0.1)
        if key == 'scenario_table_capacity':
            ax.vlines(1000, 0, 1, linestyle = ':', color='black', alpha=1)
            ax.vlines(3000, 0, 1, linestyle = ':', color='black', alpha=1)
            ax.axvspan(0,1000,0, 1, color="red", alpha=0.1)
            ax.axvspan(1000,3000,0, 1, color="green", alpha=0.1)
            ax.axvspan(3000,maxval,0, 1, color="red", alpha=0.1)

    # preprocessing time is done via timeout here (to include runs that suffered from a timeout)
    ax = fig.axes[-2]
    maxval = plot_cdf(ax, use_orig + use_orig_timelimit, 'hit_timelimit', r'Preprocessing time (s)', scale=pdata.get('scale'),
        color='red', linewidth=2)  
    ax.vlines(30, 0, 1, linestyle = ':', color='black', alpha=1)
    ax.axvspan(0,30,0, 1, color="green", alpha=0.1, label='Included in scenario\ngeneration process')
    ax.axvspan(30,maxval,0, 1, color="red", alpha=0.1, label='Not included')
    ax.set_xlim(0,maxval)

    # in the last ax, plot the legend
    ax = fig.axes[-1]
    handles, labels = fig.axes[-2].get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=1, fontsize=14)
    ax.patch.set_alpha(0)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False) 
    utils.export(fig, 'scenario_selection_precut_cdf1.pdf', folder='scenario_selection')
    plt.close()

    exit(1)




    # important parameters (those used to reduce the scenario set)
    PLOT = [
        dict(key='scenario_switch_cnt', title=r'Number of switches'),
        dict(key='scenario_rules_per_switch_max', title=r'Maximum rules/switch'),
        dict(key='scenario_rules_total', title=r'Rules total'),
        dict(key='scenario_table_capacity', title=r'Flow table capacity'),
        dict(key='scenario_table_util_max_total', title=r'Maximum flow table utilization'),
        dict(key='scenario_table_capacity_reduction', title=r'Capacity reduction (\%)'),
        dict(key='scenario_gen_param_topo_num_hosts', title=r'Number of hosts'),
        dict(key='scenario_link_util_mbit_avg', title=r'Average link utilization (Mbit/s)'),
        dict(key='timer_scenario_generator_precalculate_data_dts', title=r'Preprocessing time (s)'),
    ]

    # there are other parameters that are not that important
    PLOTOTHER = [
        dict(key='scenario_gen_param_topo_bottleneck_cnt', title=r'Number of bottlenecks'),
        dict(key='scenario_gen_param_topo_bottleneck_duration', title=r'Bottleneck duration'),
        dict(key='scenario_gen_param_topo_bottleneck_intensity', title=r'Bottleneck intensity'),
        dict(key='scenario_gen_param_topo_concentrate_demand', title=r'Hotspots'),
        dict(key='scenario_gen_param_topo_concentrate_demand_retries', title=r'Hotspot intensity'),
        dict(key='scenario_gen_param_topo_scenario_ba_modelparam', title=r'Parameter m'),
        dict(key='scenario_gen_param_topo_traffic_scale', title=r'Traffic scale parameter'),
        dict(key='scenario_gen_param_topo_traffic_interswitch', title=r'Interswitch rate'),
        dict(key='scenario_gen_param_topo_idle_timeout', title=r'Minium rule lifetime'),
        #dict(key='scenario_gen_param_topo_iat_scale', title=r'param_topo_iat_scale'), 
    ]

    # -----------------------
    # Figure: CDF of the original data (important parameters)
    # -----------------------
    fig, axes = plt.subplots(3,3, figsize=(8,7), sharex=False, sharey=False)
    fig.tight_layout(pad=2.7)
    axcnt = 0
    quantils_original = {}
    for i, pdata in enumerate(PLOT):
        plot_cdf(fig.axes[i], use_runs, pdata.get('key'), title=pdata.get('title'), scale=pdata.get('scale'),
            color='red', linewidth=2)  
        quantils_original[pdata.get('key')] = get_quantils(use_runs,  pdata.get('key'))
    
    utils.export(fig, 'scenario_selection_full_cdf1.pdf', folder='scenario_selection')
    plt.close()

    # -----------------------
    # Figure: CDF of the original data (other parameters)
    # -----------------------
    fig, axes = plt.subplots(3,3, figsize=(8,7), sharex=False, sharey=False)
    fig.tight_layout(pad=2.7)  
    axcnt = 0
    for i, pdata in enumerate(PLOTOTHER):
        plot_cdf(fig.axes[i], use_runs, pdata.get('key'), title=pdata.get('title'), scale=pdata.get('scale'),
            color='red', linewidth=2)  
    utils.export(fig, 'scenario_selection_full_cdf2.pdf', folder='scenario_selection')
    plt.close()

    # -----------------------
    # the sampling mechanism
    # -----------------------
    use_seeds = sorted(use_seeds)
    best_diff = 1000000000
    best_result = None
    best_seeds = None
    started = time.time()
    random.seed(SAMPLE_SEED)
    for tries in range(SAMPLE_TRIES):
        totaldiff = 0
        sampled_seeds = set()
        sampled_runs = []
        while len(sampled_seeds) < SAMPLES:
            sampled_seeds.add(random.choice(use_seeds))

        for run in use_runs:
            if run.get('param_topo_seed') in sampled_seeds:
                sampled_runs.append(run)
       
        for i, pdata in enumerate(PLOT):
            key = pdata.get('key')
            qold = quantils_original[key]
            qnew = get_quantils(sampled_runs, key)

            remove_candidates = []
            diff = 0
            for xold, xnew in zip(qold, qnew):
                for r in sampled_runs:
                    if xnew < xold:
                        if r.get(key) > xold:
                            remove_candidates.append(r.get('param_topo_seed'))
                        if (r.get(key) > xnew) and (r.get(key) < xold):
                            diff += 1
                    if xnew > xold:
                        if r.get(key) < xold:
                            remove_candidates.append(r.get('param_topo_seed'))
                        if (r.get(key) < xnew) and (r.get(key) > xold):
                            diff += 1
                totaldiff += diff

        if totaldiff < best_diff:
            print("> new best diff", totaldiff, time.time() - started)
            best_diff = totaldiff
            best_result = sampled_runs[:]
            best_seeds = sampled_seeds

    # -----------------------
    # Figure: CDF of the sampled data (1)
    # -----------------------
    fig, axes = plt.subplots(3,3, figsize=(8,7), sharex=False, sharey=False)
    fig.tight_layout(pad=2.7)  
    for i, pdata in enumerate(PLOT):
        key = pdata.get('key')
        ax = fig.axes[i]
        # the sampled result
        plot_cdf(ax, best_result, key, title=pdata.get('title'), scale=pdata.get('scale'),
            color='red', linewidth=2, label='Sampled scenario set') 
        # the original distribution
        plot_cdf(ax, use_runs, key, title=pdata.get('title'), scale=pdata.get('scale'),
            color='black', linewidth=1, linestyle='--', label='Original scenario set')  
        #qold = quantils_original[key]
        #qnew = get_quantils(sampled_runs, key)
        #colors = ['red', 'blue', 'green', 'orange', 'purple', 'magenta', 'cyan']
        #for xold, xnew, c in zip(qold, qnew, colors):
        #    ax.vlines(xnew, 0, 1, color='blue', alpha=0.2)
        #    ax.vlines(xold, 0, 1, linestyle = ':', color='blue', alpha=0.2) 
    fig.suptitle("Sampled scenario set (original=%d, sampled=%d, seed=%d)" % (len(use_runs), SAMPLES, SAMPLE_SEED), fontsize=16)
    fig.subplots_adjust(top=0.9, hspace=0.3)
    if len(sampled_seeds) > 0:
        filename = "sample-%.5d-seed-%d-tries-%d.txt" % (len(sampled_seeds), SAMPLE_SEED, SAMPLE_TRIES)
        print("export %s" % filename)
        text = ','.join(['%d' % x for x in sorted(sampled_seeds)])
        utils.export_textfile(text, filename, folder='scenarios')    
        utils.export(fig, 'scenario_selection_sample_%.5d_seed_%d_tries_%d_cdf1.pdf' % (len(sampled_seeds), SAMPLE_SEED, SAMPLE_TRIES), folder='scenario_selection')
        plt.close()

    # -----------------------
    # Figure: CDF of the sampled data (2)
    # -----------------------
    fig, axes = plt.subplots(3,3, figsize=(8,7), sharex=False, sharey=False)
    fig.tight_layout(pad=2.7)  
    for i, pdata in enumerate(PLOTOTHER):
        key = pdata.get('key')
        ax = fig.axes[i]
        # the sampled result
        plot_cdf(ax, best_result, key, title=pdata.get('title'), scale=pdata.get('scale'),
            color='red', linewidth=2, label='Sampled scenario set') 
        # the original distribution
        plot_cdf(ax, use_runs, key, title=pdata.get('title'), scale=pdata.get('scale'),
            color='black', linewidth=1, linestyle='--', label='Original scenario set')  
    fig.suptitle("Sampled scenario set (original=%d, sampled=%d, seed=%d)" % (len(use_runs), SAMPLES, SAMPLE_SEED), fontsize=16)
    fig.subplots_adjust(top=0.9, hspace=0.3)
    if len(sampled_seeds) > 0:    
        utils.export(fig, 'scenario_selection_sample_%.5d_seed_%d_tries_%d_cdf2.pdf' % (len(sampled_seeds), SAMPLE_SEED, SAMPLE_TRIES), folder='scenario_selection')
        plt.close()

