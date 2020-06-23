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


def plot(blob, **kwargs):
    "Dataset Info"

    utils.EXPORT_BLOB = blob

    includes = ['scenario_switch_cnt', 'scenario_table_capacity',
        'scenario_concentrated_switches', 'scenario_edges', 'scenario_bottlenecks', 
        'scenario_hosts_of_switch']

    # extract these parameters and print statistics
    params = [
    'param_topo_num_switches',
    'param_topo_num_hosts',
    'param_topo_num_flows',
    'param_topo_switch_capacity',
    'param_topo_bottleneck_cnt',
    'param_topo_concentrate_demand',
    'param_topo_scenario_ba_modelparam',
    'param_topo_traffic_interswitch',
    'param_topo_traffic_scale',
    'param_dts_algo',   
    'param_dts_look_ahead',
    'param_dts_weight_table',
    'param_dts_weight_link',
    'param_dts_weight_ctrl', 
    'param_rsa_look_ahead',
    'param_rsa_max_assignments',
    'param_rsa_weight_table',
    'param_rsa_weight_link',
    'param_rsa_weight_ctrl',
    'param_debug_total_time_limit',
    'param_dts_timelimit',
    'param_rsa_timelimit'
    ]


    keywords = ['hit_timelimit', 
    'rsa_solver_cnt_infeasable',
    'scenario_gen_param_topo_num_flows'
    ] + params

    for keyword in keywords:
        includes += blob.find_columns(keyword)

    blob.include_parameters(**dict.fromkeys(includes, 1))

    runs = blob.filter(**dict())

    seeds = []
    unique_seeds = {}
    time_limit = 0
    cnt_rsa_solver_cnt_infeasable = 0

    arr_params = {}
    for param in params:
        arr_params[param] = []

    for run in runs:
        seed = run.get('param_topo_seed')
        unique_seeds[seed] = 1
        if run.get('hit_timelimit'):
            time_limit += 1
        if run.get('rsa_solver_cnt_infeasable') and run.get('rsa_solver_cnt_infeasable') > 0:
            cnt_rsa_solver_cnt_infeasable += rsa_solver_cnt_infeasable
        
        for param in params:
            if run.get('scenario_gen_%s' % param):
                arr_params[param].append(run.get('scenario_gen_%s' % param))
            else:
                if run.get(param):
                    arr_params[param].append(run.get(param))
                else:
                    arr_params[param].append(-1) 

    PRINT_DATA = []

    file = blob.db_statistics
    blobfolder = file.split('/')[-2]
    filesize = '%.2f' % float(os.path.getsize(file)/(1024*1024))

    print(">> dataset", blobfolder)
    print(">> filesize", filesize)
    print(">> experiments", len(runs))        
    print(">> time_limit", time_limit)
    print(">> unique_seeds", len(unique_seeds.keys()))    

    PRINT_DATA += [blobfolder, filesize, str(len(runs)), str(time_limit), str(len(unique_seeds.keys()))]


    for param in params:
        data = arr_params[param]
        if len(data) > 0:
            value = '-'

            if min(data) == max(data):
                if min(data) != -1:
                    value = '%d' % min(data)   
            if max(data) > min(data):

                value = '%d-%d' % ( max(0, min(data)), max(data))

            print(">>", param, value)  

            PRINT_DATA.append(value)


    csv = ','.join(PRINT_DATA)
    print(csv)

    #text = ''
    #text += '%d\\%%-%d\\%% & %.2f & %d & %.2f\\%% & %.2f\\%% & %.2f\\%%\\\\\n' % (x, y, 
    #    mean_ratio, len(failure_rate), ratio_0, ratio_01, ratio_1)
