import logging, math, json, pickle, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import time

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
matplotlib.rcParams.update(params)

from . import agg_2_utils as utils


logger = logging.getLogger(__name__)

metrics = ['', 'Underutilization', 'Link overhead', 'Table overhead', 'Control message overhead']
colors = ['', 'red', 'green', 'blue', 'purple']
markers = ['', '^', 's', '+', 'o']


def plot(blob, **kwargs):
    """DTS functional evaluation"""

    success = 0
    failure = 0
    failuresum = 0

    for switch_cnt in [1,2]:
        d1 = 'dts_%d_verify_diff_ctrl_overhead' % (switch_cnt-1)
        d2 = 'dts_%d_verify_diff_total_delegated_demand_in' % (switch_cnt-1)
        d3 = 'dts_%d_verify_diff_total_demand_in' % (switch_cnt-1)
        check = [d1,d2,d3]
        blob.include_parameters(
            **{
            d1 : 1,
            d2 : 1,
            d3 : 1
            }
        )
        runs = blob.filter(**dict(param_topo_num_switches=switch_cnt))
        print("runs with switch_cnt=%d: %d" % (switch_cnt, len(runs)))
        major_error_cnt = 0
        total_errors = 0
        total_error_sum = []
        total_checked = 0
        for run in runs:
            total_checked += 1
            for c in check:
                result = run.get(c)
                
                if result > 0:
                    total_errors += 1
                    total_error_sum.append(result)
                    #print("    >>", c, result)
                if result > 0.1:
                    rerun = "\n"
                    for k,v in run.items():
                        if k.startswith('param'):
                            rerun += "%s %d\n" % (k, v)
                    path = os.path.join(blob.path, 'verify_errors')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    filename = os.path.join(path, 'error_%d.txt' % major_error_cnt)
                    major_error_cnt += 1
                    print("major error: ", filename)
                    with open(filename, 'w') as file:
                        file.write(rerun)


        print("  total_errors", total_errors)
        print("  total_major_errors", major_error_cnt)
        print("  total_error_sum", sum(total_error_sum))
        print("  total_checked", total_checked)
    return



    # older version

    for run in runs:
        v1 = run.get('verify_diff_total_demand_in')
        v2 = run.get('verify_diff_total_demand_out')
        v3 = run.get('verify_diff_total_delegated_demand_in')
        v4 = run.get('verify_diff_ctrl_overhead')
        if v1+v2+v3+v4 == 0:
            success += 1
        else:
            failure += 1
            print(v1, v2, v3, v4)
            failuresum += (v1+v2+v3+v4)
    print("success", success)
    print("failure", failure)
    print("failuresum", failuresum)

    return
