import os, shutil
import logging, math, json, pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import time
import networkx as nx

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
matplotlib.rcParams.update(params)

EXPORT_FOLDER = os.path.join(os.getcwd(), 'data', 'plots')
if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)

EXPORT_BLOB = None # set by utils.configure(blob)

# for file names
def name_dts_solver(run):
    if run.get('param_dts_algo') == 1: return 'select_opt'
    if run.get('param_dts_algo') == 2: return 'select_copyfirst'
    if run.get('param_dts_algo') == 3: return 'select_greedy'
    return "unknownSolver"

def name_dts_weights(run):
    w1 = int(run.get('param_dts_weight_table'))
    w2 = int(run.get('param_dts_weight_link'))
    w3 = int(run.get('param_dts_weight_ctrl'))
    return "table_%d_link_%d_ctrl_%d" % (w1, w2, w3)

# for labels etc
def label_dts_solver(run):
    try:
        if run.get('param_dts_algo') == 1: return 'Select-Opt'
        if run.get('param_dts_algo') == 2: return 'Select-Copyfirst'
        if run.get('param_dts_algo') == 3: return 'Select-Greedy'
        return "unknownSolver"
    except:
        if run == 1: return 'Select-Opt'
        if run == 2: return 'Select-Copyfirst'
        if run == 3: return 'Select-Greedy'
        return "unknownSolver"     

def dts_weights(w1, w2, w3):
    return '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Table} = %d}$~~' \
        '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Link} = %d}$~~' \
        '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Ctrl} = %d}$ ' % (w1, w2, w3)

def rsa_weights(w1, w2, w3):
    return '$\\boxed{\\omega_\\texttt{\\small RSA}^\\textrm{\\small Table} = %d}$~~' \
        '$\\boxed{\\omega_\\texttt{\\small RSA}^\\textrm{\\small Link} = %d}$~~' \
        '$\\boxed{\\omega_\\texttt{\\small RSA}^\\textrm{\\small Ctrl} = %d}$ ' % (w1, w2, w3)

# label with all three objectives (as boxes)
def label_dts_objectives_all(run):
    w1 = int(run.get('param_dts_weight_table'))
    w2 = int(run.get('param_dts_weight_link'))
    w3 = int(run.get('param_dts_weight_ctrl'))
    return '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Table} = %d}$~~' \
        '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Link} = %d}$~~' \
        '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Ctrl} = %d}$ ' % (w1, w2, w3)

# label with one objective (assuming only one is set to 1)
def label_dts_objectives_one(run):
    w1 = int(run.get('param_dts_weight_table'))
    w2 = int(run.get('param_dts_weight_link'))
    w3 = int(run.get('param_dts_weight_ctrl'))
    if w1 == 1:   
        return '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Table} = %d}$' % (w1)
    if w2 == 1:   
        return '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Link} = %d}$' % (w2)
    if w2 == 1:   
        return '$\\boxed{\\omega_\\texttt{\\small DTS}^\\textrm{\\small Ctrl} = %d}$' % (w2)
    raise Exception("w1 or w2 or w3 have to be set to 1")

def get_parameter_string(run):
    solver = 'unknown'
    if run.get('param_dts_algo') == 1: solver =  'select_opt'
    if run.get('param_dts_algo') == 2: solver =  'select_copyfirst'
    if run.get('param_dts_algo') == 3: solver =  'select_greedy'
    objective = 'unknown'
    w1 = int(run.get('param_dts_weight_table'))
    w2 = int(run.get('param_dts_weight_link'))
    w3 = int(run.get('param_dts_weight_ctrl'))
    return "%s__tlc__%d-%d-%d" % (solver, w1, w2, w3)


def configure(blob):
    file = blob.db_statistics
    if os.path.exists(file):
        folder = file.split('/')[-2]
        print(folder)

    print("configure blob", )
    EXPORT_BLOB = blob

def export(fig, filename, folder=None):

    if EXPORT_BLOB:
        file = EXPORT_BLOB.db_statistics
        if os.path.exists(file):
            blobfolder = file.split('/')[-2]
            path = os.path.join(EXPORT_FOLDER, blobfolder)
            if folder is not None:
                path = os.path.join(path, folder)
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.join(path, filename)  

            if type(fig) == str:
                with open(filepath, 'w') as file:
                    file.write(fig)
                return

            fig.savefig(filepath, dpi=300)

        else:
            raise RuntimeError('EXPORT_BLOB set but file not found: %s' % file)

    else:
        # export without blob configured
        filepath = os.path.join(EXPORT_FOLDER, filename)
        if folder is not None:
            path = os.path.join(EXPORT_FOLDER, folder)
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.join(path, filename)  

        if type(fig) == str:
            with open(filepath, 'w') as file:
                file.write(fig)
            return

        fig.savefig(filepath, dpi=300)

def export_textfile(text, filename, folder=None):
    filepath = os.path.join(EXPORT_FOLDER, filename)
    if folder is not None:
        path = os.path.join(EXPORT_FOLDER, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)  

    if type(text) == str:
        with open(filepath, 'w') as file:
            file.write(text)
        return
    
def export_scenario(run):

    DIR = "/home/bauer/Code/PipeFlex/plugins/de.bauer.research.delegation/standalone/eval/data"

    TMP = os.path.join(EXPORT_FOLDER, 'scenarios', 'tmp')
    if not os.path.exists(TMP):
        os.makedirs(TMP)


    figure = run.get('topo_figure')
    topo = os.path.join(DIR, figure.replace('/home/bauer/eval/', ''))
    detailed = topo.replace("topo.pdf", "scenario.pdf")

    new_filename = 'switches_%d_hosts_%d_bottlenecks_%d_m_%d_hotspots_%d.pdf' % (
        run.get('param_topo_num_switches'),   
        run.get('param_topo_num_hosts'),
        run.get('param_topo_bottleneck_cnt'),
        run.get('param_topo_scenario_ba_modelparam'),
        run.get('param_topo_concentrate_demand')
    )
    shutil.copyfile(detailed, os.path.join(TMP, new_filename))
    #os.system("evince %s" % detailed)


def plotcdf(ax, data, **kwargs):
    total = len(data)
    data_summed = {}
    for v in data:  
        try:
            data_summed[v] += 1
        except KeyError:
            data_summed[v] = 1
    vsum = 0
    data_x = []
    data_y = []
    for x, v in sorted(data_summed.items()):
        vsum += v
        data_x.append(x)
        data_y.append(vsum/total)    
    ax.plot(data_x, data_y,  **kwargs)

        
def plot_dts_utilization_over_time(blob, run, switch, filter=None):

    assert(run.get('scenario_switch_cnt') >= 1)
    assert(run.get('param_topo_seed'))
    assert(run.get('param_dts_weight_table') >= 0)
    assert(run.get('param_dts_weight_link') >= 0)
    assert(run.get('param_dts_weight_ctrl') >= 0)

    if not run.get('dts_0_table_datax'):
        print("fetch data")
        includes = ['scenario_switch_cnt', 'scenario_table_capacity']
        for switch_cnt in range(0, run.get('scenario_switch_cnt')):
            includes.append('dts_%d_ctrl_overhead' % switch_cnt)
            includes.append('dts_%d_link_overhead' % switch_cnt)
            includes.append('dts_%d_table_overhead ' % switch_cnt)   
            d1 = 'dts_%d_table_datax' % (switch_cnt)
            d2 = 'dts_%d_table_datay_raw' % (switch_cnt)
            d3 = 'dts_%d_table_datay' % (switch_cnt)
            d4 = 'dts_%d_table_datay_shared' % (switch_cnt)
            m1 = 'dts_%d_table_overhead_percent' % (switch_cnt)
            m2 = 'dts_%d_link_overhead_percent' % (switch_cnt)
            m3 = 'dts_%d_ctrl_overhead_percent' % (switch_cnt)
            m4 = 'dts_%d_underutil_percent' % (switch_cnt)
            includes += [d1,d2,d3,d4,m1,m2,m3,m4]

        blob.include_parameters(**dict.fromkeys(includes, 1))
        runs = blob.filter(param_topo_seed=run.get('param_topo_seed'))
        use_run = None
        for run in runs:
            try:
                for k, v in filter.items():
                    assert(run.get(k) == v)
                assert(use_run == None)
                use_run = run
            except:
                pass
        assert(use_run != None)
    else:
        raise Exception()


    d1 = use_run.get('dts_%d_table_datax' % (switch))
    d2 = use_run.get('dts_%d_table_datay_raw' % (switch))
    d3 = use_run.get('dts_%d_table_datay' % (switch))
    d4 = use_run.get('dts_%d_table_datay_shared' % (switch))
    m1 = use_run.get('dts_%d_table_overhead_percent' % (switch))
    m2 = use_run.get('dts_%d_link_overhead_percent' % (switch))
    m3 = use_run.get('dts_%d_ctrl_overhead_percent' % (switch))
    m4 = use_run.get('dts_%d_underutil_percent' % (switch))


    #fig, axarr = plt.subplots(4,2, figsize=(8, 12))
    #fig.tight_layout(pad=2.7)  

    fig, (ax, ax2) = plt.subplots(2,1, figsize=(10, 6), gridspec_kw = {'height_ratios':[3, 1]})
    #figures.append((fig, dict(solver=solver, objective=objective)))
    fig.tight_layout(h_pad=-1.5, pad=2.9) 

    v1 = '%.2f' % float(m1) + r'\%'
    v2 = '%.2f' % float(m2) + r'\%'
    v3 = '%.2f' % float(m3) + r'\%'
    v4 = '%.2f' % float(m4) + r'\%'
    #metrics_ds_overhead_percent

    params = 'Table overhead : %s \n' % (v1)
    params += 'Link overhead : %s\n' % (v2)
    params += 'Control overhead : %s\n' % (v3)
    params += 'Underutilization : %s' % (v4)

    threshold = int(use_run.get("scenario_table_capacity"))
    """
    ax.text(0.01, 0.97, r''+params, fontsize=12, 
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, color='black',
        bbox=dict(facecolor='white', edgecolor='white'))
    """
    ds, = ax.plot(d1, d3, color="black", linewidth=1.5)

    ax.fill_between(d1, d3, 0, interpolate=True, color='orange', alpha=0.3)

    total, = ax.plot(d1, d2, 
        linestyle=':', color='blue', label='Utilization without flow delegation', linewidth=1)


    # plot threshold
    t1 = ax.hlines(threshold, 0, 450, color='black', 
        label="Flow table capacity", linestyle='--', linewidth=1.5)

    """
    fill_overutil = [True if x > threshold else False for x in cnt_active_flows]
    ax.fill_between(np.arange(len(cnt_active_flows)), cnt_active_flows, [threshold]*len(cnt_active_flows),
        where=fill_overutil, interpolate=True, color='orange', alpha=0.3, label='Utilization with flow delegation')
    
    fill_underutil = [True if x < threshold and x+e > threshold else False for x, e in zip(cnt_active_flows, cnt_active_flows_evicted)]
    ax.fill_between(np.arange(len(cnt_active_flows)), cnt_active_flows, [threshold]*len(cnt_active_flows), 
        where=fill_underutil, interpolate=True, color='red', alpha=1, label='Underutilization')
    

    # legend
    red = patches.Patch(color='red',  alpha=0.3, label='The red data')
    green = patches.Patch(color='red',  alpha=1, label='The red data')
    #orange = patches.Patch(color='orange',  alpha=0.5, label='The red data')
    ax.legend(loc=1, ncol=2)
    
    for x in range(50,450,50):
        ax.vlines(x, 0, 1150, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax2.vlines(x, 0, 22, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    """
    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    #ax.set_ylim(0,1150)
    #ax.set_xlim(0,400)
    ax.set_title('%s ~~~ %s' % (
        label_dts_solver(use_run), label_dts_objectives_all(use_run)), fontsize=17,y=0.95)
    ax.set_ylabel(r'\#rules in flow table', fontsize=14)
    ax.xaxis.set_ticks_position('none') 
    ax.set_xticks([], [])
    ax.set_yticks((250,500,750,1000))
    ax.set_yticklabels(('250','500','750','1000'))

    #ax.set_xlabel(r'\textbf{time (s)}')
    """
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
    """
    plt.show()

    return fig

    #ax2.tick_params(axis='both', which='major', labelsize=14)

    #plt.legend()

    # kwargs.get('exportdir')

    filename = "%d_%s_%s.pdf" % (threshold, name_dts_solver(use_run), name_dts_weights(use_run))
    export(fig, filename, folder='dts_functional_001')
    #plt.show()
    plt.close()

def plot_topo_small(ax, hosts_of_switch, edges, switches, concentrated_switches, **kwargs):
    """
    Plots the scenario topology
    see https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
    """
    switch_node_size = kwargs.get('switch_node_size', 50)
    font_size = kwargs.get('font_size', 8)
    highlight_blue = kwargs.get('highlight_blue', [])
    highlight_one = kwargs.get('highlight_one', [])
    highlight_other = [] # neighbors of highlight_one
    highlight_edge = [] # edges of highlight_one
    gnew = nx.Graph() # create a copy for the visualization
    gnew.add_nodes_from(switches)
    gnew.add_edges_from(edges)
    switch_cnt = len(switches)

    # calculate highlighted nodes/edges if highlight_one is set
    if len(highlight_one) > 0:
        assert(len(highlight_one) == 1)
        for n in gnew.neighbors(highlight_one[0]):
            highlight_other.append(n)
            highlight_edge.append((n, highlight_one[0]))
            highlight_edge.append((highlight_one[0], n))

    # now add the hosts (not present in g)
    for switch, hosts in hosts_of_switch.items():
        for h in hosts:
            gnew.add_node('h%d' % h)
            gnew.add_edge(switch, 'h%d' % h)

    node_size = []
    color_map = []
    edge_color_map = []
    label_map = {}
    style = []
    width = []
    for node in gnew:
        try:
            if node < switch_cnt:
                if node in highlight_one:
                    color_map.append('red')       
                elif node in highlight_other:
                    color_map.append('green')
                elif node in concentrated_switches:
                    color_map.append('red') # lightcoral
                elif node in highlight_blue:
                    color_map.append('lightgreen')    
                else:
                    color_map.append('black')
                node_size.append(switch_node_size)
                label_map[node] = node
        except:
            color_map.append('blue')
            node_size.append(5) 
            #label_map.append(' ')


    for n1, n2 in gnew.edges():
        try:
            if n1 < switch_cnt and n2 < switch_cnt:
                if (n1, n2) in highlight_edge:
                    edge_color_map.append('green')
                else:
                    edge_color_map.append('black') # lightblue
                style.append('solid')
                width.append(4.0)
            else:
                edge_color_map.append('lightgray')
                style.append('dashed')
        except TypeError:
            edge_color_map.append('blue')   
            style.append('dashed')
            width.append(0.5)


    draw_params = dict(node_color=color_map, labels=label_map, font_color="white", 
            font_size=font_size,edge_color=edge_color_map,node_size=node_size,style=style,width=width)
    if ax:
        draw_params['ax'] = ax
        nx.draw_kamada_kawai(gnew, **draw_params)          
    else:
        plt.close()     
        nx.draw_circular(gnew, **draw_params)
        plt.show()




if __name__ == "__main__":
    # execute only if run as a script
    fig, axes = plt.subplots(6,3, figsize=(10, 14), sharex=False, sharey=False)
    fig.tight_layout(pad=0)
    edges = [[0,2], [0,4], [4,3], [1,2]]
    hosts_of_switch = {0: [3, 11, 12, 13, 20, 27, 28, 31, 35, 39, 43, 44, 45, 49, 51, 56, 59, 61, 67, 71, 73, 83], 1: [0, 8, 10, 14, 22, 24, 29, 30, 34, 55, 57, 65, 75, 81, 84, 87, 91, 93, 96], 2: [1, 6, 33, 36, 46, 47, 48, 58, 62, 66, 77, 86, 88, 99], 3: [2, 4, 5, 7, 9, 18, 25, 37, 40, 41, 42, 50, 54, 60, 68, 69, 70, 72, 76, 78, 85, 89, 92, 94, 98], 4: [15, 16, 17, 19, 21, 23, 26, 32, 38, 52, 53, 63, 64, 74, 79, 80, 82, 90, 95, 97]}
    plot_topo_small(fig.axes[0], hosts_of_switch, edges, [0,1,2,3,4], [3])
    plt.show()
