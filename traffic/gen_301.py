import logging
import random
import math 
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from core.flow import Flow
from core.events import *

from plotter import flow_timeline

logger = logging.getLogger(__name__)

class gen_301(object):
    # PBCE style, but you can specify a source array where sources occur multiple times
    # to emulate nodes that have more traffic than others

    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        time_range = 400
        epoch_selection = [
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,3],
            [1,1,1,2,4,5,4,3,4,2,1,1,2,1,1],
            [1,1,1,2,4,5,10,3,4,2,1,1,2,1,1,1,1,1,1,1,2,2,2,2,2,2,3,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,1,1,1,1,5,5,5,10,10,25]
        ]
        epoch = epoch_selection[1] # default one
        epoch_start = [0,0.2,0.4,0.6,0.8]
        epoch_end = [0.8,0.6,0.4,0.2,0]
        seed = ctx.config.get("param_topo_seed")
        num_epoch = ctx.config.get("param_topo_epoch")
        num_flows = ctx.config.get("param_topo_num_flows")
        num_hosts = ctx.config.get("param_topo_num_hosts")

        if num_epoch >= 0:
            if num_epoch < len(epoch_selection):
                epoch = epoch_selection[num_epoch]

        demand_classes = [100,1000,20000] # short / medium / long flows

        print("num_flows", num_flows)
        # apply seed
        if seed:
            if seed > -1:
                random.seed(seed)

        # create host names
        sources = []
        for i in range(num_hosts):
            sources.append('h'+str(i))

        # calculate flow rule durations and demands;
        durations = []
        demands = []
        for f in range(num_flows):
            d = random.expovariate(1.0/5)
            if d > 30 or d < 3: 
                while d > 30 or d < 3: d = random.expovariate(1.0/5)
            durations.append(int(d))
            demand = demand_classes[0] + random.uniform(0, demand_classes[0]/2)
            if d > 5:
                # change to get more demand in case of higher duration
                if random.random() > 0.5: demand += demand_classes[1] + random.uniform(0, demand_classes[1]/2)
            if d > 10:
                if random.random() > 0.8: demand += demand_classes[1] + random.uniform(0, demand_classes[1]/2)
            if d > 20:
                if random.random() > 0.9: demand += demand_classes[2] + random.uniform(0, demand_classes[2]/2)
            demands.append(demand)
        
        #durations = list(sorted(durations))
        #fig, ax = plt.subplots(figsize=(16, 8))
        #ax.plot(np.arange(len(durations)), durations, color='green')     
        #plt.show()

        flows_per_port = []
        port_distribution = []
        for h in range(num_hosts):
            s = random.normalvariate(1, 0.1)
            port_distribution.append(s)
        for h in range(num_hosts):
            mypart = port_distribution[h]/sum(port_distribution)
            flows_per_port.append(int(mypart * num_flows))
        if sum(flows_per_port) < num_flows:
            flows_per_port[-1] += (num_flows - sum(flows_per_port))

        #print("sum", sum(flows_per_port))      
        #flows_per_port = list(sorted(flows_per_port))
        #fig, ax = plt.subplots(figsize=(16, 8))
        #ax.plot(np.arange(len(flows_per_port)), flows_per_port, color='green')     
        #plt.show()
        flow_counter = 0
        for h in range(num_hosts):
            cnt_flows = 0
            random.shuffle(epoch)
            use_epoch = epoch_start + epoch + epoch_end

            # distribute flows for this port over its epoch array
            span = time_range / len(use_epoch)
            flowcnt_in_epoch = []
            for i, e in enumerate(use_epoch):
                c = int((e/sum(use_epoch))*flows_per_port[h])
                flowcnt_in_epoch.append(c)
            diff = flows_per_port[h] - sum(flowcnt_in_epoch)
            while diff > 0:
                ix = random.choice(range(len(flowcnt_in_epoch)))
                flowcnt_in_epoch[ix] += 1
                diff -= 1

            # run through all epochs
            for i, e in enumerate(use_epoch):
                offset = float(i)*float(span)
                src = 'h'+str(h)
                for f in range(flowcnt_in_epoch[i]):
                    dst = src
                    while dst == src:
                        dst = random.choice(sources)
                    source = self.ctx.topo.get_host_by_label(src)
                    target = self.ctx.topo.get_host_by_label(dst)

                    if not source:
                        raise RuntimeError("source is none (kwargs=%s)" % str(kwargs))
                    if not target:
                        raise RuntimeError("target is none (kwargs=%s)" % str(kwargs))
                    if source == target:
                        raise RuntimeError("target and source are the same (kwargs=%s)" % str(kwargs))

                    # port identifier for pbce engine
                    flow_gen = dict(**kwargs)
                    params = dict(port=sources.index(src))
                    flow_gen['fg_params'] = params

                    # create flow
                    self.generator.add_flow(Flow(self.ctx,  
                        label='%s->%s' % (source.label, target.label),
                        start=offset + random.uniform(0, span), 
                        demand_per_tick=demands[flow_counter],
                        duration=durations[flow_counter],
                        source=source,
                        target=target, 
                        flow_gen=flow_gen), source.link)
                    flow_counter += 1