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

class gen_global4(object):

    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        time_range = 400
        seed = ctx.config.get("param_topo_seed")
        num_switches = ctx.config.get("param_topo_num_switches", 3)
        num_epoch = ctx.config.get("param_topo_epoch")
        num_flows = ctx.config.get("param_topo_num_flows")
        num_hosts = ctx.config.get("param_topo_num_hosts")
        traffic_scale = ctx.config.get("param_topo_traffic_scale", 1000)

        print("num_flows", num_flows, "num_switches", num_switches)
        # apply seed
        if seed:
            if seed > -1:
                random.seed(seed)

        # create host names
        sources = []
        for i in range(num_hosts):
            sources.append('h'+str(i))

        clusters = [x+1 for x in range(num_switches)]
        number_of_clusters = num_switches
        INTER_CLUSTER = 0.15

        DEMAND_CLASSES = [1000,20000,400000] # short / medium / long flows
        DEMAND_CLASSES = [500,10000,100000] # short / medium / long flows
        
        DEMAND_CLASSES = [x * (traffic_scale/1000) for x in DEMAND_CLASSES]

        EPOCH_CYCLE = 1

        EPOCH_BURST_PROPABILITY = 0.1

        RANDOMIZE_CLUSTERWEIGHTS = False # should be set to True if EPOCH_CYCLE > 1 to randomize traffic inside the epochs

        # calculate flow rule durations and demands;
        durations = []
        demands = []
        for f in range(num_flows):
            d = random.expovariate(1.0/5)
            if d > 30 or d < 3: 
                while d > 30 or d < 3: d = random.expovariate(1.0/5)
            durations.append(int(d))
            demand = DEMAND_CLASSES[0] + random.uniform(0, DEMAND_CLASSES[0]/2)
            if d > 5:
                # change to get more demand in case of higher duration
                if random.random() > 0.5: demand += DEMAND_CLASSES[1] + random.uniform(0, DEMAND_CLASSES[1]/2)
            if d > 10:
                if random.random() > 0.8: demand += DEMAND_CLASSES[1] + random.uniform(0, DEMAND_CLASSES[1]/2)
            if d > 20:
                if random.random() > 0.9: demand += DEMAND_CLASSES[2] + random.uniform(0, DEMAND_CLASSES[2]/2)
            demands.append(demand)


        time_range = 400
        epoch_start = [0,0.2,0.4,0.6,0.8]
        epoch_end = [0.8,0.6,0.4,0.2,0]
        epoch_burst = [1.3, 1.35, 1.4, 1.45, 1.5]
        epoch_normal = [1, 1.12, 1.14, 1.16, 1.18]
        

        epoch_clusterweight = [
            [2,3,2,2,2,4,2,2,2,2],
            [2,2,2,3,1,1,2,2,2,2],
            [2,2,2,2,2,1,1,3,3,2],
        ]

        # calculate cluster epochs
        cluster_epoch = {}
        weights = []
        for cluster_id in range(number_of_clusters):
            # create an epoch for this cluster
            clusterweight = epoch_clusterweight[cluster_id % len(epoch_clusterweight)] * EPOCH_CYCLE
            if RANDOMIZE_CLUSTERWEIGHTS:
                random.shuffle(clusterweight)
            use_epoch = [] + epoch_start 
            for w in clusterweight:
                if random.random() < EPOCH_BURST_PROPABILITY:
                    # use burst epoch
                    random.shuffle(epoch_burst)
                    use_epoch += [x * w for x in epoch_burst]
                else:
                    # use normal epoch
                    random.shuffle(epoch_normal)
                    use_epoch += [x * w for x in epoch_normal]             
            use_epoch += epoch_end

            logger.info('cluster_id=%d epoch_length=%d, epoch_sum=%d' % (
                cluster_id, len(use_epoch), sum(use_epoch)))
            weights.append(sum(use_epoch))
            cluster_epoch[cluster_id] = use_epoch

        #calculate rules per cluster
        rules_per_cluster = {}
        for cluster_id in range(number_of_clusters):
            mypart = sum(cluster_epoch[cluster_id])/sum(weights)
            rules_per_cluster[cluster_id] = int(mypart * num_flows)
        #if sum(flows_per_port) < num_flows:
        #    rules_per_cluster[-1] += (num_flows - sum(flows_per_port))

        set_all = []
        set_inter = {}
        flow_counter = 0
        for cluster_id in range(number_of_clusters):


            # distribute rules for this cluster over its epoch array
            span = time_range / len(cluster_epoch[cluster_id])
            flowcnt_in_epoch = []
            for i, e in enumerate(cluster_epoch[cluster_id]):
                c = int((e/sum(cluster_epoch[cluster_id]))*rules_per_cluster[cluster_id])
                flowcnt_in_epoch.append(c)
            diff = rules_per_cluster[cluster_id] - sum(flowcnt_in_epoch)
            while diff > 0:
                ix = random.choice(range(len(flowcnt_in_epoch)))
                flowcnt_in_epoch[ix] += 1
                diff -= 1
            logger.info('cluster_id=%d flowcnt_in_epoch=%d' % (
                cluster_id, sum(flowcnt_in_epoch)))

            # run through all epochs
            src = cluster_id+1 # s_1...s_c
            
            avg_duration = 0
            for i, e in enumerate(cluster_epoch[cluster_id]):
                offset = float(i)*float(span)
                for f in range(flowcnt_in_epoch[i]):
                    dst = src # intra cluster traffic
                    is_inter_flow = False
                    if random.random() < INTER_CLUSTER:
                        is_inter_flow = True
                        while dst == src :
                            # inter cluster traffic
                            dst = random.choice(clusters)

                    srcPort = random.choice(range(num_hosts))
                    dstPort = srcPort
                    while dstPort == srcPort:
                        dstPort = random.choice(range(num_hosts)) 
                        
                    src_label = 's%dh%d' % (src, srcPort)
                    dst_label = 's%dh%d' % (dst, dstPort)  
                    source = self.ctx.topo.get_host_by_label(src_label)
                    target = self.ctx.topo.get_host_by_label(dst_label)

                    # port identifier for pbce engine
                    flow_gen = dict(**kwargs)
                    params = dict(port=srcPort)
                    flow_gen['fg_params'] = params

                    demand = demands[flow_counter]
                    #if is_inter_flow:
                    #    demand = demand * 2

                    new_flow = (Flow(self.ctx,  
                        label='%s->%s' % (source.label, target.label),
                        start=offset + random.uniform(0, span), 
                        demand_per_tick=demand,
                        duration=durations[flow_counter],
                        source=source,
                        target=target, 
                        flow_gen=flow_gen), source.link)

                    if is_inter_flow:
                        try:
                            set_inter['%d->%d' % (src, dst)].append(new_flow)
                        except KeyError:
                            set_inter['%d->%d' % (src, dst)] = [new_flow]
                    set_all.append(new_flow)

                    # create flow
                    
                    avg_duration += durations[flow_counter]
                    flow_counter += 1

            logger.info('cluster_id=%d (%d) current_flow_cnt=%d avg_duration=%.3f' % (
                cluster_id, src, flow_counter, avg_duration/sum(flowcnt_in_epoch)))

        logger.info('adjust demands for inter-cluster flows')
        sum_flows = 0
        for key, flows in set_inter.items():
            sum_flows += len(flows)
            
            max_demand = 0
            check_demand = 0



            for t in range(1, 499):
                active = [flow for flow, _ in flows if flow.start < t and flow.start+flow.duration > t]
                demands = [f.demand_per_tick for f in active]
                if sum(demands) > max_demand:
                    max_demand = sum(demands)

            if max_demand > 1000000:
                for f, _ in flows:
                    f.demand_per_tick = f.demand_per_tick / (max_demand/1000000)

                for t in range(1, 499):
                    active = [flow for flow, _ in flows if flow.start < t and flow.start+flow.duration > t]
                    demands = [f.demand_per_tick for f in active]
                    if sum(demands) > check_demand:
                        check_demand = sum(demands)


            logger.info('%s : %d (sum=%d) max_demand=%f check_demand=%f' % (key, len(flows), sum_flows, max_demand, check_demand))

        #raise RuntimeError('bam')

        for flow, source in set_all:
            self.generator.add_flow(flow, source) 
