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

class gen_global2(object):

    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        time_range = 400
        seed = ctx.config.get("param_topo_seed")
        num_epoch = ctx.config.get("param_topo_epoch")
        num_flows = ctx.config.get("param_topo_num_flows")
        num_hosts = ctx.config.get("param_topo_num_hosts")

        print("num_flows", num_flows)
        # apply seed
        if seed:
            if seed > -1:
                random.seed(seed)

        # create host names
        sources = []
        for i in range(num_hosts):
            sources.append('h'+str(i))

        clusters = [1,2,3]

        INTER_CLUSTER = 0.15

        NUM_RULES = num_flows

        demand_classes = [1000,10000,400000] # short / medium / long flows

        # calculate flow rule durations and demands;
        durations = []
        demands = []
        for f in range(NUM_RULES):
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


        time_range = 400
        epoch_burst_probability = 0.1
        epoch_start = [0,0.2,0.4,0.6,0.8]
        epoch_end = [0.8,0.6,0.4,0.2,0]
        epoch_burst = [1.3,1.6,1.8,2,2.1]
        epoch_normal = [1,1.1,1.1,1.2,1.2]
        number_of_clusters = 3

        epoch_clusterweight = [
            [2,4,2,2,2,5,2,2,2,2],
            [2,2,2,3,1,1,2,2,2,2],
            [2,2,2,2,2,1,1,3,3,2],
        ]

        # calculate cluster epochs
        cluster_epoch = {}
        weights = []
        for cluster_id in range(number_of_clusters):
            # create an epoch for this cluster
            clusterweight = epoch_clusterweight[cluster_id]
            use_epoch = [] + epoch_start 
            for w in clusterweight:
                if random.random() < epoch_burst_probability:
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
            rules_per_cluster[cluster_id] = int(mypart * NUM_RULES)
        #if sum(flows_per_port) < num_flows:
        #    rules_per_cluster[-1] += (num_flows - sum(flows_per_port))



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
                    if random.random() < INTER_CLUSTER:
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

                    # create flow
                    self.generator.add_flow(Flow(self.ctx,  
                        label='%s->%s' % (source.label, target.label),
                        start=offset + random.uniform(0, span), 
                        demand_per_tick=demands[flow_counter],
                        duration=durations[flow_counter],
                        source=source,
                        target=target, 
                        flow_gen=flow_gen), source.link)
                    avg_duration += durations[flow_counter]
                    flow_counter += 1

            logger.info('cluster_id=%d (%d) current_flow_cnt=%d avg_duration=%.3f' % (
                cluster_id, src, flow_counter, avg_duration/sum(flowcnt_in_epoch)))