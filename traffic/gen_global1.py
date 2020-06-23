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

class gen_global1(object):

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

        allpeaks = [
            [10,50,150,200,250,300],
            [15,70,100,180],
            [60,130,220,270]
        ]

        for peaks in allpeaks:
            for peak in peaks:
                s = random.choice(clusters)
                d = s
                
                if random.random() > 0.5:
                    while d == s:
                        d = random.choice(clusters)

                srcPort = random.choice(range(num_hosts))
                dstPort = srcPort
                while dstPort == srcPort:
                    dstPort = random.choice(range(num_hosts))



                src = 's%dh%d' % (s, srcPort)
                dst = 's%dh%d' % (d, dstPort)  
                source = self.ctx.topo.get_host_by_label(src)
                target = self.ctx.topo.get_host_by_label(dst)
                # port identifier for pbce engine
                flow_gen = dict(**kwargs)
                params = dict(port=srcPort)
                flow_gen['fg_params'] = params

                 # create flows
                cnt = 0
                for f in range(0,100):
                    self.generator.add_flow(Flow(self.ctx,  
                        label='%s->%s' % (source.label, target.label),
                        start=peak+cnt, 
                        demand_per_tick=100,
                        duration=30,
                        source=source,
                        target=target, 
                        flow_gen=flow_gen), source.link) 
                    cnt += 0.1
        return




        patterns = [
            dict(src=3, dst=2, base=(10,300), peak=(100,30)),
            dict(src=1, dst=2, base=(25,300), peak=(200,30)),
            dict(src=2, dst=3, base=(15,300), peak=(50,30)),
            dict(src=3, dst=1, base=(20,300), peak=(170,30))
        ]

        for p in patterns:
            # base rules
            start, duration = p.get('base')
            for i in range(num_hosts):
                src = 's%dh%d' % (p.get('src'), int(i/2))
                dst = 's%dh%d' % (p.get('dst'), int(i/2))  
                source = self.ctx.topo.get_host_by_label(src)
                target = self.ctx.topo.get_host_by_label(dst)
                # port identifier for pbce engine
                flow_gen = dict(**kwargs)
                params = dict(port=i)
                flow_gen['fg_params'] = params

                 # create flows
                cnt = 0
                for f in range(0,10):
                    self.generator.add_flow(Flow(self.ctx,  
                        label='%s->%s' % (source.label, target.label),
                        start=start+cnt, 
                        demand_per_tick=100,
                        duration=duration,
                        source=source,
                        target=target, 
                        flow_gen=flow_gen), source.link) 
                    cnt += 1

            # peak rules
            start, duration = p.get('peak')
            for i in range(num_hosts):
                src = random.choice(range(num_hosts))
                dst = src
                while dst == src:
                    dst = random.choice(range(num_hosts))
                source = self.ctx.topo.get_host_by_label('s%dh%d' % (p.get('src'), src))
                target = self.ctx.topo.get_host_by_label('s%dh%d' % (p.get('src'), dst))
                # port identifier for pbce engine
                flow_gen = dict(**kwargs)
                params = dict(port=i)
                flow_gen['fg_params'] = params

                 # create flows
                cnt = 0
                for f in range(0,10):
                    self.generator.add_flow(Flow(self.ctx,  
                        label='%s->%s' % (source.label, target.label),
                        start=start+cnt, 
                        demand_per_tick=100,
                        duration=duration,
                        source=source,
                        target=target, 
                        flow_gen=flow_gen), source.link)
                    cnt += 1   