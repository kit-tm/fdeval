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

class gen_pbce(object):
    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        fg_seed = kwargs.get("fg_seed")
        fg_random_destination = kwargs.get("fg_random_destination")
        fg_time_range = kwargs.get("fg_time_range")
        fg_epoch = kwargs.get("fg_epoch")

        if fg_seed:
            if fg_seed > -1:
                random.seed(fg_seed)
            
        reduce_scale = kwargs.get("fg_scale")

        span = fg_time_range / len(fg_epoch)
        for i, epoch in enumerate(fg_epoch):
            offset = float(i)*float(span)

            
            for i in range(math.floor(1000/reduce_scale*epoch)):

                label_src = random.choice(fg_random_destination)
                label_dst = label_src
                while label_dst == label_src:
                    label_dst = random.choice(fg_random_destination)

                source = self.ctx.topo.get_host_by_label(label_src)
                target = self.ctx.topo.get_host_by_label(label_dst)

                if not source:
                    raise RuntimeError("source is none (kwargs=%s)" % str(kwargs))
                if not target:
                    raise RuntimeError("target is none (kwargs=%s)" % str(kwargs))

                if source == target:
                    raise RuntimeError("target and source are the same (kwargs=%s)" % str(kwargs))

                # port identifier for pbce engine
                flow_gen = dict(**kwargs)
                params = dict(port=fg_random_destination.index(label_src))
                flow_gen['fg_params'] = params

                self.generator.add_flow(Flow(self.ctx,  
                    label='%s->%s' % (source.label, target.label),
                    start=offset + random.uniform(0, span), 
                    demand_per_tick=100,
                    duration=5.0*reduce_scale + random.uniform(0, 2),
                    #on_event=trace_flow_history, 
                    source=source,
                    target=target, 
                    flow_gen=flow_gen), source.link)

