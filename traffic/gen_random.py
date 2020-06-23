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

class gen_random(object):
    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        fg_shuffle_epoch = kwargs.get("fg_shuffle_epoch")
        fg_numflows = kwargs.get("fg_numflows")
        fg_seed = kwargs.get("fg_seed")
        fg_random_destination = kwargs.get("fg_random_destination")
        fg_time_range = kwargs.get("fg_time_range")
        fg_epoch = kwargs.get("fg_epoch")

        if fg_seed:
            random.seed(fg_seed)
            
        reduce_scale = kwargs.get("fg_scale")

        iat = scipy.stats.expon()
        demand_samples = iat.rvs(size=fg_numflows) 
        duration_samples = iat.rvs(size=fg_numflows) 

        print(len(demand_samples))
        print(min(demand_samples), max(demand_samples), sum(demand_samples)/len(demand_samples))
        print(demand_samples) 

        print("duration >10", len([x for x in duration_samples if x > 3]))

        # shuffle the epoch values to "shift" the peaks around
        if fg_shuffle_epoch:
            np.random.shuffle(fg_epoch)

        epoch_cnt = [0]*len(fg_epoch)
        for i, v in enumerate(fg_epoch):
            epoch_cnt[i] = int(v/sum(fg_epoch)*fg_numflows)
        if sum(epoch_cnt) < fg_numflows:
            epoch_cnt[0] += fg_numflows-sum(epoch_cnt)

        print("epoch_cnt", epoch_cnt, sum(epoch_cnt))

        span = fg_time_range / len(fg_epoch)
        for epoch, _  in enumerate(fg_epoch):
            for i in range(0, epoch_cnt[epoch]):

                start = random.uniform(float(epoch)*float(span), float(epoch+1)*float(span))

                # random src/dst pair
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

                if duration_samples[i] > 3:
                    demand_per_tick = 1000 * demand_samples[i]
                else:
                    demand_per_tick = 100 * demand_samples[i]
                duration = 10 * duration_samples[i]
                if duration > 150:
                    duration = 150


                flow_gen = dict(**kwargs)
                params = dict(port=fg_random_destination.index(label_src))
                flow_gen['fg_params'] = params

                self.generator.add_flow(Flow(self.ctx,  
                    label='%s->%s' % (source.label, target.label),
                    start=start, 
                    demand_per_tick=demand_per_tick,
                    duration=duration,
                    #on_event=trace_flow_history, 
                    source=source,
                    target=target, 
                    flow_gen=flow_gen), source.link)
