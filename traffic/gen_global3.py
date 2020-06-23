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

class gen_global3(object):

    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        seed = ctx.config.get("param_topo_seed")
        random.seed(seed)

        alldata = ['s1h0', 's1h1', 's1h2','s2h0', 's2h1', 's2h2','s3h0', 's3h1', 's3h2']
        for i in range(500):

            src = random.choice(alldata)
            dst = src
            while dst == src:
                dst = random.choice(alldata)
            source = self.ctx.topo.get_host_by_label(src)
            target = self.ctx.topo.get_host_by_label(dst)

            self.generator.add_flow(Flow(self.ctx,  
            label='%s->%s' % (source.label, target.label),
            start=random.randint(10,250) + random.random(), 
            demand_per_tick=1,
            duration=random.randint(10,50) + random.random(),
            source=source,
            target=target, 
            flow_gen={}), source.link)


        return




