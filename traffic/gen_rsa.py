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

class gen_rsa(object):
    """
    Takes an array of existing flow objects and creates an exact copy 
    """
    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        logger.info("gen:gen_rsa flows=%d" % len(self.ctx.scenario.all_flow_rules))

        for flow in self.ctx.scenario.all_flow_rules:
            
            src = flow.get('source_label')
            dst = flow.get('target_label')
            source = self.ctx.topo.get_host_by_label(src)
            target = self.ctx.topo.get_host_by_label(dst)

            # port identifier for pbce engine
            flow_gen = dict(
                fg_params=dict(port=flow.get('path'))
            )

            self.generator.add_flow(Flow(self.ctx,  
                label=flow.get('label'),
                start=flow.get('start'), 
                demand_per_tick=flow.get('demand_per_tick'),
                duration=flow.get('duration'),
                source=source,
                target=target, 
                flow_gen=flow_gen), source.link) 
