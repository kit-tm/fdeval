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

class gen_copy(object):
    """
    Takes an array of existing flow objects and creates an exact copy 
    """
    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator

        fg_params = kwargs.get('fg_params')

        flow_params = fg_params.get("flow_params")
        all_labels = fg_params.get('all_labels')

        logger.info("gen:gen_copy flows=%d" % len(flow_params))

        for flow in flow_params:
            
            src = flow.get('source_label')
            dst = flow.get('target_label')
            source = self.ctx.topo.get_host_by_label(src)
            target = self.ctx.topo.get_host_by_label(dst)
            #print(src, dst, source, target)
            # port identifier for pbce engine
            flow_gen = dict()
            params = dict(port=all_labels.index(src))
            flow_gen['fg_params'] = params

            self.generator.add_flow(Flow(self.ctx,  
                label=flow.get('label'),
                start=flow.get('start'), 
                demand_per_tick=flow.get('demand_per_tick'),
                duration=flow.get('duration'),
                source=source,
                target=target, 
                flow_gen=flow_gen), source.link) 
