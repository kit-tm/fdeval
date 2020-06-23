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

def trace_flow_history(ev, **kwargs):
    """
    WARNING: SLOW! ONLY FOR DEBUGGING / SMALL SCENARIOS!

    This function is executed in the on_event() callback of
    every flow, i.e., every time there is an event triggered for a flow, this
    function will create a trace entry for this event. Based on this trace,
    the flow_timeline.py plotter can create a nice visualization of what happens.
    """
    if isinstance(ev, EventPtr): 
        logger.warn("received EentPtr object in on_event flow-callback; was ignored")
        return; # should not happen

    link = kwargs.get("link")
    flow = ev.flow

    # add flow_history structure to store information
    if not hasattr(flow, 'flow_history'):
        setattr(flow, 'flow_history', [])
        pass

    demand = 0
    pcc = -1
    if hasattr(ev, 'pcc'):
        pcc = ev.pcc
    if hasattr(ev, 'demand_processed'):
        demand = ev.demand_processed

    flow.flow_history.append(dict(
        link=link.id,
        tick=ev.tick, # not event!!
        util=link.get_current_utilization(),
        event=ev.__class__.__name__,
        pcc=pcc,
        demand=demand
    ))

    linkname = flow.ctx.topo.print_link(link.id)
    print(" TRACE flow %s on link %s at tick %f; util=%f; pcc=%d, demand=%.2f" 
        %  (str(flow.id), linkname, ev.tick, 
            link.get_current_utilization(), pcc, demand), type(ev).__name__)


class gen_single(object):
    def __init__(self, ctx, generator, **kwargs):
        self.ctx = ctx
        self.generator = generator
        # parameters for class=FlowBurst

        fg_notrace = kwargs.get("fg_notrace")
        fg_label = kwargs.get("fg_label")
        fg_start = kwargs.get("fg_start")
        fg_demand = kwargs.get("fg_demand")
        fg_duration = kwargs.get("fg_duration")
        fg_fixed_path = kwargs.get("fg_fixed_path")
        fg_shortest_path = kwargs.get("fg_shortest_path")

        source = None
        target = None

        # path is given as (src, ..., dst)
        if fg_fixed_path:
            assert(len(fg_fixed_path) >= 2)
            source = self.ctx.topo.get_host_by_label(fg_fixed_path[0])
            target = self.ctx.topo.get_host_by_label(fg_fixed_path[-1])

        # shortest path is given as (src, dst)
        if fg_shortest_path:
            assert(len(fg_shortest_path) == 2)
            source = self.ctx.topo.get_host_by_label(fg_shortest_path[0])
            target = self.ctx.topo.get_host_by_label(fg_shortest_path[1])

        if not source:
            raise RuntimeError("source is none (kwargs=%s)" % str(kwargs))
        if not target:
            raise RuntimeError("target is none (kwargs=%s)" % str(kwargs))
        if source == target:
            raise RuntimeError("Seems you try to send traffic form src to dest with src==dst;" \
                " This is not allowed. Please change traffic configuration (kwargs=%s)" % str(kwargs))
        
        flow = Flow(self.ctx,  
            label=fg_label,
            start=fg_start, 
            demand_per_tick=fg_demand/fg_duration, 
            duration=fg_duration,
            on_event=trace_flow_history,
            source=source, 
            target=target,
            flow_gen=kwargs)

        if fg_notrace == 1:
            flow.on_event = None

        self.generator.add_flow(flow, source.link)
