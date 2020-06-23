import logging

from core.events import *

logger = logging.getLogger(__name__)

class FlowGenerator:
    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.topo = ctx.topo
        self.graph = ctx.topo.graph
        logger.debug("start flow generator")
        self.prepare_flows()
        
    def on_finished(self):
        pass

    def add_flow(self, flow, link):
        ev = EVLinkFlowAdd(flow.start, flow=flow, pcc=0)
        link.push_event(ev)
        self.ctx.flows.append(flow)

    def prepare_flows(self):
        generators = []
        for flow_spec in self.ctx.traffic:

            fg_class = flow_spec.get("fg_class")
            if not fg_class in generators: generators.append(fg_class)

            if (fg_class == "Single"):  
                from .gen_single import gen_single
                gen_single(self.ctx, self, **flow_spec)

            if (fg_class == "PBCE"):  
                from .gen_pbce import gen_pbce
                gen_pbce(self.ctx, self, **flow_spec)

            if (fg_class == "PBCE2"):  
                from .gen_pbce2 import gen_pbce2
                gen_pbce2(self.ctx, self, **flow_spec)

            if (fg_class == "Random"):  
                from .gen_random import gen_random
                gen_random(self.ctx, self, **flow_spec)

            if (fg_class == "gen_301"):
                from .gen_301 import gen_301
                gen_301(self.ctx, self, **flow_spec)

            if (fg_class == "gen_global1"):
                from .gen_global1 import gen_global1
                gen_global1(self.ctx, self, **flow_spec)

            if (fg_class == "gen_global2"):
                from .gen_global2 import gen_global2
                gen_global2(self.ctx, self, **flow_spec)

            if (fg_class == "gen_global3"):
                from .gen_global3 import gen_global3
                gen_global3(self.ctx, self, **flow_spec)

            if (fg_class == "gen_global4"):
                from .gen_global4 import gen_global4
                gen_global4(self.ctx, self, **flow_spec)

            if (fg_class == "gen_copy"):
                from .gen_copy import gen_copy
                gen_copy(self.ctx, self, **flow_spec)

            if (fg_class == "gen_rsa"):
                from .gen_rsa import gen_rsa
                gen_rsa(self.ctx, self, **flow_spec)


        self.ctx.statistics['traffic.generators'] = generators
        self.ctx.statistics['traffic.cnt_flows'] = len(self.ctx.flows)


