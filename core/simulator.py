import logging
import random
import time
from heapq import heappush, heappop, nsmallest

from core.events import *
from core.host import Host
from core.switch import Switch
from core.link import Link
from core.flow import Flow
from core.metrics import calculate_standard_metrics

from plotter import flow_timeline
from traffic.generator import FlowGenerator

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.events = []
        self.tick = -1
        self.ctx.sim = self

    def push_event(self, ev, handler):
        """Add new event to the main simulation heap"""
        ev.handler = handler # store the handler inside the event
        if ev.tick < self.tick:
            raise RuntimeError("event.tick < self._tick_current")
        heappush(self.events, ev)  

    def is_minheap(self):
        """Check minheap condition (debugging only, not used atm)"""
        return all(self.events[i] >= self.events[(i-1)//2] for i in range(1, len(self.events)))

    def setup(self):
        # we first have to create all the link and switch objects for the simulation
        # switches are attached to networkx node as attribute "_switch"
        # links are attached to networkx edge as attribute "_link"
        # hosts are attached to networkx node as attribute "_host"
        logger.debug("create toplogy objects")
        cnt_switches = 0
        cnt_links = 0
        cnt_hosts = 0

        for idFrom, idTo, opts in self.ctx.topo.graph.edges(data=True):
            self.ctx.topo.graph.edges[idFrom, idTo]['_link'] = Link(self.ctx, id=(idFrom, idTo), **opts)
            cnt_links += 1
        
        for id, opts in self.ctx.topo.graph.nodes(data=True):
            if opts.get("isSwitch"):
                logger.debug(".. create switch: %s" % str(opts))
                # each switch has a routing/forwarding engine
                self.ctx.topo.graph.nodes[id]['_switch'] = Switch(self.ctx, id=id, **opts)
                cnt_switches += 1
            else:
                # nodes that are not labeled as switches are considered end systems
                logger.debug(".. create end system %s" % str(opts))
                neighbors = list(self.ctx.topo.graph.neighbors(id))
                if len(neighbors) == 1:
                    # end systems are connected via exactly one link
                    link = self.ctx.topo.graph.edges[id,neighbors[0]]['_link']
                    self.ctx.topo.graph.nodes[id]['_host'] = Host(self.ctx, link, id=id, **opts)
                    cnt_hosts += 1
                else:
                    logger.error("graph contains host-node with multiple neighbors; not supported (skipped)")

        logger.debug("create toplogy objects done; switches=%d links=%d hosts=%d" % (cnt_switches, cnt_links, cnt_hosts))

        # next step is to create the traffic
        flowGen = FlowGenerator(self.ctx)

        # handle global on_simulation_setup_complete callback registered in ctx
        if self.ctx.on_simulation_setup_complete:
            self.ctx.on_simulation_setup_complete(self.ctx) 

    def run(self):

        self.setup()

        # get all relevant objects
        switches = [opts['_switch'] for _, opts in self.ctx.topo.graph.nodes(data=True) if opts.get('_switch')]
        links = [opts['_link'] for _, _, opts in self.ctx.topo.graph.edges(data=True) if opts.get('_link')]
        objects = switches + links

        # insert stats
        # the small offset ensures that the switch stats operations are always
        # executed in the same order, i.e., switch0 always calculates its stats
        # before switch1 and so on
        for i in range(0, 450, 1):
            offset = 0
            for s in switches:
                s.register_stats(i + (offset * 0.00000001)) #  
                offset += 1

        # add custom events from ctx (if defined)
        if len(self.ctx._events) > 0:
            pass

        t = time.time()
        active = True
        cnt_events = 0
        while active:

            if len(self.events) > 0:
                ev = heappop(self.events)
                if ev.tick < self.tick:
                    raise RuntimeError('min_tick - tick < 0')
                cnt_events += 1
                self.tick = ev.tick
                ev.handler.on_event(ev)
            else:
                active = False

        # handle on_simulation_finished callbacks; This is useful for
        # debugging/plotting in conjuction with the on_event callback in the flows
        for obj in objects:
            if obj.on_simulation_finished:
                obj.on_simulation_finished(self.tick)

        simulated_time = time.time()-t
        logger.info("done tick=%d processed_events=%d time=%f" % (self.tick, cnt_events, simulated_time ))
        
        # handle plotters
        for plotter in self.ctx.plotters:
            if plotter == 'flow_timeline.py':
                from plotter import flow_timeline
                flow_timeline.plot(self.ctx)  
            if plotter == 'overutil_single.py':
                from plotter import overutil_single
                overutil_single.plot(self.ctx) 
            if plotter == 'switch_table_counts.py':
                from plotter import switch_table_counts
                switch_table_counts.plot(self.ctx)       

        # add statistics    
        self.ctx.statistics['sim.final_tick'] = self.tick
        self.ctx.statistics['sim.processed_events'] = cnt_events
        self.ctx.statistics['sim.simulated_time'] = simulated_time

        # calculate default metrics for switches etc
        calculate_standard_metrics(self.ctx)

        # handle global on_simulation_finished callback registered in ctx
        if self.ctx.on_simulation_finished:
            self.ctx.on_simulation_finished(self.ctx) 

        # same as on_simulation_finished but for tests (with some additional output)
        if self.ctx.on_test_finished:
            errors = self.ctx.on_test_finished(self.ctx)
            if not isinstance(errors, list):
                raise RuntimeError('self.ctx.on_test_finished should return an array')
            if len(errors) > 0:
                logger.error("test %s finished with %d errors:" % (self.ctx.topo.filename, len(errors)))
                for error in errors:
                    logger.error("   ->%s" % error)
            else:
                logger.info("test %s passed with no errors" % self.ctx.topo.filename)
            return errors

