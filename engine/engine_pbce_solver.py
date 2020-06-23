import logging
import random
import time
import networkx as nx
import sys
import numpy as np
import math

from heapq import heappush, heappop, nsmallest
from core.engine import Engine
from core.events import *

logger = logging.getLogger(__name__)

class PBCEEngineSolver(Engine):
    def __init__(self, ctx, **kwargs):
        super().__init__(ctx, **kwargs)
        self.type = 'PBCE Engine Version Solver'
        self.solution = None
        self.delegations = {}

    def on_new_flow(self, switch, port, ev):
        if port.delegated:
            if not ev.flow.delegation.get_extension_switch(switch):
                # remember flow-to-ingress-port mapping
                if not self.delegations.get(port.id):
                    self.delegations[port.id] = {}
                if not ev.flow in self.delegations[port.id]:
                    self.delegations[port.id][ev.flow.id] = ev.flow
                    #print("    register (%d): %s" % (port.id, ev.flow.label))
                self.add_delegation(ev.tick, ev.flow, switch, port.delegated)   

    # overwrite
    def on_stats(self, switch, ev):
        t = int(ev.tick)

        command = None
        
        if self.ctx.stored_solution:
            command = self.ctx.stored_solution.get(t)    

        if self.solution:
            command = self.solution.get(t)
              
        if command:
            # the remote switch was changed
            if command.get('change_es'):
                if len(command['change_es']) > 0:
                    for p, use_switch in command['change_es']:
                        for pid, port in switch.ports.items():
                            if port.id  == p: 
                                removed = []
                                old_switch = port.delegated.label
                                # remove delegation
                                if self.delegations.get(port.id):

                                    for _, flow in self.delegations.get(port.id).items():
                                        if flow.start+flow.duration > ev.tick:
                                            es = flow.delegation.get_extension_switch(switch)
                                            if es:   
                                                removed.append(flow)
                                                self.remove_delegation(ev.tick, flow, switch, es, action=2)

                                # update the delegation status of the port (new remote switch)
                                port.delegated = self.ctx.topo.get_switch_by_label(use_switch)

                                # add flows as delegated again
                                for flow in removed:
                                    # the different action values are important for metric calculation
                                    #print("    changed (%d):" % port.id, flow.label)
                                    self.add_delegation(ev.tick, flow, switch, port.delegated, action=3) 


                                print(ev.tick, "change [P=%d] %s->%s ==> %s->%s     removed=%d" % (port.id, 
                                    switch.label, old_switch, switch.label, use_switch, len(removed)))


            if len(command['remove']) > 0:
                for p, use_switch in command['remove']:
                    for pid, port in switch.ports.items():
                        if port.id  == p: 
                            #print("remove delegation", port.id)
                            port.delegated = None
                            
                            removed = []
                            # remove delegation
                            if self.delegations.get(port.id):
                                # run through all the delegated flow and revoke their
                                # delegation status if they are still active
                                for _, flow in self.delegations.get(port.id).items():
                                    if flow.start+flow.duration > ev.tick:
                                        es = flow.delegation.get_extension_switch(switch)
                                        if es:   
                                            removed.append(flow)
                                            self.remove_delegation(ev.tick, flow, switch, es)
                   
                            print(ev.tick, "remove [P=%d] %s->%s  removed=%d" % (port.id, switch.label, 
                                use_switch, len(removed)))
   
            if len(command['add']) > 0:
                for p, use_switch in command['add']:
                    for pid, port in switch.ports.items():
                        if port.id  == p: 
                            print(ev.tick, "deladd [P=%d] %s->%s" % (
                                port.id, switch.label, use_switch))
                            port.delegated = self.ctx.topo.get_switch_by_label(use_switch) 

