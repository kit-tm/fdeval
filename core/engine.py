import logging
import random
import time
import networkx as nx

from core.events import *

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, ctx, **kwargs):
        self.type = 'DefaultRoutingEngine'
        self.ctx = ctx

        self.processing_delay = kwargs.get("processing_delay", 0)
        self.active_flows = {} # map of currently active flows in the flow table
        self.active_flows_counter = {} # count active flows (the same flow can be processed by the same device multiple times in case of detour/loop)
        self.cnt_active_flows = 0 # counter 

        # to be used within super classes
        self.flow_stats = {} # the current stats of the flowtable
        self.port_stats = {} # the current port stats

    def add_delegation(self, tick, flow, ds, es, action=1):

        extension_path = nx.shortest_path(self.ctx.topo.graph, source=ds.id, target=es.id)

        switch_point = -1
        newPath = []
        done = False
        for i, node in enumerate(flow.path):
            if node == ds.id and not done:
                switch_point = i
                # if we arrive at switch, we add switch + detour
                for n in extension_path: newPath.append(n);
                for n in reversed(extension_path[:-1]): newPath.append(n);
                # it is important to only delegate the first matching id;
                # if the same id appears twice, it means that this flow was already
                # delegated (because there is a loop already); in this case, the second
                # or later occurence is the return path and should  NOT be delegated again
                done = True
            else:
                # all the other nodes are not changes
                newPath.append(node)


        
        flow.delegation.save_path = flow.path[:]
        flow.change_path(tick, newPath)
        flow.delegation.add_delegation(tick, ds, es, action=action)
        ds.cnt_adddelegations += 1

    def remove_delegation(self, tick, flow, ds, es, action=0):

        extensionPath = nx.shortest_path(self.ctx.topo.graph, source=ds.id, target=es.id)

        to_remove = extensionPath + list(reversed(extensionPath[:-1]))

        #print("current", flow.path)
        #print("remove", to_remove)

        new_path = None
        for i, node in enumerate(flow.path):
            try:
                for j, node2 in enumerate(to_remove):
                    if flow.path[i+j] != to_remove[j]: 
                        raise Error() # just trigger some exception
                # if we got here, we found the matching subarray
                new_path = flow.path[:i]+[ds.id]+flow.path[i+len(to_remove):]
                break;
            except:
                pass

        if not new_path:
            # This seems to be a multi-delegation case such as 
            #   original: 3, 2, 3
            #   later: 3, 2, 4, 2, 3 (there was another delegation from 2 to 4)
            # This should usually not happen; if it happens, it can be 
            # handled by removing the second delegation prior to the one that
            # should be removed right here; 
            found = False
            check = []
            for i, r in enumerate(flow.delegation.reports):
                if found:
                    if r.action == 1:
                        # this delegation was added afterwards; check whether is was
                        # not yet removed
                        check.append([r.source, r.target])
                    if r.action == 0:
                        if [r.source, r.target] in check:
                            check.remove([r.source, r.target])      
                if r.action == 1 and extensionPath == [r.source, r.target]:
                    # this is the delegation that should be removed right now
                    found = True

            # there was/is a second-level delegation
            if len(check) > 0:
                ds2, es2 = check[-1]
                ds2 = self.ctx.topo.get_switch_by_id(ds2)
                es2 = self.ctx.topo.get_switch_by_id(es2)
                # remove second-level delegation
                logger.warn('Second-level delegation occured, remove older delegation first; tick=%f flow=%s ds=%s es=%s' % (
                    tick, flow.label, ds2.label, es2.label))
                logger.warn('Second-level delegation occured (could be handled); tick=%f flow=%s ds=%s es=%s' % (
                    tick, flow.label, ds.label, es.label))
                logger.warn("   extensionPath: %s" % str(extensionPath))
                logger.warn("   flow.path: %s" % str(flow.path))
                logger.warn("   flow.reports:")
                for i, r in enumerate(flow.delegation.reports):
                    logger.warn("   %d - t=%d %d->%d a=%d   %s" % (i, r.tick, 
                        r.source, r.target, r.action, r.current_path))


                self.remove_delegation(tick, flow, ds2, es2, action=0)
                try:
                    # now try this call again (easiest solution to call it recursively)
                    self.remove_delegation(tick, flow, ds, es, action=action) 
                    try:
                        self.ctx.statistics['warning_second_level_delegation'] += 1
                    except KeyError:
                        self.ctx.statistics['warning_second_level_delegation'] = 1   
                    return # avoid runtime error, was handled
                except RuntimeError as e:
                    logger.error('Second-level delegation occured (could not be handled); tick=%f flow=%s ds=%s es=%s' % (
                        tick, flow.label, ds.label, es.label))
                    logger.error("   extensionPath: %s" % str(extensionPath))
                    logger.error("   flow.path: %s" % str(flow.path))
                    logger.error("   flow.reports:")
                    for i, r in enumerate(flow.delegation.reports):
                        logger.error("   %d - t=%d %d->%d a=%d   %s" % (i, r.tick, 
                            r.source, r.target, r.action, r.current_path)) 
                    raise e # now raise exception
            # will be avoided by return if possible to handle
            raise RuntimeError('detour undo failed, paths invalid')        
        
        #print("new_path", new_path)
        flow.change_path(tick, new_path)
        flow.delegation.remove_delegation(tick, ds, es, action=action)
        ds.cnt_backdelegations += 1

    def on_stats(self, switch, ev):
        pass

    def on_new_flow(self, switch, port, ev):
        pass

    def on_EVSwitchStats(self, switch, ev):
        
        # get stats
        finished_flows = []
        cnt_active_flows_total = 0 # total number of flows in table
        cnt_active_flows_evicted = 0 # number of flows that are evicted to some extension switch
        cnt_active_flows_stored = 0 # number of flows that other switches have evicted to this switch

        self.flow_stats = {}
        for id, flow in self.active_flows.items():
            cnt_active_flows_total += 1

            if flow.is_finished:
                finished_flows.append(flow)
                continue

            # calculate flow counters
            flow_duration = ev.tick - flow.start
            flow_demand = flow_duration * flow.demand_per_tick
            self.flow_stats[id] = (flow_duration, flow_demand,)

            if flow.delegation.is_delegated:
                evicted, stored = flow.delegation.get_status(switch)
                cnt_active_flows_evicted += evicted 
                cnt_active_flows_stored += stored
                # if stored is >1 more than one delegation switch has evicted
                # rules to this switch so we correct the total count accordingly
                if stored > 1:
                    cnt_active_flows_total += stored-1


        # remove finished flows
        for flow in finished_flows:
            self.active_flows_counter[flow.id] = 0
            del self.active_flows[flow.id]

        # port stats
        cnt_ports_delegated = 0
        cnt_switch_total_flows_arrived = 0
        cnt_port_flows_arrived_since_last_cycle = 0
        cnt_port_util_out = []
        cnt_port_util_in = []
        cnt_port_delegation_status = []
        self.port_stats = {}
        for id, port in switch.ports.items():
            if port.delegated: 
                cnt_ports_delegated += 1
                cnt_port_delegation_status.append(1)
            else:
                cnt_port_delegation_status.append(0)
            self.port_stats[id] = port.cnt_flows_arrived
            cnt_switch_total_flows_arrived += port.cnt_flows_arrived 
            cnt_port_flows_arrived_since_last_cycle = port.cnt_flows_arrived
            cnt_port_util_out.append(port.link_out.current_demand_sum)
            cnt_port_util_in.append(port.link_in.current_demand_sum)
            # reset per port flow arrival counter
            port.reset_arrival_counter()


        cnt_flows = cnt_active_flows_total - len(finished_flows) - cnt_active_flows_evicted + cnt_ports_delegated
        switch.flowtable.cnt_flows = cnt_flows

        if cnt_flows < 0:
            raise RuntimeError('cnt_flows<0')

        # Code snippets like the one below can be handy for debugging certain events inside a (seeded)
        # experiment; The example is very specific but it is left here for future reference
        if False:
            # example 1
            if switch.label == 's1':
                if ev.tick > 75 and ev.tick < 90:
                    print("!!!", ev.tick, cnt_active_flows_total, cnt_active_flows_evicted, cnt_active_flows_stored)
                    for id, flow in self.active_flows.items():
                        if flow.delegation.is_delegated:
                            es = flow.delegation.ds_to_es[switch.id] 
                            foundport = []
                            for pid, port in switch.ports.items():
                                for fid, flow2 in port.get_flows():
                                    if flow.id == fid:
                                        foundport.append(port.id)

                            print("    >> flow", flow.label, flow.start+flow.duration-ev.tick, es.label, foundport)

            # example 2
            if ev.tick > 130 and ev.tick < 150:
                if switch.label == 'DS':
                    data = {}
                    for id, port in switch.ports.items():
                        cnt = 0
                        total = 0
                        for _, flow in port.flows.items():
                            total += 1
                            if not flow.delegation.is_delegated: cnt+=1
                        data[port.id] = (cnt, total)
                    logger.info(str(ev.tick) + ' '  + ' '.join(['%d=%d|%d' % (x, v[0], v[1]) for x, v in data.items()]))
                
        report = EvSwitchReport(ev.tick);
        report.cnt_active_flows_total = cnt_active_flows_total
        report.cnt_active_flows_finished = len(finished_flows)
        report.cnt_active_flows = cnt_flows
        report.cnt_active_flows_stored = cnt_active_flows_stored
        report.cnt_active_flows_evicted = cnt_active_flows_evicted
        report.cnt_switch_total_flows_arrived = cnt_switch_total_flows_arrived
        report.cnt_port_util_out = cnt_port_util_out
        report.cnt_port_util_in = cnt_port_util_in
        report.cnt_port_delegation_status = cnt_port_delegation_status
        report.cnt_ports_delegated = cnt_ports_delegated
        report.cnt_port_flows_arrived_since_last_cycle = cnt_port_flows_arrived_since_last_cycle
        report.cnt_backdelegations = switch.cnt_backdelegations
        report.cnt_adddelegations = switch.cnt_adddelegations

        switch.push_report(report)
        switch.reset_counter()

        self.on_stats(switch, ev)

    def on_EVSwitchLastPacketOfFlowArrived(self, switch, ev):
        # unregister flow from port
        port = switch.ports.get(ev.link.id)
        port.unregister_flow(ev.flow)    
         
        self.active_flows_counter[ev.flow.id] -= 1
        if self.active_flows_counter[ev.flow.id] == 0:
            self.cnt_active_flows -= 1;
            try:
                del self.active_flows[ev.flow.id]
            except:
                raise RuntimeError("error del active_flows in on_EVSwitchLastPacketOfFlowArrived")

    def on_EVSwitchNewFlow(self, switch, ev):
        port = switch.ports.get(ev.link.id)
        port.register_flow(ev.flow)

        self.active_flows[ev.flow.id] = ev.flow;
        if not self.active_flows_counter.get(ev.flow.id):
            self.active_flows_counter[ev.flow.id] = 0
            self.cnt_active_flows += 1;
            
        self.active_flows_counter[ev.flow.id] += 1

        self.on_new_flow(switch, port, ev)
        ev.flow.forward_next(ev.tick, switch, self.processing_delay)
