import time
import networkx as nx
import math
import sys
import logging

from core.events import *
from core.reports import *

logger = logging.getLogger(__name__)

class FlowDelegation:

    def __init__(self, ctx, flow):   
        self.ctx = ctx
        self.flow = flow
        self.is_redirected = False; # set to true if this flow was redirected
        self.save_path = None
        self.delegated_counter = 0
        self.undelegated_counter = 0
        self.delegated_miss_counter = 0
        self.is_delegated = False
        self.active = {}
        self.ds_to_es = {} # support structure to identify the extension switch

        # stores events of type FlowDelegationReport that are created
        # by add_delegation() and remove_delegation()
        self.reports = []

    def get_status(self, switch):
        # returns (evicted, stored) where evicted is set to 1 if
        # switch is the delegation switch and stored is set to the number
        # of delegation switches that have an active delegation relationship
        # with switch (there can be multiple delegation relationships
        # for the same flow because several switches can delegate the flow 
        # to the same extension switch)
        active = self.active.get(switch.id)
        if active:
            # switch has the role of an extension switch for this flow;
            # we thus returns the number of delegated rules that are
            # stored on the switch
            return 0, len(active)
        else:
            # switch is not an extension switch but might be a delegation
            # switch
            for id, switch_list in self.active.items():
                if switch.id in switch_list:
                    return 1, 0
        # switch is not involed in the delegation relationships of this flow
        return 0, 0

    def get_extension_switch(self, ds):
        """Returns the extension switch if this flow is currently delegated;
        returns None of the flow is not delegated"""
        return self.ds_to_es.get(ds.id, None)

    def add_delegation(self, tick, ds, es, action=1):
        active = self.active.get(es.id)
        self.is_delegated = True
        if not active: self.active[es.id] = []
        # make sure we do not add delegations twice
        if ds.id in self.active[es.id]:
            raise RuntimeError('add_delegation error: the delegation relationship for '
                + 'flow=%d,ds=%s,es=%s does already exist!' % (self.flow.id, ds.label, es.label))
        self.active[es.id].append(ds.id)
        self.ds_to_es[ds.id] = es
        self.reports.append(FlowDelegationReport(tick=tick, source=ds.id, target=es.id, action=action, 
            current_path=self.flow.path[:]))

    def remove_delegation(self, tick, ds, es, action=0):
        if not ds.id in self.active.get(es.id):
            raise RuntimeError('remove_delegation error: the delegation relationship for '
                + 'flow=%d,ds=%s,es=%s does not exist!' % (self.flow.id, ds.label, es.label))      
        self.active.get(es.id).remove(ds.id)
        del self.ds_to_es[ds.id]
        if len(self.active.get(es.id)) == 0:
            self.is_delegated = False
        # add timestamp to the report
        self.reports.append(FlowDelegationReport(tick=tick, source=ds.id, target=es.id, action=action,
            current_path=self.flow.path[:]))
        #for report in self.reports:
        #    if report.target == es.id and report.source == ds.id:
        #        report.removed_at = tick;
        #        return
        #raise RuntimeError('remove_delegation could not find delegation report')

class Flow:

    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        # generic
        self.id = kwargs.get("id", ctx._counter_flows)
        self.label = kwargs.get("label", None) # custom name for the flow (if set)
        # routing
        self.source = kwargs.get("source"); # host object representing the source of this flow
        self.target = kwargs.get("target"); # host object representing the target of this flow
        # traffic generation
        self.flow_gen = kwargs.get("flow_gen"); # initial paramters the flow was created with
        self.start = kwargs.get("start", None) 
        self.duration = kwargs.get("duration", 1)
        self.demand_per_tick = kwargs.get("demand_per_tick", 1) # the average amount of bandwith required per tick    
        self.total_demand = self.duration * self.demand_per_tick
        self.duration_bck = self.duration # for debugging, not used in simulation (duration is changed over time)!
        # events
        self.on_event = kwargs.get("on_event", None); # used to realize callbacks that are called for every event this flow is processed in (used for debugging only!!)
        self.on_finished = kwargs.get("on_finished", None); # not used atm
        # stores delegation related infos
        self.delegation = FlowDelegation(ctx, self)
        self.finished_at = None # tick where this flow was stopped (debugging only)
        self.is_finished = False # set to true if first EVLinkUpdateOnFinished is triggered

        # total amount of processing delay experienced by this flow; this is added
        # every time on_EVSwitchNewFlow is executed (i.e. inside the engine)
        self.processing_delay_summed = 0 

        # path_change_counter (pcc) is increased every time the path was changed; this is
        # required because the same flow can be present on the same link multiple times
        # (e.g., if a delegation is added/removed). To make sure events such as EVLinkFlowStopped
        # and EVLinkFlowAdded can be interpreted correctly, it is necessary
        # to inject this information into the events. This is used in the flow_timeline
        # plotter to calculate the timeline correctly. The pcc mechanic is very similar
        # to a "path-based flowlet" mechanic, just think of all events with the same pcc
        # as one flowlet.   
        self.pcc = 0 
     
        # The effective_datarate is an path-global variable for the maximum 
        # datarate that can be applied based on the current bottleneck of the path of the flow.
        self.effective_datarate = None

        self.path_tuples = None
        self.path = None # path of this flow
        self.path_index = 0 # how "far" this flow has get inside path
        self.path_capacity = sys.maxsize # minimum capacity on path
        self.path_delay_summed = 0 # summed value of all propagation delay on original path (debugging only)
        self.calculate_path()
        for linkid in self.path_tuples:
            link = self.ctx.topo.graph.edges[linkid]['_link']
            self.path_delay_summed += link.propagation_delay
            if link.capacity < self.path_capacity:
                self.path_capacity = link.capacity
  
        ctx._counter_flows += 1

        # per pcc structures
        self.handler = []
        self.pcc_demand_remaining = {}
        self.pcc_stopped = {}



    def verbose(self, *msg):
        if self.ctx.verbose:
            print(*msg)

    def register_link(self, handler):
        self.handler.append(handler)

    def get_effective_datarate(self, tick, pcc, link):
        # get datarate for this flow on the specified link
        maxrate = min(self.demand_per_tick, self.path_capacity)
        myShare = 1.0 # share is only reduced if we experience overutilization
        if link.current_demand_sum > link.capacity:
            # we currently use a naive fairness model where each flow gets its perfect share
            # without any delay
            myShare = self.demand_per_tick/link.current_demand_sum
        datarate = maxrate*myShare

        if not self.effective_datarate:
            self.effective_datarate = datarate

        # Before we can calculate the demand that has been processed, we have to
        # take bottleneck situations into account. Assume the flow is associated
        # with two links L1 and L2. Assume its datarate is bottlenecked by L1 and we
        # are currently calculating the processed demand for L2. In this case, we 
        # are not only limited by the current utilization of L2 but also by the
        # bottleneck rate of L1. 
        bottleneck_rate = sys.maxsize
        for evFlowRemoved in self.handler:
            if evFlowRemoved.invalid: continue;
            #print("check", self.ctx.topo.print_link(evFlowRemoved.link.id))
            share = 1
            if evFlowRemoved.link.current_demand_sum > evFlowRemoved.link.capacity:
                share = evFlowRemoved.flow.demand_per_tick/evFlowRemoved.link.current_demand_sum
            max_dr = share * maxrate
            if max_dr < bottleneck_rate:
                bottleneck_rate = max_dr

        # if link is the new bottleneck
        if datarate < bottleneck_rate:
            bottleneck_rate = datarate

        if bottleneck_rate == sys.maxsize:
            bottleneck_rate = datarate

        if self.is_finished:
            self.verbose("this flow is already finished --> upper limit is", self.effective_datarate)
            if bottleneck_rate > self.effective_datarate:
                bottleneck_rate = self.effective_datarate
        
        # in case our datarate is different
        if not math.isclose(bottleneck_rate, self.effective_datarate):
            self.verbose("rate changed from", self.effective_datarate, bottleneck_rate)
            # recalculate for all links that are associated with this flow
            for i, evFlowRemoved in enumerate(self.handler):
                if evFlowRemoved.invalid: continue;
                self.verbose("___ RECALCULATE %d/%d for flow=%s link=%s" % (i+1, len(self.handler),
                    evFlowRemoved.flow.label, self.ctx.topo.print_link(evFlowRemoved.linkid)),
                    "effective_dr=%f trigger_at=%f" % (bottleneck_rate, evFlowRemoved.trigger_at))
                evFlowRemoved.check_finished(tick, pcc, bottleneck_rate)
                
        # update effective_datarate
        self.effective_datarate = bottleneck_rate

        self.verbose("___ effective rate for flow=%s link=%s" % (
                    self.label, self.ctx.topo.print_link(link.id)), bottleneck_rate)
        return bottleneck_rate

        # some old debugging info
        print("@", tick, "get_effective_datarate for flow=%d triggered by link=%s" % (self.id,
            self.ctx.topo.print_link(link.id)), datarate, pcc)
        print("maxrate", maxrate)
        print("fairshare", fairshare)
        print("stored effective_datarate", self.effective_datarate)
        print("bottleneck_rate", bottleneck_rate)
        print("datarate", datarate)
        print("fair_datadate", fair_datadate)
        print("")


    def detour_undo(self, tick, switch, extension_switch):
        extensionPath = nx.shortest_path(self.ctx.topo.graph, source=switch.id, target=extension_switch.id)

        to_remove = extensionPath + list(reversed(extensionPath[:-1]))

        print("current", self.path)
        print("remove", to_remove)

        new_path = None
        for i, node in enumerate(self.path):
            try:
                for j, node2 in enumerate(to_remove):
                    if self.path[i+j] != to_remove[j]: 
                        raise Error() # just trigger some exception
                # if we got here, we found the matching subarray
                new_path = self.path[:i]+[switch.id]+self.path[i+len(to_remove):]
                break;
            except:
                pass

        if not new_path:
            raise RuntimeError('detour undo failed, paths invalid')        
        print("new_path", new_path)
        self.change_path(tick, new_path)

    def change_path_detour(self, tick, switch, extension_switch):
        extensionPath = nx.shortest_path(self.ctx.topo.graph, source=switch.id, target=extension_switch.id)
        switch_point = -1
        newPath = []
        for i, node in enumerate(self.path):
            if node == switch.id:
                switch_point = i
                # if we arrive at switch, we add switch + detour
                for n in extensionPath: newPath.append(n);
                for n in reversed(extensionPath[:-1]): newPath.append(n);

            else:
                # all the other nodes are not changes
                newPath.append(node)
        self.change_path(tick, newPath)

    def change_path(self, tick, new_path):
        """called if path is changed"""
        new_tuples = list(zip(new_path[0:], new_path[1:]));

        # check that the new_path is actually different from the old path;
        # this should not happen if the upper layer programs work correctly
        # (not critical though, so we just throw a warning)
        if new_path == self.path:
            msg = "@%.1f change_path for flow=%s called with new_path == old_path; path=%s" % (
                tick, self.label, self.ctx.topo.print_link_tuples(new_tuples))
            print(msg)
            logger.warn(msg)
            return
        
        # get the first node that is different in the two paths; this is 
        # called the "switchover" point from where the new path emerges.
        # The path prior to this point is not changed!
        #print(".. new", self.ctx.topo.print_link_tuples(new_tuples))
        #print(".. old", self.ctx.topo.print_link_tuples(self.path_tuples))
        switchover = 0
        old = None
        updated = None
        seccnt = 0
        while switchover < len(new_tuples) and switchover < len(self.path_tuples):
            if new_tuples[switchover] != self.path_tuples[switchover]: 
                updated = self.ctx.topo.print_link(new_tuples[switchover])
                old = self.ctx.topo.print_link(self.path_tuples[switchover])
                break
            seccnt += 1
            if seccnt > 1000:
                raise RuntimeError('change_path whileloop forever: newpath=%s' % str(new_path))
            switchover += 1
        self.verbose("@", tick, "change_path -> calculate paths:", self.path, "->", new_path,
            "switchover=", self.ctx.topo.print_link(self.path_tuples[switchover]))

        # check1
        links_infront_switchover = self.path_tuples[:switchover+1]
        for link_id in links_infront_switchover:
            
            link_obj = self.ctx.topo.graph.edges[link_id]['_link']
            add_event = link_obj.pre_registered_flows.get(self.pccid())
            if add_event:
                add_event.invalid = True
                self.verbose("@", tick, "change_path -> invalidate pre_registration", self.ctx.topo.print_link(link_obj.id))

        if self.path_index < switchover:
            # change the path
            pccid = '%d.%d.%d' % (link_id[0], link_id[1], self.pcc)
            self.pcc_demand_remaining[self.pcc] = self.total_demand
            self.path = new_path;
            self.path_tuples = new_tuples;
            self.pcc += 1
            self.verbose("@", tick, "change_path -> path_index < switchover; skip")
            #new_link = new_tuples[switchover]
            #new_link_obj = self.ctx.topo.graph.edges[new_link]['_link']
            #ev = EVLinkFlowAdd(tick, flow=self, processing_delay=0, pcc=self.pcc)
            #new_link_obj.push_event(ev)
            #self.verbose("@", tick, "change_path -> invalidate pre_registration", self.ctx.topo.print_link(link_obj.id))
            #self.verbose("@", tick, "change_path -> switchto", self.ctx.topo.print_link(new_link))
            #if self.on_event:
            #    self.on_event(EVLinkPathChanged(tick, flow=self, pcc=self.pcc-1),
            #        link=new_link_obj)
            return

        # step 2
        demand_remaining = self.total_demand
        link_off_id = self.path_tuples[switchover]
        link_obj = self.ctx.topo.graph.edges[link_off_id]['_link']
        for pcc in range(self.pcc, -1, -1):
            pccid = '%d.%d.%d' % (link_off_id[0], link_off_id[1], pcc)
            #print("check",self.ctx.topo.print_link(link_id), pcc)
            if link_obj.flows.get('%d.%d' % (self.id, pcc)):
                if self.pcc_stopped.get(pccid): continue
                self.verbose("@", tick, "change_path -> call immediate stop on", 
                    self.ctx.topo.print_link(link_off_id))
                ev = EVLinkFlowStopped(tick=tick, flow=self, 
                    pcc=pcc)
                demand_remaining = link_obj.on_EVLinkFlowStopped(ev)
                self.pcc_stopped[pccid] = demand_remaining
                #self.pcc_demand_remaining[self.pcc] = demand_remaining
                #self.pcc_demand_remaining[pcc] = demand_remaining
                #break;

        # step 3 stop all the links currently running
        propagation_delay = 0
        links_after_switchover = self.path_tuples[switchover+1:]
        for link_id in links_after_switchover:
            link_obj = self.ctx.topo.graph.edges[link_id]['_link']  
            propagation_delay += link_obj.propagation_delay  
            for pcc in range(self.pcc, -1, -1):
                pccid = '%d.%d.%d' % (link_id[0], link_id[1], pcc)
                #print("check", self.ctx.topo.print_link(link_id), pcc, pccid)
                if link_obj.flows.get('%d.%d' % (self.id, pcc)):
                    if self.pcc_stopped.get(pccid): continue
                    ev = EVLinkFlowStopped(tick=tick+propagation_delay, 
                        flow=self, pcc=pcc)
                    self.verbose("@", tick, "change_path -> propagate stop to", 
                        self.ctx.topo.print_link(link_id), "with pcc", pcc, "delay", tick+propagation_delay) 
                    link_obj.push_event(ev)
                    self.pcc_stopped[pccid] = demand_remaining
                    #break
                                    
        self.pcc_demand_remaining[self.pcc] = demand_remaining
        self.path_index = switchover; # "restart" this flow from the point where the detour started
        #self.duration = remaining_time;
        self.path = new_path;
        self.path_tuples = new_tuples;
        self.pcc += 1

        link = new_tuples[switchover]
        self.verbose("@", tick, "change_path (%s) -> push add flow to" % str(self.path),   
            self.ctx.topo.print_link(link), "pcc=", self.pcc, "demand=", demand_remaining)
        link_obj = self.ctx.topo.graph.edges[link]['_link']
        ev = EVLinkFlowAdded(tick, flow=self, pcc=self.pcc, demand_remaining=demand_remaining)
        link_obj.on_EVLinkFlowAdded(ev) # can be triggered directly

        return
  
        # DEPRECATED FROM HERE ON

        # at the switchover point (i.e., the switch where the output interface of the
        # flow has to be changed), we have to calculate the amount of demand that 
        # was processed so far. This is required so that we can change the 
        # flow configuration later on (because parts of the demand were already
        # handled). This is what "pcc_demand_remaining" is used for.
        pccid = self.pccid() 
        link_off_id = self.path_tuples[switchover]
        link_obj = self.ctx.topo.graph.edges[link_off_id]['_link']
        if not link_obj.flows.get(pccid):


            # case 1: even if the flow is not registered with its current pcc value,
            # it might be the case that the pcc was changed in between (e.g.,
            # because of another call to change_path later on time and 
            # later in the path). If this is the case, there should be some "remainings"
            # that we have to stop.
            stopped = 0
            propagation_delay = link_obj.propagation_delay
            for link_id in self.path_tuples[switchover+1:]:     
                new_link_obj = self.ctx.topo.graph.edges[link_id]['_link']  
                if new_link_obj.flows.get('%d.%d' % (self.id, self.pcc)):
                    print("@", tick, "change_path -> propagate stop to", 
                        self.ctx.topo.print_link(link_id), tick+propagation_delay)
                    stopped += 1
                    ev = EVLinkFlowStopped(tick=tick+propagation_delay, 
                        flow=self, pcc=self.pcc)
                    new_link_obj.push_event(ev)
                    propagation_delay += new_link_obj.propagation_delay
            if stopped > 0:
                # we now have to find the "older" pcc so that we can stop it
                # and start the new flow
                old = link_obj.flows.get('%d.%d' % (self.id, self.pcc-1))
                if old:
                    self.verbose("@", tick, "change_path -> call immediate stop on", 
                        self.ctx.topo.print_link(link_off_id))
                    ev = EVLinkFlowStopped(tick=tick, flow=self, 
                        pcc=self.pcc-1)
                    demand_remaining = link_obj.on_EVLinkFlowStopped(ev)
                    
                    self.pcc_demand_remaining[self.pcc] = self.total_demand
                    #self.path_index = switchover-1
                    self.path = new_path;
                    self.path_tuples = new_tuples;
                    self.pcc += 1

                    link = new_tuples[switchover]
                    print("@", tick, "change_path -> push add flow to", 
                        self.ctx.topo.print_link(link), "pcc=", self.pcc)
                    link_obj = self.ctx.topo.graph.edges[link]['_link']
                    ev = EVLinkFlowAdded(tick, flow=self, pcc=self.pcc)
                    link_obj.on_EVLinkFlowAdded(ev) # can be triggered directly
                    return;

            # case 2: the flow is not registered as an active flow with the link which means that
            # the path was changed before the flow reached the switchover point;
            # If this is the case, there has to be an entry in one of the pre_registered_flows
            # arrays alongside the old path.             
            self.verbose("@", tick, "change_path -> check pre-registration", self.path, "-->", new_path)
            for link_id in  self.path_tuples:
                link_obj = self.ctx.topo.graph.edges[link_id]['_link']
                add_event = link_obj.pre_registered_flows.get(pccid)
                if add_event:
                    add_event.invalid = True

                    # change the path
                    self.pcc_demand_remaining[self.pcc] = self.total_demand
                    self.path = new_path;
                    self.path_tuples = new_tuples;
                    self.pcc += 1

                    new_link = new_tuples[switchover]
                    ev = EVLinkFlowAdded(add_event.tick, flow=self, 
                        processing_delay=add_event.processing_delay, pcc=self.pcc)
                    new_link_obj = self.ctx.topo.graph.edges[new_link]['_link']
                    new_link_obj.pre_registered_flows[self.pccid()] = ev
                    new_link_obj.push_event(ev)

                    self.verbose("@", tick, "change_path -> invalidate pre_registration", self.ctx.topo.print_link(link_obj.id), add_event.tick)
                    self.verbose("@", tick, "change_path -> switchto", self.ctx.topo.print_link(new_link))


                    if self.on_event:
                        self.on_event(EVLinkPathChanged(tick, flow=self, pcc=self.pcc-1),
                            link=new_link_obj)
                    return

            # no pre_registration found (this happens frequently if the path is changed reactively)
            # in this, we can simply change the flow parameters and increase the pcc; Important: 
            # we DO NOT have to add the new flow manually, because the regular add flow function
            # will be called later (the flow has not yet reached the switch point)
            self.pcc_demand_remaining[self.pcc] = self.total_demand
            #self.path_index = switchover-1
            self.path = new_path;
            self.path_tuples = new_tuples;
            self.pcc += 1
            # we have to return here!
            return
           
        else:

            self.verbose("@", tick, "change_path -> call immediate stop on", 
                self.ctx.topo.print_link(link_off_id))
            ev = EVLinkFlowStopped(tick=tick, flow=self, 
                pcc=self.pcc)
            demand_remaining = link_obj.on_EVLinkFlowStopped(ev)

            #print(".. remain", demand_remaining)
            next_links = self.path_tuples[switchover+1:]
            if len(next_links) > 0:
                propagation_delay = link_obj.propagation_delay
                for link_id in next_links:
                    print("@", tick, "change_path -> propagate stop to", 
                        self.ctx.topo.print_link(link_id), tick+propagation_delay)

                    ev = EVLinkFlowStopped(tick=tick+propagation_delay, 
                        flow=self, pcc=self.pcc)
                    link_obj = self.ctx.topo.graph.edges[link_id]['_link']
                    link_obj.push_event(ev)
                    propagation_delay += link_obj.propagation_delay

        self.path_index = switchover; # "restart" this flow from the point where the detour started
        #self.duration = remaining_time;
        self.path = new_path;
        self.path_tuples = new_tuples;
        self.pcc += 1

        link = new_tuples[switchover]
        print("@", tick, "change_path -> push add flow to", 
            self.ctx.topo.print_link(link), "pcc=", self.pcc)
        link_obj = self.ctx.topo.graph.edges[link]['_link']
        ev = EVLinkFlowAdded(tick, flow=self, pcc=self.pcc)
        link_obj.on_EVLinkFlowAdded(ev) # can be triggered directly
        pass

    def pccid(self):
        return '%d.%d' % (self.id, self.pcc)

    def get_total_demand(self):
        return self.duration * self.demand_per_tick;

    def calculate_path(self):
        """Calculate shortest path based on flow parameters"""

        # if the path was defined a-priori as an array of labels
        if self.flow_gen.get("fg_fixed_path"):
            path = self.flow_gen.get("fg_fixed_path")
            self.path = [self.ctx.topo.get_node_by_attr('label', x) for x in path]
            self.path_tuples = list(zip(self.path[0:], self.path[1:]))
            #print (self.path, self.path_tuples)
            # make sure all tuples defined by fg_fixed_path actually exist in the graph 
            for t in self.path_tuples:
                try:
                    self.ctx.topo.graph.edges[t]
                except KeyError:
                    raise RuntimeError('fg_fixed_path error: %s does not exist as an edge. Valid edges are %s' % (t, str(self.ctx.topo.graph.edges())))
            return

        # use shortest path based on target
        if self.target:
            self.path = nx.shortest_path(self.ctx.topo.graph, source=self.source.id, target=self.target.id)
            self.path_tuples = list(zip(self.path[0:], self.path[1:]))
            return;

        print("Error: calculate_path()")
        print("  fg_fixed_path?", self.flow_gen.get("fg_fixed_path"))
        print("  self.target =", self.target)
        raise RuntimeError("error executing shortest_path(); This function " +
            "is used to calculate the hops for a flow object; if this error occurs, " + 
            "this path was not calculated. Check target, fg_fixed_path, fg_fixed_destination etc")

    def forward_next(self, tick, switch, processing_delay):
        """triggered by on_EVSwitchNewFlow inside engine"""
        link = self.next_link(None)
        if link:
            self.processing_delay_summed += processing_delay
            ev = EVLinkFlowAdd(tick+processing_delay, flow=self, 
                processing_delay=processing_delay, pcc=self.pcc)
            link.push_event(ev)

    def next_link(self, switch):
        """Returns the next link object on the path"""
        if self.path_index < len(self.path_tuples)-1:
            self.path_index += 1
            linkid = self.path_tuples[self.path_index]
            #print("next_link", self.path_tuples, self.path_index, linkid)
            return self.ctx.topo.graph.edges[linkid]['_link']
