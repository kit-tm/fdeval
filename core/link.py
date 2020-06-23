import logging

from core.consumable import Consumable
from core.events import *
from core.engine import Engine

logger = logging.getLogger(__name__)

class Link(Consumable):

    counter = 0;

    def __init__(self, ctx, **kwargs):
        super().__init__(ctx, **kwargs);
        self.id = kwargs.get('id', Link.counter) # networkx link identifier
        self.propagation_delay = kwargs.get("propagation_delay", 0.01); # propagation delay in ticks
        self.observe_utilization = kwargs.get("observe_utilization", False) # store utilization metrics if set to true
        self.active = {} # map of currently active flows on the link
        self.capacity = kwargs.get("capacity", 100000000000) # capacity in demands/tick
        self.cnt_active = 0 # number of elements in self.active
        self.cnt_terminated = 0 # number of flows that terminated at this link

        self.label = ctx.topo.print_link(self.id)
        self.flows = {}

        # Holds the sum of all demands that are registered with this link. If this
        # value exceeds the capacity of the link, the link is overutilized.
        self.current_demand_sum = 0;

        # Flows are added to a link via the on_EVLinkFlowAdd() method. This marks the 
        # event in time when the first packet has entered the link. Because the model
        # keeps track of the received/processed demand only, the "real" 
        # on_EVLinkFlowAdded event is delayed by the propagation delay of the link. However,
        # it is important that we have access to the events that are added via 
        # the on_EVLinkFlowAdd() method, because the path of the flow may change 
        # BEFORE the on_EVLinkFlowAdded() method was called. See the change_path()
        # method in core/flow.py to see how pre_registered_flows is used to deal with this
        # issue.
        self.pre_registered_flows = {}

        # max_pcc stores the current maximum path_change_counter for every flow. This 
        # value is important because delays (like propagation delay) can cause events
        # to arrive later than expected.
        self.max_pcc = {}

        Link.counter += 1

    def get_current_utilization(self):
        return self.current_demand_sum/self.capacity;

    def get_next_node(self):
        # trigger last packet arrived at switch
        return self.ctx.topo.graph.nodes[self.id[1]]

    def on_EVLinkFlowRemoved(self, ev, tick):
        """Handle EVLinkFlowRemoved; the tick is passed as a parameter because of EventPtr"""
        self.verbose("@ ----------")
        self.verbose("@", tick, "REMOVE_FLOW", ev.flow.label, self.ctx.topo.print_link(self.id))
        self.verbose("@ ----------")
        demand_remaining = ev.check_finished(tick, ev.pcc)
        if demand_remaining > 0:
            # if demand_remaining returns a value > 0 the check_finished function
            # has already rescheduled the flow and there is nothing else to do here
            return 
        else:

            # handle verbose flow logging
            if ev.flow.on_event:
                ev.flow.on_event(EVLinkUpdateOnFinished(tick, flow=ev.flow,
                    pcc=ev.pcc,
                    demand_processed=0), link=self) 
            
            # In case the flow is really finished, it is removed from the list of
            # active flows and the utilization of the link (the currently utilized demand)
            # is reduced accordingly. This also requires an update of all remaining active 
            # flows (utilization is reduced) 
            try:
                del self.active[ev.flow.id]; # remove active flow from list
                ev.invalid = True # important to avoid clashes with future pointers!
            except:
                logger.debug("WARNING: tried to remove flow=%s from link=%s that was never registered; tick=%f; ev.pcc=%d, flow.pcc=%d, flowFinished=%s" % 
                    (ev.flow.label, self.ctx.topo.print_link(self.id), ev.tick, 
                    ev.pcc, ev.flow.pcc, str(ev.flow.is_finished)))
                return
                # should not happen but not really a problem in most cases
                #raise RuntimeError()

            ev.flow.is_finished = True # mark this flow as finished
            ev.flow.finished_at = tick

            # handle verbose flow logging
            if ev.flow.on_event:
                ev.flow.on_event(EVLinkUpdateOnFinished(tick, flow=ev.flow,
                    pcc=ev.pcc,
                    demand_processed=0), link=self) 

            # trigger flow removal on switch
            switch = self.get_next_node()
            if (switch.get("_switch")):
                switch['_switch'].push_event(
                    EVSwitchLastPacketOfFlowArrived(tick, flow=ev.flow, link=self))



            self.cnt_active -= 1; # decrement active counter
            self.current_demand_sum -= ev.flow.demand_per_tick

            # available bandwith of all active flows increases; this is only relevant if
            # the "old" utilization (before the flow was removed) was above 100%; If the old
            # utilization was below 100%, all flow demands are satisfied regardless
            if self.current_demand_sum+ev.flow.demand_per_tick > self.capacity:
                # if we come from a bottlenecked situation, all other flows on the link
                # have to be updated
                for _, item in self.active.items():
                    flow, flowRemoveEvent = item
                    self.verbose("")
                    self.verbose("         REMOVED FOLLOWUP --> flow=%d" % flow.id, self.ctx.topo.print_link(self.id))
                    flow.get_effective_datarate(tick, ev.pcc, self)


    def on_EVLinkFlowStopped(self, ev):
        """
        for pcc in range(ev.pcc, -1, -1):
            pccid = '%d.%d.%d' % (self.id[0], self.id[1], pcc)
            if self.flows.get(pccid):
                flow, flow_removed = self.flows[pccid]
                if not flow_removed.invalid:
                    self.verbose("@", tick, "should kill", 
                    self.ctx.topo.print_link(self.id))
        """
        max_pcc = self.max_pcc.get(ev.flow.id)
        pccid = '%d.%d' % (ev.flow.id, ev.pcc)
        try:
            flow, flow_removed = self.flows[pccid]
        except Exception as e:
            msg = '@%.0f on_EVLinkFlowStopped %s could not find flow=%s at pcc=%d' % (
                ev.tick, self.ctx.topo.print_link(self.id), ev.flow.label, ev.pcc)
            print(msg)
            assert(ev.pcc < max_pcc)
            #logger.warn(msg)
            if ev.pcc > 0:
                d2 = ev.flow.pcc_demand_remaining[ev.pcc]
                d1 = ev.flow.pcc_demand_remaining[ev.pcc-1]
                missing_demand = d1-d2
                #ev.flow.buffered += missing_demand
                # get the current flow_removed
                #flow_max, flow_removed_max = self.flows['%d.%d' % (ev.flow.id, self.max_pcc[ev.flow.id])]
                #flow_removed_max.buffered += missing_demand
                #self.push_event(EventPtr(ev.tick, flow_removed_max)) 
                print("missing demand", max_pcc, missing_demand)   
                print(self.flows)   
            return None

        if max_pcc is not None:
            if ev.pcc < max_pcc:
                pass
                # the flow removed entry was found but this is
                # a stopped event with a smaller pcc, i.e., the
                # same flow has already been rescheduled to this
                # link (which is common in delegation scenarios)
                #print("@", ev.tick, "stopped pcc=%d (not max) on link=%s" % (
                #    ev.pcc, self.ctx.topo.print_link(self.id)))
                #flow_removed.invalid = True
                #ev.flow.pcc_demand_remaining[ev.pcc] = demand_remaining
                #return
            #raise(e)

        flow, flow_removed = self.flows[pccid]
        demand_remaining = flow_removed.check_finished(ev.tick, ev.pcc, stopped=True)
        self.verbose("@", ev.tick, "on_EVLinkFlowStopped", self.ctx.topo.print_link(self.id), 'flow=%d' % ev.flow.id, "remaining=", demand_remaining)
        remaining_time = flow_removed.next_event
        flow_removed.invalid = True # disable the currently active flow_removed event!

        # trigger flow removal on switch
        switch = self.get_next_node()
        if (switch.get("_switch")):
            switch['_switch'].push_event(
                EVSwitchLastPacketOfFlowArrived(ev.tick, flow=flow, link=self))

        # remember how much demand was handled in this pcc
        flow.pcc_demand_remaining[ev.pcc] = demand_remaining

        update_events = 0
        self.cnt_active -= 1; # decrement active counter
        old_demand_sum = self.current_demand_sum
        self.current_demand_sum -= ev.flow.demand_per_tick
        if old_demand_sum > self.capacity:
            for _, item in self.active.items():
                flow, flowRemoveEvent = item
                if flow != ev.flow:
                    update_events+=1
                    flow.get_effective_datarate(ev.tick, ev.pcc, self)

        # it is ok for this variable to be >>1000 but only in heavy overutilization
        # scenarios. This can slow down the simulation drastically, so we print at least
        # some kind of warning in case this is not the intendend behavior.
        if update_events > 1000:
            logger.warn('update_events in link.py->on_EVLinkFlowStopped >1000; is this an overload scenario?')
        # update demand
        ev.demand_processed = 0

        # handle verbose flow logging
        if ev.flow.on_event:
            ev.flow.on_event(ev, link=self)

        """
        # remove flow from flows
        try:
            del self.flows[pccid];
            print("@", ev.tick, "removed flow with pcc=%d from self.flows on %s" % (ev.pcc, self.ctx.topo.print_link(self.id)))
        except:
            msg = "ERROR: del self.flows[%s] failed in on_EVLinkFlowStopped" % ev.flow.pccid()
            logger.error(msg)
            raise RuntimeError(msg)
        """
        return demand_remaining

    def on_EVLinkFlowAdd(self, ev):
        if ev.flow.on_event:
            ev.flow.on_event(ev, link=self)  

        ev = EVLinkFlowAdded(ev.tick+self.propagation_delay, flow=ev.flow, 
            processing_delay=ev.processing_delay, pcc=ev.pcc)

        max_pcc = self.max_pcc.get(ev.flow.id)
        if max_pcc is None: 
            self.max_pcc[ev.flow.id] = ev.pcc
        else:
            if ev.pcc > max_pcc:
                self.max_pcc[ev.flow.id] = ev.pcc

        self.pre_registered_flows[ev.flow.pccid()] = ev

        self.push_event(ev)

    def on_EVLinkFlowAdded(self, ev):

        # remove entry from pre_registered_flows if necessary
        pccid = ev.flow.pccid()
        if self.pre_registered_flows.get(pccid):
            del self.pre_registered_flows[pccid]

        # check whether the pcc value of ev is valid
        max_pcc = self.max_pcc.get(ev.flow.id)
        if max_pcc is not None:
            if ev.pcc < max_pcc:
                remaining = ev.flow.pcc_demand_remaining[ev.pcc]
                if ev.pcc > 0:
                    last = ev.flow.pcc_demand_remaining[ev.pcc-1]   
                #print("@", ev.tick, self.ctx.topo.print_link(self.id), "ignore this on_EVLinkFlowAdded", last-remaining)
                #ev.flow.get_effective_datarate(ev.tick, max_pcc, self)
                #return

        self.current_demand_sum += ev.flow.demand_per_tick
        util = self.current_demand_sum/self.capacity;

        self.verbose("@ ----------")
        self.verbose("@", ev.tick, 'ADDFLOW flow=%d, link=%s with pcc=%d, util=%f' % (ev.flow.id, 
                self.ctx.topo.print_link(self.id), ev.pcc, util))
        self.verbose("@ ----------")   
        datarate = ev.flow.get_effective_datarate(ev.tick, ev.pcc, self)

        # update demand/capacity
        
        # calculate the demand
        remaining = ev.flow.total_demand
        if ev.demand_remaining >-1:
            remaining = ev.demand_remaining
        else:
            if ev.pcc > 0:
                #pccid = '%d.%d.%d' % (self.id[0], self.id[1], ev.pcc-1)
                #logger.info(ev.flow.pcc_stopped)
                #remaining = ev.flow.pcc_stopped[pccid]
                remaining = ev.flow.pcc_demand_remaining[ev.pcc-1]
                # update the maximum pcc value for this flow on this link
                # so that "older" on_EVLinkFlowAdded with lower pcc values
                # that might still be scheduled somewhere can be ignored
            
        #max_pcc = self.max_pcc.get(ev.flow.id)
        if max_pcc is None: 
            self.max_pcc[ev.flow.id] = ev.pcc
        else:
            if ev.pcc > max_pcc:
                self.max_pcc[ev.flow.id] = ev.pcc
                #print("update pcc @added", self.max_pcc[ev.flow.id])


        duration = remaining/datarate

        if util > 1:
            self.verbose("")
            self.verbose("    FOLLOWUP ADDFLOW (util > 0)")
            self.verbose("")
            for _, item in self.active.items():
                flow, flowRemoveEvent = item
                flow.get_effective_datarate(ev.tick, ev.pcc, self)

        # trigger flow arrival on switch after propagation delay;
        # The get_switch(link) function will return the next switch on
        # the path of this flow or None if the flow terminates 
        switch = self.get_next_node()
        #switch = ev.flow.path.next_node(self)
        if (switch.get("_switch")):
            #switch['_switch'].engine.on_EVLinkFlowAdded(self, ev);
            switch['_switch'].push_event(
                EVSwitchNewFlow(ev.tick, flow=ev.flow, link=self))

        # trigger the flow remove event after all packets have passed
        util = self.current_demand_sum/self.capacity;
        removeFlow = EVLinkFlowRemoved(
            ev.tick+duration, flow=ev.flow, 
            propagation_delay=self.propagation_delay, linkid=self.id, link=self,
            pcc=ev.pcc, demand=remaining)

        removeFlow.trigger_at = ev.tick + duration
        removeFlow.current_datarate = datarate
        removeFlow.current_tick = ev.tick
        removeFlow.init_utilization(ev.tick, util)
        self.push_event(removeFlow); 

        #print("@", ev.tick, self.ctx.topo.print_link(self.id), "STORE", '%d.%d' % (ev.flow.id, ev.pcc))
        # be careful to use the pcc from the event and NOT the pcc from the flow
        # the pcc in the flow might have been updated in the meantime!
        self.flows['%d.%d' % (ev.flow.id, ev.pcc)] = [ev.flow, removeFlow]
        self.active[ev.flow.id] = [ev.flow, removeFlow]; # save active flow
        ev.flow.register_link(removeFlow)

        if ev.flow.on_event:
            ev.flow.on_event(ev, link=self)  

    def on_EVStats(self, event):
        if self.cnt_active > 0:
            print("link %d" % self.id, ev.tick, self.cnt_active, self.current_demand_sum/self.capacity)

    def on_event(self, ev):
        # EventPtr are events that will finish sooner as originally expected, e.g., because
        # other flows were removed. Because it is expensive to remove items from the heap,
        # the item is flagged as invalid and 'replaced' with a EventPtr item. This item contains
        # a new (lower) tick and a pointer to the original event.
        if isinstance(ev, EventPtr):
            if ev.ptr.invalid: return;
            #print("handle ptr", ev, ev.tick)
             # handle verbose logging  for selected flows
            #if ev.ptr.flow:
            #    if ev.ptr.flow.on_event:
            #        ev.ptr.flow.on_event(ev, link=self)
            return self.on_EVLinkFlowRemoved(ev.ptr, ev.tick)

        if ev.invalid: return; # remove events from the heap that are flagged as invalid

        # periodic counter for statistics
        if isinstance(ev, EVStats):
            return self.on_EVStats(ev);
              
        # first packet of flow put onto a link -> trigger EVLinkFlowAdded after prop. delay
        if isinstance(ev, EVLinkFlowAdd):
            return self.on_EVLinkFlowAdd(ev); 

        # a flow was stopped, e.g., because path changed (not finished)
        if isinstance(ev, EVLinkFlowStopped):
            return self.on_EVLinkFlowStopped(ev);

        # a new flow arrives at the link
        if isinstance(ev, EVLinkFlowAdded):
            return self.on_EVLinkFlowAdded(ev);

        # a flow is finished
        if isinstance(ev, EVLinkFlowRemoved):
            return self.on_EVLinkFlowRemoved(ev, ev.tick)
