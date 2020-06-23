import math, sys

class EventPtr:
    def __init__(self, tick, ev_pointer):
        self.tick = tick
        self.ptr = ev_pointer

    def __lt__(self, other):
        return self.tick < other.tick

    def __gt__(self, other):
        return self.tick > other.tick

    def __ge__(self, other):
        return self.tick >= other.tick

class Event:
    """Event class that connects consumables and consumers
        id: a unique id number that is increased with every new event
        tick: the tick this event is executed (might be in the future)
        flow: contains a flow object associated with the event (not required for all events)
        pcc: path_change_counter, is changed if the path of a flow changes during simulation (see core/flow.py for details)
    """
    id = 0;

    def __init__(self, tick=sys.maxsize, flow=None, pcc=-1):
        self.id = Event.id
        self.tick = tick;
        self.flow = flow;
        self.pcc = pcc
        # invalid flag is set to True if the event becomes invalid/obsolete.
        # These events are ignored if is_finished is set and will be removed 
        # from the heap automatically
        self.invalid = False
        self.handler = None # the handler object; this event will be handled by handler.on_event()
        Event.id += 1

    def __lt__(self, other):
        return self.tick < other.tick

    def __gt__(self, other):
        return self.tick > other.tick

    def __ge__(self, other):
        return self.tick >= other.tick

# generic events
class EVNop(Event): pass;
class EVStats(Event): pass;
class EVLinkUpdateEvents(Event): # flow is finished

    def __init__(self, tick=sys.maxsize, flow=None, pcc=-1, demand_processed=0):
        super().__init__(tick, flow=flow, pcc=pcc);
        self.demand_processed = demand_processed # updated by link class

# link update events; used for flow tracing (debugging only)
# these events are emitted if the on_event() callback within a flow is defined
class EVLinkUpdateOnAdded(EVLinkUpdateEvents): pass; # flow was updated because another flow was added to the link
class EVLinkUpdateOnRemoved(EVLinkUpdateEvents): pass; # another link was removed
class EVLinkUpdateOnRemoveDelayed(EVLinkUpdateEvents): pass; #  the original remove event triggered but flow was not finished
class EVLinkUpdateBuffered(EVLinkUpdateEvents): pass; #  the original remove event triggered but flow was not finished
class EVLinkUpdateOnFinished(EVLinkUpdateEvents): pass
class EVLinkFlowStopped(EVLinkUpdateEvents): pass
class EVLinkPathChanged(EVLinkUpdateEvents): pass

class EVLinkFlowRemoved(Event): 
    """This event is triggered if a flow is removed from a link, i.e., the total
    demand was handled. Because links are a shared and limited resource, multiple
    flows compete against each other. In case there is not enough capacity for 
    a flow, it's 'share' is reduced which means that it takes longer to pass the link.
    In this case, the event has to be delayed until the demand was handled. This is
    handled by the check_finished().
    """

    def __init__(self, tick=sys.maxsize, flow=None, propagation_delay=0, 
        pcc=None, linkid=None, link=None, demand=0):
        super().__init__(tick, flow=flow, pcc=pcc);
        self.tick_old = flow.start
        self.linkid=linkid
        self.link = link
        self.propagation_delay = propagation_delay

        self.next_event = None
        self.current_datarate = None
        self.current_tick = None
        self.demand = demand # the demand that has to be processed for the given pcc
        self.processed_total = 0
        self.trigger_at = 0 # will be triggered next at this tick

    def check_finished(self, tick, pcc, new_dr=None, stopped=False):
        """Called to recalculate the remaining demand that has to 
        be processed by the flow based on the utilization of the link. 
        In case the duration of the flow has to be extended (not all
        the demand was processed), the function returns a positive float indicating
        the remaining demand. This function also triggeres a new 
        flow remove event that triggeres when the flow is expected to
        be finished based on the current datarate.

        return value > 0 means event is delayed
        return value == 0 means event is finished now
        """
        demand_processed = 0
        current_datarate = 0

        if self.current_datarate is None:
            raise RuntimeError('expected self.current_datarate != None')
        
        diff = tick-self.current_tick
        current_datarate = self.current_datarate
        demand_processed = diff*self.current_datarate

        # handle verbose flow logging
        if self.flow.on_event:
            self.flow.on_event(EVLinkUpdateOnRemoveDelayed(tick, flow=self.flow, pcc=self.pcc,
                demand_processed=demand_processed), link=self.link)

        self.processed_total += demand_processed

        remaining = self.demand-self.processed_total

        if new_dr != None:
            self.current_datarate = new_dr

        self.current_tick = tick
        self.next_event = remaining/self.current_datarate

        # calculate new end time
        if stopped == False:
            trigger_at = tick + remaining/self.current_datarate
            if not math.isclose(trigger_at, self.trigger_at):
                self.trigger_at = trigger_at
                #print("    .. reschedule", self.trigger_at)
                self.link.push_event(EventPtr(trigger_at, self))       

        if remaining > 0.001:
            return remaining
        else:
            return 0

    def init_utilization(self, tick, util):
        """Save utilization after the flow was first registered with the link"""
        #print("@%d update flowid=%d at tick=%d --> util is %f" % (tick, self.flow.id, tick, util))
        self.tick_old = tick;
        self.util_old = util;
        self.util_delayed = 0;
        self.demand_total = self.flow.get_total_demand() # won't be changed over time
        self.demand_remaining = self.flow.get_total_demand() # will be changed over time


class EvFlowReport(Event):

    def __init__(self, tick=sys.maxsize, flow=None, **kwargs):
        super().__init__(tick, flow, **kwargs);
        self.started = 0
        self.finished = 0 

class EvSwitchReport(Event):

    def __init__(self, tick=sys.maxsize, flow=None, **kwargs):
        super().__init__(tick, flow, **kwargs);
        self.cnt_active_flows = 0;
        self.cnt_active_flows_total = 0;
        self.cnt_active_flows_evicted = 0 
        self.cnt_active_flows_stored = 0
        self.cnt_evicted = 0
        self.cnt_removed_delegations = 0
        self.cnt_switch_total_flows_arrived = 0
        self.cnt_active_flows_finished = 0
        self.cnt_ports_delegated = 0
        self.cnt_backdelegations = 0
        self.cnt_adddelegations = 0
        self.cnt_port_flows_arrived_since_last_cycle = 0
        self.cnt_port_util_out = []
        self.cnt_port_util_in = []
        self.cnt_port_delegation_status = []

class EVLinkFlowAdd(Event):

    def __init__(self, tick=sys.maxsize, flow=None, processing_delay=0, demand_remaining=-1, pcc=-1):
        super().__init__(tick, flow=flow, pcc=pcc);
        self.processing_delay = processing_delay
        self.demand_remaining = demand_remaining

class EVLinkFlowAdded(Event):

    def __init__(self, tick=sys.maxsize, flow=None, processing_delay=0, demand_remaining=-1, pcc=-1):
        super().__init__(tick, flow=flow, pcc=pcc);
        self.tick_old = flow.start
        self.processing_delay = processing_delay
        self.demand_remaining = demand_remaining

class EVLinkFlowResume(Event):

    def __init__(self, tick=sys.maxsize, flow=None, pcc=-1):
        super().__init__(tick, flow=flow, pcc=pcc);
        self.tick_old = flow.start

class SwitchEvent(Event): pass;

class EVSwitchNewFlow(SwitchEvent):

    def __init__(self, tick=sys.maxsize, flow=None, pcc=-1, link=None):
        super().__init__(tick, flow=flow, pcc=pcc);
        self.link = link

class EVSwitchLastPacketOfFlowArrived(SwitchEvent): 

    def __init__(self, tick=sys.maxsize, flow=None, pcc=-1, link=None):
        super().__init__(tick, flow=flow, pcc=pcc);
        self.link = link


