from core.consumable import Consumable
from core.events import *
from core.engine import Engine

class Port:

    def __init__(self, id):
        self.id = id 
        self.target = None
        self.link_out = None
        self.link_in = None
        self.delegated = None

        # This variable is used to calculate the flows that have 
        # arrived on this port during two decision cycles. The variable
        # is managed by the engine and reset there.
        self.cnt_flows_arrived = 0
        self.cnt_flows_removed = 0
        self.flows = {} # map of registered flows
        self.arrival_data = []
        self.arrival_rate = 0
        self.arrival_last = 0
        self.arrival_remove = 0

    def register_flow(self, flow):
        if not self.flows.get(flow.id):
            self.cnt_flows_arrived += 1
            self.flows[flow.id] = flow

    def unregister_flow(self, flow):
        if self.flows.get(flow.id):
            del self.flows[flow.id]
            self.cnt_flows_removed +=1

    def get_flows(self):
        """Returns a list of flows that entered the switch via this port"""
        return filter(lambda flow: flow[1].is_finished == False, self.flows.items())
    
    def reset_arrival_counter(self, history=10):
        self.arrival_data.append(self.cnt_flows_arrived)
        self.arrival_last = self.cnt_flows_arrived
        self.cnt_flows_arrived = 0
        self.arrival_remove = self.cnt_flows_removed
        self.cnt_flows_removed = 0

        lastn = [i for i in self.arrival_data[-history:]]
        #diffs = [v2-v1 for v1, v2 in list(zip(lastn[0:], lastn[1:]))]
        self.arrival_rate = 0
        if len(lastn) > 0:
            self.arrival_rate = sum(lastn)/float(len(lastn))

        # avoid flow_arrival_per_port getting more than usen entries
        self.arrival_data = self.arrival_data[-history:]

class FlowTable:

    def __init__(self, switch):
        self.switch = switch
        self.cnt_flows = 0

class Switch(Consumable):

    def __init__(self, ctx, **kwargs):
        super().__init__(ctx, **kwargs);
        self.id = kwargs.get("id") # networkx node id
        self.label = kwargs.get("label", "NoLabelSet"); # name in topology
        self.x = kwargs.get("x"); # coordinates in topology
        self.y = kwargs.get("y"); # coordinates in topology

        # create a port object for each port of the switch; these are used 
        # to store and access port related statistics
        cnt = 0
        self.ports = {}

        for n in ctx.topo.graph.neighbors(self.id):
            port = Port(cnt)
            cnt += 1 
            port.target = n
            port.link_in = ctx.topo.graph.edges[n, self.id]['_link']
            port.link_out = ctx.topo.graph.edges[self.id, n]['_link']
            self.ports[(n, self.id)] = port

        # create a flow table object for this switch
        self.flowtable = FlowTable(self)

        # logic of the switch is implemented inside the engine; This is
        # similar to connecting a switch to a controller
        self.engine = kwargs.get("engine", Engine(self.ctx, **kwargs)) # routing engine

        self.cnt_backdelegations = 0
        self.cnt_adddelegations = 0
        
    def reset_counter(self):
        self.cnt_backdelegations = 0
        self.cnt_adddelegations = 0

    def on_event(self, ev):
        # periodic counter for statistics
        if isinstance(ev, EVStats):
            return self.engine.on_EVSwitchStats(self, ev)
        # a new flow arrives at the switch
        if isinstance(ev, EVSwitchNewFlow):
            return self.engine.on_EVSwitchNewFlow(self, ev)
        # the last packet of a flow arrives at the switch
        if isinstance(ev, EVSwitchLastPacketOfFlowArrived):
            return self.engine.on_EVSwitchLastPacketOfFlowArrived(self, ev)

