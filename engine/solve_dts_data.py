import logging
import math
import heapq
import copy
import time

logger = logging.getLogger(__name__)

from engine.errors import TimeoutError
from engine.solve_util import VDelegationStatus

class DTSData:
    """
    Stores DTS analytics based on raw input scenario
    """
    def __init__(self, ctx, switch):
        self.ctx = ctx
        self.timelimit = self.ctx.config.get('param_debug_total_time_limit', -1)
        self.look_ahead = ctx.config.get('param_dts_look_ahead')
        self.switch = switch # switch id from scenario
        self.ports = [] # list of all port labels
        self.utils_raw = {}
        self.utils_raw_per_port = {}
        self.demand_in = {}
        self.demand_in_per_port = {}
        self.demand_out = {}
        self.demand_out_per_port = {}
        self.flows_by_id = {} # <flow_id> -> flow-dict
        self.demand_by_flow_id = {} # <flow_id,tick> -> demand of flow in tick
        self.flows_of_timeslot_per_port = {} # <port_label,tick> -> [only active flows at the end of the time slot]
        self.flows_in_timeslot_per_port = {} # <port_label,tick> -> [all active flows in the whole time slot]
        self.new_flows_of_timeslot_per_port = {} # <port_label,tick> -> [new flows]

        # data after dts and rsa are executed, i.e., with flow delegation
        self.with_fd_delegated_demand_out_per_switch = {} # <remote_switch, tick> -> demand
        self.with_fd_delegated_demand_out_per_tick = {} # <tick> -> demand
        self.with_fd_delegated_demand_in_per_tick = {} # <tick> -> demand
        self.with_fd_demand_out = {}
        self.with_fd_delegated_demand_per_port = {} # <port_label, tick> -> delegated demand

        self.history_A2 = {}

        self.delegation_status = {}

    def get2(self):

        data = self.parse_flows(self.ctx.scenario.all_flow_rules)
        for k, v in data.items():
            print(k, v)

    def get_util(self, resultsDTS):
        """
        Return the effective utilization with flow delegation;
        To calculate this, the delegation decisions from the DTS algorithm are required which
        can be accessed via resultsDTS (returned by the DTS algo)
        """
        utils_raw = {}
        utils_per_port = {}
        total_delegated_demand_in = {}
        total_delegated_demand_out = {}
        total_backdelegations = 0

        data = self.get_flows_by_port()

        if len(data) == 0:
            # this happens if a switch has no host attached to it and there are no
            # flows for this switch; Note that this function does NOT calculate the final
            # utilization including delegated rules from other switches; it only focuses on one
            # switch (the other part is calculated separately); so basically all metrics
            # for this switch are simply zero 
            return utils_raw, utils_per_port, total_delegated_demand_in, total_delegated_demand_out, total_backdelegations


        for port, flows in data.items():

            # get id of the port
            index = -1
            for i, port_label in resultsDTS.ports.items():
                if port_label == port:
                    index = i
            assert(index >= 0)

            # extract delegated status and remote switch for this port
            delegated = {}
            remote_switch = {}
            for t, ports in resultsDTS.delegation_status.items():
                delegated[t] = ports.get(index).status 
                if ports.get(index).status == 1:
                    assert(ports.get(index).es >= 0) # there has to be an assigned remote switch or rsa has failed
                    remote_switch[t] = ports.get(index).es 


            # then calculate the utilization given the flows of the port and
            # the delegation status over time (delegated)
            util, demand_in, demand_out, delegated_demand_in, delegated_demand_out, backdelegations = self.parse_flows_with_delegation(flows, delegated, remote_switch)
            total_backdelegations += backdelegations
            utils_per_port[port] = util
            for t, v in utils_per_port[port].items():
                if not utils_raw.get(t):
                    utils_raw[t] = 0
                utils_raw[t] += v + delegated[t] # second part is overhead of aggregation rules!

            self.with_fd_delegated_demand_per_port[port] = {}
            for t, dataset in delegated_demand_in.items():
                if dataset.get(port):
                    assert(len(dataset) == 1)
                    try:
                        self.with_fd_delegated_demand_per_port[port][t] += dataset.get(port)
                    except KeyError:
                        self.with_fd_delegated_demand_per_port[port][t] = dataset.get(port)

                    try:
                        total_delegated_demand_in[t] += dataset.get(port)
                        self.with_fd_delegated_demand_in_per_tick[t] += dataset.get(port)
                    except KeyError:
                        total_delegated_demand_in[t] = dataset.get(port)
                        self.with_fd_delegated_demand_in_per_tick[t] = dataset.get(port)

            # delegated_demand_out: <tick, remote_switch> -> demand
            for t, dataset in delegated_demand_out.items():
                for remote_switch, v in dataset.items():
                    assert(remote_switch >= 0) # has to be a remote switch
                    if not self.with_fd_delegated_demand_out_per_switch.get(remote_switch):
                        self.with_fd_delegated_demand_out_per_switch[remote_switch] = {}
                    if not self.with_fd_delegated_demand_out_per_switch.get(remote_switch).get(t):  
                        self.with_fd_delegated_demand_out_per_switch[remote_switch][t] = 0
                    self.with_fd_delegated_demand_out_per_switch[remote_switch][t] += v   

        # collect aggregated demand for each time slot in outgoing direction
        for p, dataset in self.with_fd_delegated_demand_out_per_switch.items():
            for t, v in dataset.items():
                try:
                    self.with_fd_delegated_demand_out_per_tick[t] += v
                except KeyError:
                    self.with_fd_delegated_demand_out_per_tick[t] = v

        for t, v in total_delegated_demand_in.items():
            assert(math.isclose(self.with_fd_delegated_demand_out_per_tick.get(t, 0), v))

        return utils_raw, utils_per_port, total_delegated_demand_in, total_delegated_demand_out, total_backdelegations

    def calculate_raw_utils(self):
        utils_raw, utils_raw_per_port, demand_in, demand_out, flows_of_timeslot_per_port = self.get_raw_util()
        self.utils_raw = utils_raw
        self.utils_raw_per_port = utils_raw_per_port
        self.demand_in = demand_in
        self.demand_out = demand_out
        self.flows_of_timeslot_per_port = flows_of_timeslot_per_port
        self.ports = sorted(self.ports)




    def get_raw_util(self):
        demand_in = {}
        demand_out = {}
        utils_raw = {}
        utils_raw_per_port = {} 
        flows_of_timeslot_per_port = {}
        self.demand_out_per_port = {}
        for port, flows in self.get_flows_by_port().items():
            # store port labels
            self.ports.append(port)

            for flow in flows:
                self.flows_by_id[flow.get('flow_id')] = flow
                self.demand_by_flow_id[flow.get('flow_id')] = {} # init here, filled parse_flows

            util, _demand_in, _demand_out, flows_of_timeslot, flows_in_timeslot, new_flows_of_timeslot = self.parse_flows(flows)
            utils_raw_per_port[port] = util

            flows_of_timeslot_per_port[port] = flows_of_timeslot
            self.flows_in_timeslot_per_port[port] = flows_in_timeslot
            self.new_flows_of_timeslot_per_port[port] = new_flows_of_timeslot

            for t, v in util.items(): 
                try:
                    utils_raw[t] += v
                except KeyError:
                    utils_raw[t] = v 

            self.demand_in_per_port[port] = {}
            for t, dataset in _demand_in.items():
                if dataset.get(port):
                    assert(len(dataset) == 1) # only one port allowed here
                    self.demand_in_per_port[port][t] = dataset.get(port, 0)
                    try:
                        demand_in[t] += dataset.get(port)
                    except KeyError:
                        demand_in[t] = dataset.get(port)

            for t, dataset in _demand_out.items():
                for p, v in dataset.items():
                    assert(p != port) # redirection to ingress port is not allowed
                    if not self.demand_out_per_port.get(p): self.demand_out_per_port[p] = {}
                    if not self.demand_out_per_port.get(p).get(t):  self.demand_out_per_port[p][t] = 0
                    self.demand_out_per_port[p][t] += v              

        # collect aggregated demand for each time slot in outgoing direction
        for p, dataset in self.demand_out_per_port.items():
            for t, v in dataset.items():
                try:
                    demand_out[t] += v
                except KeyError:
                    demand_out[t] = v

        for t, v in demand_in.items():
            assert(math.isclose(demand_out.get(t, 0), v))


        return utils_raw, utils_raw_per_port, demand_in, demand_out, flows_of_timeslot_per_port
 

    def get_flows_by_port(self):
        switch = self.switch
        all_labels = []
        dummy_switches = []


        flows_by_port = {}

        # collect all relevant flows here; this includes all flows
        # associated with switch (obviously), but also all flows from
        # neighboring switches that forward traffic via switch
        flows = self.ctx.scenario.flows_of_switch.get(switch)
        #for neighbor in self.ctx.scenario.g.neighbors(switch):
        #    for flow in self.ctx.scenario.flows_of_switch.get(neighbor):
        #        if switch in flow.get('path'):
        #            flows.append(flow)

        for flow in flows:

            path = flow.get('path')
            assert(switch in path)
                
            # case 1: path has length 1
            if len(path) == 1:
                source_label = flow.get('source_label')
                target_label = flow.get('target_label')

            # case 2: path > 1 and switch is the first hop
            #   [switch, others, ...] sxh5 -> switch -> others -> syh18
            #   => this flow starts at this switch, i.e., ingress port is the connection to the host 
            if len(path) > 1 and path[0] == switch:
                source_label = flow.get('source_label')
                target_label = 'dummy_switch_%d' % path[1]
                #print("add dummy target", source_label, target_label)
                if not target_label in dummy_switches:
                    dummy_switches.append(target_label)

            # case 3: path > 1 and switch is the last hop
            if len(path) > 1 and path[-1] == switch:
                source_label = 'dummy_switch_%d' % path[-2]
                target_label = flow.get('target_label')
                #print("add dummy source", source_label, target_label)
                if not source_label in dummy_switches:
                    dummy_switches.append(source_label)

            # case 4: path > 2 and switch is not first or last hop
            if len(path) > 2 and path[0] != switch and path[-1] != switch:
                source_label = 'dummy_switch_%d' % path[path.index(switch)-1]
                target_label = 'dummy_switch_%d' % path[path.index(switch)+1]
                if not source_label in dummy_switches:
                    dummy_switches.append(source_label)
                if not target_label in dummy_switches:
                    dummy_switches.append(target_label)

            if not source_label in all_labels:
                all_labels.append(source_label)
            if not target_label in all_labels:
                all_labels.append(target_label)

            used_flow_rule = dict(flow) # make a copy
            used_flow_rule['source_label'] = source_label
            used_flow_rule['target_label'] = target_label

            try:
                flows_by_port[source_label].append(used_flow_rule)
            except KeyError:
                flows_by_port[source_label] = [used_flow_rule]

        # it can happen that flow_by_port is not initiated for some of the
        # connected hosts or switches (if no flow is assigned to this host/switch at all)
        for label in all_labels:
            if not flows_by_port.get(label):
                flows_by_port[label] = []

        return flows_by_port

    def parse_flows_with_delegation(self, flows, delegated, remote_switch):
        """
        The function simulates utilization and demand based on whether the port is
        delegated in time slot t (if delegated[t]==1) or not  (if delegated[t]==0).    
        Note that the flows array is dedicated to a single port (delegation template) and the
        delegated dict is also dedicated to the same port. 
        """
        self.check_for_time_limit()
        backdelegations = 0
        flows_of_timeslot = {}
        utils_raw = {} # the raw utilization in time slot t

        # first step is to pre-attach the flows to timeslots
        timeslots = []
        for flow in flows:
            s = math.floor(flow.get('start'))
            timeslots.append(s)
            try:
                flows_of_timeslot[s].append(flow)
            except KeyError:
                flows_of_timeslot[s] = [flow]  # int(s) is the time slot
                utils_raw[s] = 0
    
        # make sure there are no gaps 
        for s in range(0, self.ctx.maxtick):
            if not flows_of_timeslot.get(s):
                flows_of_timeslot[s] = []
                utils_raw[s] = 0    

        still_active = [] # the flows that are active and not delegated
        delegated_active = [] # the flows that are active and delegated
        demand_in = {} # store received demand (key is port label of sender)
        demand_out = {} # store forwarded demand (key is port label of receiver)
        delegated_demand_in = {}
        delegated_demand_out = {}

        for t, flows in sorted(flows_of_timeslot.items()):

            demand_in[t] = {} 
            demand_out[t] = {} 
            delegated_demand_in[t] = {}
            delegated_demand_out[t] = {}

            # decision was revoked --> all delegated rules are moved back to the delegation switch
            if delegated.get(t-1) == 1 and delegated.get(t) == 0:
                still_active += delegated_active
                backdelegations += len(delegated_active)
                delegated_active = []  
                heapq.heapify(still_active) # important!

            # if flow are delegated we need to track the demand
            if len(delegated_active) > 0:
                assert(delegated.get(t) == 1)
                assigned_remote_switch = remote_switch.get(t)
                assert(assigned_remote_switch >= 0)

                # take care of the rules that finish in this time slot
                while len(delegated_active) > 0 and heapq.nsmallest(1, delegated_active)[0][0] < t+1:
                    endtime, flow = heapq.heappop(delegated_active) 
                    demand_factor = endtime-t
                    assert(demand_factor >= 0 and demand_factor <= 1)
                    demand = flow.get('demand_per_tick') * demand_factor
                    src = flow.get('source_label')
                    tgt = flow.get('target_label')
                    try:
                        delegated_demand_in[t][src] += demand
                    except KeyError:
                        delegated_demand_in[t][src] = demand
                    try:
                        delegated_demand_out[t][assigned_remote_switch] += demand
                    except KeyError:
                        delegated_demand_out[t][assigned_remote_switch] = demand 
                # take care of the remaining delegated rules (are delegated for the whole time slot)
                for endtime, flow in delegated_active:
                    demand = flow.get('demand_per_tick')
                    src = flow.get('source_label')
                    tgt = flow.get('target_label')
                    try:
                        delegated_demand_in[t][src] += demand
                    except KeyError:
                        delegated_demand_in[t][src] = demand
                    try:
                        delegated_demand_out[t][assigned_remote_switch] += demand
                    except KeyError:
                        delegated_demand_out[t][assigned_remote_switch] = demand   

            # handle the active rules (some might have ended by now)
            while len(still_active) > 0 and heapq.nsmallest(1, still_active)[0][0] < t+1:
                endtime, flow = heapq.heappop(still_active) 
                #if endtime < t:
                #    # because we do not remove the rules from delegated_active, there are 
                #    # also older rules in here that are not relevant any more (these rules
                #    # finished while delegation was active); are skipped here
                #    continue
                demand_factor = endtime-t
                assert(demand_factor >= 0 and demand_factor <= 1)
                demand = flow.get('demand_per_tick') * demand_factor
                src = flow.get('source_label')
                tgt = flow.get('target_label')
                try:
                    demand_in[t][src] += demand
                except KeyError:
                    demand_in[t][src] = demand
                try:
                    demand_out[t][tgt] += demand
                except KeyError:
                    demand_out[t][tgt] = demand  

            utils_raw[t] += len(still_active)
            # handle demand of all rules that are left in still_active (means the expire time of
            # these rules is in the future)
            for endtime, flow in still_active:
                demand = flow.get('demand_per_tick')
                src = flow.get('source_label')
                tgt = flow.get('target_label')
                try:
                    demand_in[t][src] += demand
                except KeyError:
                    demand_in[t][src] = demand
                try:
                    demand_out[t][tgt] += demand
                except KeyError:
                    demand_out[t][tgt] = demand  
 


            # then handle the "new" flow rules associated with this time slot; this part is now different
            # from the other function because a new flow rule is only included here if the delegation decision
            # for that time slot is set to 0 (not delegated)
            if delegated.get(t) == 0:
                for flow in flows:
                    s = flow.get('start')
                    d = flow.get('duration')
                    end = s+d
                    # then check if the new flow rule is still active at the barrier to the next time slot;
                    # if this is the case, it counts (otherwise not)
                    if end >= t+1:
                        utils_raw[t] += 1
                        heapq.heappush(still_active, (end, flow))
                        demand_factor = 1-(s-t)
                        assert(demand_factor >= 0 and demand_factor <= 1)
                        demand = flow.get('demand_per_tick') * demand_factor
                        src = flow.get('source_label')
                        tgt = flow.get('target_label')
                        try:
                            demand_in[t][src] += demand
                        except KeyError:
                            demand_in[t][src] = demand
                        try:
                            demand_out[t][tgt] += demand
                        except KeyError:
                            demand_out[t][tgt] = demand 

            else:
                # we also need to keep track of delegated flow rules because the delegation decision can be revoked
                # in the future; in this case, all flows that are still active (and delegated) will "come back"
                assigned_remote_switch = remote_switch.get(t)
                assert(assigned_remote_switch >= 0)
                for flow in flows:
                    s = flow.get('start')
                    d = flow.get('duration')
                    end = s+d
                    if end >= t+1:
                        heapq.heappush(delegated_active, (end, flow))        
                        demand_factor = 1-(s-t)
                        assert(demand_factor >= 0 and demand_factor <= 1)
                        demand = flow.get('demand_per_tick') * demand_factor
                        src = flow.get('source_label')
                        tgt = flow.get('target_label')
                        try:
                            delegated_demand_in[t][src] += demand
                        except KeyError:
                            delegated_demand_in[t][src] = demand
                        try:
                            delegated_demand_out[t][assigned_remote_switch] += demand
                        except KeyError:
                            delegated_demand_out[t][assigned_remote_switch] = demand 

        # finally, handle all leftover flows until no one is left
        while len(still_active) > 0 or len(delegated_active) > 0:
            t += 1

            demand_in[t] = {} 
            demand_out[t] = {} 

            # decision was revoked --> all delegated rules are moved back to the delegation switch
            if delegated.get(t-1) == 1 and delegated.get(t) == 0:
                still_active += delegated_active
                backdelegations += len(delegated_active)
                delegated_active = []   
                heapq.heapify(still_active) # important!

            # if flow are delegated we need to track the demand
            if len(delegated_active) > 0:
                assert(delegated.get(t) == 1)
                assigned_remote_switch = remote_switch.get(t)
                assert(assigned_remote_switch >= 0)    
                while len(delegated_active) > 0 and heapq.nsmallest(1, delegated_active)[0][0] < t+1:
                    endtime, flow = heapq.heappop(delegated_active) 
                    demand_factor = endtime-t
                    assert(demand_factor >= 0 and demand_factor <= 1)
                    demand = flow.get('demand_per_tick') * demand_factor
                    src = flow.get('source_label')
                    tgt = flow.get('target_label')
                    try:
                        delegated_demand_in[t][src] += demand
                    except KeyError:
                        delegated_demand_in[t][src] = demand
                    try:
                        delegated_demand_out[t][assigned_remote_switch] += demand
                    except KeyError:
                        delegated_demand_out[t][assigned_remote_switch] = demand  
                # take care of the remaining delegated rule (are delegated for the whole time slot)
                for endtime, flow in delegated_active:
                    demand = flow.get('demand_per_tick')
                    src = flow.get('source_label')
                    tgt = flow.get('target_label')
                    try:
                        delegated_demand_in[t][src] += demand
                    except KeyError:
                        delegated_demand_in[t][src] = demand
                    try:
                        delegated_demand_out[t][assigned_remote_switch] += demand
                    except KeyError:
                        delegated_demand_out[t][assigned_remote_switch] = demand  

            # heapq.nsmallest(1, still_active)[0][0]
            while len(still_active) > 0 and still_active[0][0] < (t+1):
                heapq.heappop(still_active) 
            utils_raw[t] = len(still_active)     

            # handle demand of all rules that are left in still_active (means the expire time of
            # these rules is in the future)
            for endtime, flow in still_active:
                demand = flow.get('demand_per_tick')
                src = flow.get('source_label')
                tgt = flow.get('target_label')
                try:
                    demand_in[t][src] += demand
                except KeyError:
                    demand_in[t][src] = demand
                try:
                    demand_out[t][tgt] += demand
                except KeyError:
                    demand_out[t][tgt] = demand 

        # make sure ingoing and outgoing demand are equal
        for t, sources in delegated_demand_in.items():
            total_in = 0
            for p, v in sources.items():
                total_in += v
            total_out = 0
            for r, v in delegated_demand_out.get(t).items():
                total_out += v  
            assert(math.isclose(total_in, total_out))


        return utils_raw, demand_in, demand_out, delegated_demand_in, delegated_demand_out, backdelegations

    def parse_flows(self, flows):
        self.check_for_time_limit()
        # the flows are sorted based in their start value
        flows_of_timeslot = {} 
        flows_of_timeslot_ids = {} # active flows per time slot (ids only)
        flows_in_timeslot_ids = {}
        added_for_in = {}
        new_flows_of_timeslot_ids = {} # new flows per time slot (ids only)
        utils_raw = {} # the raw utilization in time slot t

        # first step is to pre-attach the flows to timeslots
        timeslots = []
        for flow in flows:
            s = math.floor(flow.get('start'))
            timeslots.append(s)
            try:
                flows_of_timeslot[s].append(flow)
                new_flows_of_timeslot_ids[s].append(flow.get('flow_id'))
            except KeyError:
                flows_of_timeslot[s] = [flow]  # int(s) is the time slot
                new_flows_of_timeslot_ids[s] = [flow.get('flow_id')]
                utils_raw[s] = 0

    
        # make sure there are no gaps 
        for s in range(0, self.ctx.maxtick):
            if not flows_of_timeslot.get(s):
                flows_of_timeslot[s] = []
                new_flows_of_timeslot_ids[s] = []
                utils_raw[s] = 0    

        demand_in = {} # store received demand (key is port label of sender)
        demand_out = {} # store forwarded demand (key is port label of receiver)
        still_active = [] # rules that are still active from last time slot(s)

        for t, flows in sorted(flows_of_timeslot.items()):

            demand_in[t] = {} 
            demand_out[t] = {} 

            # first handle the still_active array (some flows might have ended by now)
            while len(still_active) > 0 and heapq.nsmallest(1, still_active)[0][0] < t+1:
                endtime, flow = heapq.heappop(still_active) 
                #try:
                #    flows_of_timeslot_ids[t].append(flow.get('flow_id'))
                #except:
                #    flows_of_timeslot_ids[t] = [flow.get('flow_id')]
                try:
                    added_for_in[t].append(flow.get('flow_id'))
                except:
                    added_for_in[t] = [flow.get('flow_id')]
                demand_factor = endtime-t
                assert(demand_factor >= 0 and demand_factor <= 1)
                demand = flow.get('demand_per_tick') * demand_factor
                src = flow.get('source_label')
                tgt = flow.get('target_label')
                try:
                    demand_in[t][src] += demand
                except KeyError:
                    demand_in[t][src] = demand
                try:
                    demand_out[t][tgt] += demand
                except KeyError:
                    demand_out[t][tgt] = demand  
                self.demand_by_flow_id[flow.get('flow_id')][t] = demand

    
            utils_raw[t] += len(still_active)

            # handle demand of all rules that are left in still_active (means the expire time of
            # these rules is in the future)
            for endtime, flow in still_active:
                try:
                    flows_of_timeslot_ids[t].append(flow.get('flow_id'))
                except:
                    flows_of_timeslot_ids[t] = [flow.get('flow_id')]
                demand = flow.get('demand_per_tick')
                src = flow.get('source_label')
                tgt = flow.get('target_label')
                try:
                    demand_in[t][src] += demand
                except KeyError:
                    demand_in[t][src] = demand
                try:
                    demand_out[t][tgt] += demand
                except KeyError:
                    demand_out[t][tgt] = demand   
                self.demand_by_flow_id[flow.get('flow_id')][t] = demand


            # then handle the "new" flows associated with this time slot
            for flow in flows:
                try:
                    flows_of_timeslot_ids[t].append(flow.get('flow_id'))
                except:
                    flows_of_timeslot_ids[t] = [flow.get('flow_id')]
                s = flow.get('start')
                d = flow.get('duration')
                endtime = s+d
                # then check if the new flow is still active at the barrier to the next time slot;
                # if this is the case, it counts (otherwise not)
                if endtime >= t+1:
                    utils_raw[t] += 1
                    heapq.heappush(still_active, (endtime, flow))
                    demand_factor = 1-(s-t)
                    assert(demand_factor >= 0 and demand_factor <= 1)
                    demand = flow.get('demand_per_tick') * demand_factor
                    src = flow.get('source_label')
                    tgt = flow.get('target_label')
                    try:
                        demand_in[t][src] += demand
                    except KeyError:
                        demand_in[t][src] = demand
                    try:
                        demand_out[t][tgt] += demand
                    except KeyError:
                        demand_out[t][tgt] = demand  
                    self.demand_by_flow_id[flow.get('flow_id')][t] = demand
                else:
                    raise Exception('Flow rules shorter than 1 time slot are not supported! Increase rule duration.')

        # finally, handle all leftover flows until no one is left
        while len(still_active) > 0:
            t += 1
            # heapq.nsmallest(1, still_active)[0][0]
            while len(still_active) > 0 and still_active[0][0] < (t+1):
                endtime, flow = heapq.heappop(still_active) 
                #try:
                #    flows_of_timeslot_ids[t].append(flow.get('flow_id'))
                #except:
                #    flows_of_timeslot_ids[t] = [flow.get('flow_id')]            
                demand = (endtime-t)*flow.get('demand_per_tick')
                assert(demand >= 0 and demand < 1)
                src = flow.get('source_label')
                tgt = flow.get('target_label')
                try:
                    demand_in[t][src] += demand
                except KeyError:
                    demand_in[t][src] = demand
                try:
                    demand_out[t][tgt] += demand
                except KeyError:
                    demand_out[t][tgt] = demand 
                self.demand_by_flow_id[flow.get('flow_id')][t] = demand 

            utils_raw[t] = len(still_active)  

            # handle demand of all rules that are left in still_active (means the expire time of
            # these rules is in the future)
            for endtime, flow in still_active:
                try:
                    flows_of_timeslot_ids[t].append(flow.get('flow_id'))
                except:
                    flows_of_timeslot_ids[t] = [flow.get('flow_id')]       
                demand = flow.get('demand_per_tick')
                src = flow.get('source_label')
                tgt = flow.get('target_label')
                try:
                    demand_in[t][src] += demand
                except KeyError:
                    demand_in[t][src] = demand
                try:
                    demand_out[t][tgt] += demand
                except KeyError:
                    demand_out[t][tgt] = demand 
                self.demand_by_flow_id[flow.get('flow_id')][t] = demand

        
        # flows_in_timeslot_ids contains all flows IN THE WHOLE time slot, i.e., every flow that is
        # started, stopped and active. The other set (flows_of_timeslot_ids) contains only the flows that
        # are active at the end of the time slot. This is an important difference. The latter is used to
        # determine the utilization (by definition). However, the former set is also required, e.g., if
        # the total link utilization for the time slot is calculated. In this case, the flows that ended
        # prior to the end of the time slot have already contributed to the link utilization. 
        flows_in_timeslot_ids = copy.deepcopy(flows_of_timeslot_ids) # contains arrays, so a deepcopy is required!
        for t, add in added_for_in.items():
            try:
                flows_in_timeslot_ids[t] += add
            except KeyError:
                flows_in_timeslot_ids[t] = add

        # (assert stuff here slows down the process by approx. 10%, feel free to comment it out)
        # make sure that the demand sum of the individual flows associated with a time slot
        # equals the total demand calculated for the ingress port
        for t, sources in demand_in.items():
            assert(len(sources) <= 1) # parse_flows handles only flows of one port
            if len(sources) == 1:
                demand = list(sources.values())[0]
                port =  list(sources.keys())[0]
                checksum = 0
                for flow in flows_in_timeslot_ids.get(t, []):
                    checksum += self.demand_by_flow_id.get(flow).get(t, 0)  
                assert(math.isclose(demand, checksum))

        # check some other invariants
        assert( len(new_flows_of_timeslot_ids) == len(set(new_flows_of_timeslot_ids)))
        assert( len(flows_of_timeslot_ids) == len(set(flows_of_timeslot_ids)))
        for t, v in utils_raw.items():
            assert(len(flows_of_timeslot_ids.get(t, [])) == v)

        return utils_raw, demand_in, demand_out, flows_of_timeslot_ids, flows_in_timeslot_ids, new_flows_of_timeslot_ids

    def get_first_overutilization(self):
        threshold = self.ctx.scenario.threshold
        for t in range(0, self.ctx.maxtick-2):
            if self.utils_raw.get(t,0) > threshold:
                return t
        return 0   

    def get_last_overutilization(self):
        threshold = self.ctx.scenario.threshold
        for t in reversed(range(0, self.ctx.maxtick-2)):
            if self.utils_raw.get(t,0) > threshold:
                return t
        return 0 

    def getDemandRaw(self, tick, port_status):
        D = {}
        for p, port in port_status.items():
            #print(port.label, self.demand_in_per_port.get(port.label)) 
            D[p] = self.demand_in_per_port.get(port.label).get(tick, 0)
        return D

    def getDemand(self, tick, port_status):
        D = {}
        for p, port in port_status.items():
            demand = 0
            if port.is_delegated:
                pre_existing_set = self.flows_in_timeslot_per_port.get(port.label).get(port.delegated_at-1, [])
                #pre_existing_set = port.epochs[port.delegated_at-1].active
                demand = 0
                flows = self.flows_in_timeslot_per_port.get(port.label).get(tick, [])
                for flow in flows:
                    # we are interested in all delegated flows here
                    if flow not in pre_existing_set: 
                        demand += self.demand_by_flow_id.get(flow).get(tick)
                D[p] = demand
            else:
                flows = self.flows_in_timeslot_per_port.get(port.label).get(tick, [])
                for flow in flows:
                    demand += self.demand_by_flow_id.get(flow).get(tick)
                D[p] = demand 
        return D

    def getDemand2(self, tick, port_status):
        D = {}
        for p, port in port_status.items():
            D[p] = {}
            if not port.is_delegated:
                D[p] = {}
                pre_existing_set = self.flows_in_timeslot_per_port.get(port.label).get(tick-1, [])
                for d in range(0, self.look_ahead):
                    demand = 0
                    flows = self.flows_in_timeslot_per_port.get(port.label).get(tick+d, [])
                    for flow in flows:
                        # we are interested in all delegated flows here
                        if flow not in pre_existing_set: 
                            demand += self.demand_by_flow_id.get(flow).get(tick+d)     
                    D[p][d] = demand  
        return D       

    #c-1
    def getDemand1(self, tick, port_status):
        D = {}
        for p, port in port_status.items():
            D[p] = {}
            if port.is_delegated:
                D[p] = {}
                pre_existing_set = self.flows_in_timeslot_per_port.get(port.label).get(port.delegated_at-1, [])
                for d in range(0, self.look_ahead):
                    demand = 0
                    flows = self.flows_in_timeslot_per_port.get(port.label).get(tick+d-1, [])
                    for flow in flows:
                        # we are interested in all delegated flows here
                        if flow not in pre_existing_set: 
                            demand += self.demand_by_flow_id.get(flow).get(tick+d-1)
                    D[p][d] = demand
            else:
                for d in range(0, self.look_ahead):
                    D[p][d] = 0   
        return D

    # c-2
    def getMu(self):
        mu = []
        threshold = self.ctx.scenario.threshold
        for t in range(0, self.ctx.maxtick-2):
            if self.utils_raw.get(t,0) > threshold:
                mu.append(1)
            else:
                mu.append(0)
        return mu

    def getSize(self, tick):
        return self.utils_raw.get(tick)

    # overview usage A1 etc
    #
    #    [ ][ ] A1 --> do nothing (stay undelegated) (x=1)
    #    [ ][x] E1 --> add delegation (x=0)
    #    [x][x] A2 --> do nothing (stay delegated) (y=1)
    #    [x][ ] E2 --> remove delegation (y=0)

    #c-1
    def getE(self, tick, port_status):
        # flow table development if a port is delegated
        E = {}
        for p, port in port_status.items():
            if not port.is_delegated:
                E[p] = {}
                flows = self.flows_of_timeslot_per_port.get(port.label).get(tick-1, [])
                for d in range(0, self.look_ahead):
                    cnt = 0
                    # we only consider flows here that are "not new", i.e., flows that were
                    # already present when the delegation started; all the other new flows will
                    # be handled by the delegation switch
                    for flow in self.flows_of_timeslot_per_port.get(port.label).get(tick+d, []):
                        if flow in flows: cnt += 1
                    E[p][d] = cnt 
        return E

    #c-0
    def getE2(self, tick, port_status):
        E2 = {}
        for p, port in port_status.items():
            if port.is_delegated:
                E2[p] = {}
                for d in range(0, self.look_ahead):
                    E2[p][d] = self.utils_raw_per_port.get(port.label).get(tick+d)
        return E2

    #c-0
    def getA(self, tick, port_status):
        A = {}
        for p, port in port_status.items():
            if not port.is_delegated: 
                A[p] = {}
                for d in range(0, self.look_ahead):
                    A[p][d] = self.utils_raw_per_port.get(port.label).get(tick+d)
        return A

    #c-1
    def getA2(self, tick, port_status):
        A2 = {}  
        for p, port in port_status.items():
            if port.is_delegated:
                A2[p] = {}
                """
                history = self.history_A2.get(port.label)
                if history != None:
                    # this port was already delegated in the last time slot
                    print("existing history", port.label, tick)
                    pass
                else:
                    print("new history", port.label, tick, port.delegated_at)
                    assert(port.delegated_at == tick-1)
                    # first time the port is delegated, create a new history
                    active_last_timeslot = self.flows_of_timeslot_per_port.get(port.label).get(tick-2, [])
                    self.history_A2[port.label] = active_last_timeslot
                    history = self.history_A2[port.label]
                    #print(history)
                    #print(self.flows_of_timeslot_per_port.get(port.label))
                """

                flows = self.flows_of_timeslot_per_port.get(port.label).get(port.delegated_at-1, [])
                for d in range(0, self.look_ahead):
                    cnt = 0
                    # note that epochs is shifted already by 1 and we need 
                    # the next tick
                    for flow in self.flows_of_timeslot_per_port.get(port.label).get(tick+d, []):
                       if flow in flows: cnt += 1;
                    A2[p][d] = cnt
        return A2


    def simulate_delegations(self, port, delegated_at_array, restrict=None):
        
        threshold = self.ctx.scenario.threshold
        # transform delegated_at_array; this is necessary because
        # the simulator tick is shifted by one (otherwise the values
        # would not fit)
        if len(delegated_at_array) > 0:
            delegated_at_array = [x+1 for x in delegated_at_array]
        old = None

        # optimization to reduce processing time
        if restrict:
            restrict[0]
            cut = None
            last = None
            for t in reversed(delegated_at_array):
                if t > restrict[0]: continue;
                if not last:
                    last = t
                else:
                    if t+1 < last:
                        cut = t
                        break;
                    last = t
            if cut:
                #print("save", delegated_at_array.index(cut), len(delegated_at_array))
                ix = delegated_at_array.index(cut)
                delegated_at_array = delegated_at_array[ix:]  



        flowset = []
        utilization = {}

        for tick in delegated_at_array:

            new_flows = self.new_flows_of_timeslot_per_port.get(port.label).get(tick-1, [])
            active_flows = self.flows_of_timeslot_per_port.get(port.label).get(tick-1, [])

            if tick-1 == old:
                # continue with an active flowset
                flowset += new_flows[:]
            else:
                # create a new flowset
                flowset = new_flows[:]

            reduce_cnt = 0
            remove_from_flowset = []
            for flow in flowset:
                if flow in active_flows:
                    reduce_cnt += 1    
                else:
                    remove_from_flowset.append(flow)

            for flow in remove_from_flowset:
                flowset.remove(flow)    

            # the +1 resembles the overhead that we have by using the port for
            # delegation!
            utilization[tick] = len(active_flows) - reduce_cnt + 1
            old = tick

        result_vector = []

        # we can select a subset of the vector using restrict
        if restrict:
            start = restrict[0]
            stop = restrict[1]
            for tick in range(start, stop+1):
                if tick in delegated_at_array:
                    result_vector.append(utilization.get(tick))
                else:
                    result_vector.append(self.utils_raw_per_port.get(port.label).get(tick-1))
        else:  
            for tick in range(1, self.ctx.maxtick):
                if tick in delegated_at_array:
                    result_vector.append(utilization.get(tick))
                else:
                    result_vector.append(self.utils_raw_per_port.get(port.label).get(tick-1))

        # backdelegations are calculated here as well, i.e., the number of flows
        # that will be removed due to changing the delegation status from active to
        # inactive. 
        # [144,145,149,150] --> [(144,145), (145,149), (149,150)]
        backdelegations = 0
        if restrict:
            start = restrict[0]-1 # -1 is essential here to include the history delegation status!
            stop = restrict[1]
            toconsider = []
            for tick in range(start, stop+1):
                if tick in delegated_at_array:   
                    toconsider.append(tick)

            if len(toconsider) > 0:
                for t1, t2 in zip(toconsider,toconsider[1:]):
                    if t1+1 != t2:
                        # in this case, the delegation is removed at t1 and we 
                        # have to count the flows that will be moved back to the delegation
                        # switch; In the above example, this happens form the second entry (145,149)
                        active = self.flows_of_timeslot_per_port.get(port.label).get(t1-1, [])
                        without_delegation = len(active)
                        with_delegation = utilization.get(t1)
                        assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                        backdelegations += (without_delegation + 1 - with_delegation)


                t = toconsider[-1]
                active = self.flows_of_timeslot_per_port.get(port.label).get(t-1, [])
                without_delegation = len(active)
                with_delegation = utilization.get(t)
                assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                backdelegations += (without_delegation + 1 - with_delegation)

            # delegations are only required if there is overutilization. In the backdelegations case,
            # it is usually cheaper for the solver to keep a port delegated if that port
            # was already delegated before regardless of whether delegation at all is still
            # required or not. That isbecause the cost to remove the delegation is equal to
            # the amount of flows that were delegated in the recent past which is >0 in these cases.
            # To avoid such scenarios, we check whether there is still a overutilization
            # and artifically increase the cost for all solutions that try to delegate. Note that
            # we do NOT want to consider the history tick (start-1) here!
            for tick in range(restrict[0], restrict[1]+1):
                if tick in delegated_at_array and self.utils_raw.get(tick, 0) <= threshold: 
                    backdelegations += 10/self.look_ahead
                    
            """    
            if self.mu[restrict[1]+1] == 0:    
                cnt = 0
                for tick in range(restrict[0], restrict[1]+1):
                    if tick in delegated_at_array: cnt += 1
                if cnt > 0:
                    # good value for single backdelegation objective: 1.38
                    # values below 0 will keep ports longer even if no delegation is required
                    backdelegations *= 0.9 
            """
        else:
            if len(delegated_at_array) > 0:
                for t1, t2 in zip(delegated_at_array,delegated_at_array[1:]):
                    if t1+1 != t2:
                        active = self.flows_of_timeslot_per_port.get(port.label).get(t1-1, [])
                        without_delegation = len(active)
                        with_delegation = utilization.get(t1)
                        assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                        backdelegations += (without_delegation + 1 - with_delegation) 

                t = delegated_at_array[-1]
                active = self.flows_of_timeslot_per_port.get(port.label).get(t-1, [])
                without_delegation = len(active)
                with_delegation = utilization.get(t)
                assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                backdelegations += (without_delegation + 1 - with_delegation)    


        return result_vector, backdelegations


    def simulate_delegation_demand2(self, port, delegated_at_array, restrict):

        start = restrict[0]
        stop = restrict[1]
        demands = []
        if len(delegated_at_array) == 0:
            for tick in range(start, stop):
                demands.append(0)
            assert(len(demands) == stop-start)
            return demands    
        # transform delegated_at_array; this is necessary because
        # the simulator tick is shifted by one (otherwise the values
        # would not fit)
        if len(delegated_at_array) > 0:
            delegated_at_array = [x+1 for x in delegated_at_array]

        flowset = []
        use_start = min(start-1, min(delegated_at_array))
        for tick in range(use_start, stop):
            demand = 0

            new_flows = self.new_flows_of_timeslot_per_port.get(port.label).get(tick-1, [])
            active_flows = self.flows_of_timeslot_per_port.get(port.label).get(tick-1, [])

            if tick in delegated_at_array:
                if tick-1 in delegated_at_array:
                    # continue with an active flowset
                    flowset += new_flows[:]
                else:
                    flowset = new_flows[:]
                    demand = 0    
            else:
                # create a new flowset
                flowset = []
                demand = 0


            remove_from_flowset = []
            for flow in flowset:
                demand += self.demand_by_flow_id.get(flow).get(tick-1, 0) 
                if flow not in active_flows:
                    remove_from_flowset.append(flow)

            for flow in remove_from_flowset:
                flowset.remove(flow)   

            if tick >= start-1 and tick < stop-1:
                demands.append(demand)
        assert(len(demands) == stop-start)
        return demands

    def check_for_time_limit(self):
        if self.timelimit > 0:
            if self.ctx.started > 0:
                if time.time() - self.ctx.started > self.timelimit:
                    raise TimeoutError()