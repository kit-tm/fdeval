# DO NOT USE
# this file is deprecated; 
# only required for some older files that are not yet updated (global_single_v3.py)
# use solve_dts.py instead!

from gurobipy import *

import itertools
import random
import logging
import time
import sys
import os
import numpy
import io
from contextlib import redirect_stdout
from engine.solve_util import VDelegationStatus, VDSSResult


logger = logging.getLogger(__name__)


class VEpoch:

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.active = []
        self.started = []
        self.finished = []
        self.startstop = []
        self.diff = 0


class VPort:

    def __init__(self, port):
        self.port = port
        self.label = None
        self.epochs = []
        self.diffs = []
        self.is_delegated = 0
        self.delegated_at = 0
        self.old_flows = []

class VTable:

    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.verbose = kwargs.get('verbose', False)
        self.maxtick = 450
        self.verbose = False
        self.threshold = ctx.config.get('param_topo_switch_capacity')
        self.look_ahead = ctx.config.get('param_dts_look_ahead')
        self.skip_ahead = ctx.config.get('param_dts_skip_ahead')
        self.commands = {}

        self.counters_decisions = {}
        self.counters_demand = 0
        self.counters_util = {}
        self.counters_evicted = {}
        self.counters_evicted2 = {}




        # _copy_for_switch_id is set for experiments with more than one switch
        # the value specifies the port for which the delegation status selection has to be
        # calculated
        # !! edit: the first part (_copy_for_switch_id) is deprecated, left here for 
        # compatability of older experiments
        dss_ports = {} # dss result have to be serialized
        if ctx._copy_for_switch_id >= 0:
            static_ES = switch = ctx.topo.get_switch_by_label('ES').id

            switch = ctx.topo.get_switch_by_id(ctx._copy_for_switch_id)

            print("")
            print("dts solver initiated (_copy_for_switch_id is set)")
            print("  number of flow rules", len(ctx.flows))
            print("  threshold", self.threshold)
            print("  look_ahead", self.look_ahead)
            print("  skip_ahead", self.skip_ahead)

            self.ports = {}
            for _, port in switch.ports.items():
                if port.target == static_ES: continue;
                logger.info("prepare port %d of switch %d" % (port.target, ctx._copy_for_switch_id))
                self.ports[port.id] = VPort(port)
                dss_ports[port.id] = ctx.topo.get_label_by_id(port.target)

                self.graph.nodes[link_id[0]]['label'] 

                flows = []
                for flow in ctx.flows:
                    for src, tgt in flow.path_tuples:
                        if tgt == ctx._copy_for_switch_id and src == port.target:
                            flows.append(flow)

                for epoch_cnt in range(1, self.maxtick):
                    e = VEpoch(epoch_cnt-1, epoch_cnt)
                    e.active = [flow for flow in flows if flow.start < epoch_cnt and flow.start+flow.duration > epoch_cnt]
                    e.started = [flow for flow in flows if flow.start > epoch_cnt-1 and flow.start < epoch_cnt]
                    e.finished = [flow for flow in flows if flow.start+flow.duration > epoch_cnt-1 and flow.start+flow.duration < epoch_cnt]
                    e.startstop = [flow for flow in flows if flow.start > epoch_cnt-1 and flow.start+flow.duration < epoch_cnt]

                    e.diff = len(e.started) - len(e.finished)
                    self.ports[port.id].epochs.append(e)
                    self.ports[port.id].diffs.append(e.diff)          


        else:
            # in case there is only a single switch, the delegation switch is identified by label DS and
            # the extension switch is identified by label ES
            switch = ctx.topo.get_switch_by_label('DS')

            print("")
            print("dts solver initiated")
            print("  solver", self.ctx.config.get('param_dts_algo'))
            print("  objective", self.ctx.config.get('param_dts_objective'))
            print("  weight_table", self.ctx.config.get('param_dts_weight_table'))
            print("  weight_link", self.ctx.config.get('param_dts_weight_link'))
            print("  weight_ctrl", self.ctx.config.get('param_dts_weight_ctrl'))
            print("  number of flow rules", len(ctx.flows))
            print("  threshold", self.threshold)
            print("  look_ahead", self.look_ahead)
            print("  skip_ahead", self.skip_ahead)
            print("  ports", len(switch.ports))

            self.ports = {}
            for label, port in switch.ports.items():

                print("    ", port.id, port.link_out.label, port.link_in.label)
                if port.id == 0: 
                    # the port to the extension switch has to be excluded from the optimization problem
                    assert(port.link_out.label == 'DS->ES')
                    continue;

                self.ports[port.id] = VPort(port)
                self.ports[port.id].label = port.link_in.label.split('->')[0] # dirty, fix

                dss_ports[port.id] = ctx.topo.get_label_by_id(port.target)
                flows = [flow for flow in ctx.flows if flow.path_tuples[0][0] == port.target]
                for epoch_cnt in range(1, self.maxtick):
                    e = VEpoch(epoch_cnt-1, epoch_cnt)
                    e.active = [flow for flow in flows if flow.start < epoch_cnt and flow.start+flow.duration > epoch_cnt]
                    e.started = [flow for flow in flows if flow.start > epoch_cnt-1 and flow.start < epoch_cnt]
                    # TODO: <= epoch_cnt at the end for corner cases?
                    e.finished = [flow for flow in flows if flow.start+flow.duration > epoch_cnt-1 and flow.start+flow.duration < epoch_cnt]
                    e.startstop = [flow for flow in flows if flow.start > epoch_cnt-1 and flow.start+flow.duration < epoch_cnt]

                    e.diff = len(e.started) - len(e.finished)
                    self.ports[port.id].epochs.append(e)
                    self.ports[port.id].diffs.append(e.diff)



            # to make sure that everything is setup correctly, the following code gathers
            # all active flows at each time slot (as a baseline)
            tick = -1
            check_util = {}
            check = self.ctx.scenario.flows_of_switch.get(self.ctx.scenario_data.switch)
            while len(check) > 0 and tick < self.ctx.maxtick:
                tick += 1
                active = 0
                new_check = []
                for flow in check:
                    s = flow.get('start')
                    d = flow.get('duration')
                    if s+d < tick:
                        continue # skip this one, is done
                    if s < tick and s+d > tick:
                        active += 1
                    new_check.append(flow)
                check = new_check
                check_util[tick-1] = active # -1 because simulator stuff is shifted (relic from earlier versions)

            # then, compare this data to the epoch data structures used here and also the
            # data structures from scenario_data; they should match perfectly or something is
            # wrong
            for tick in range(0, self.ctx.maxtick-1):
                sum_tick = 0
                sum_tick_check = 0
                for _, port in self.ports.items():
                    check = len(self.ctx.scenario_data.flows_of_timeslot_per_port.get(port.label).get(tick, []))
                    #print(tick, port.label, len(port.epochs[tick].active)+len(port.epochs[tick].started), check)
                    
                    sum_tick += len(port.epochs[tick].active)
                    sum_tick_check += check
                assert(sum_tick == sum_tick_check == self.ctx.scenario_data.utils_raw[tick] == check_util.get(tick, 0))


        #self.plot_table()

        # initialize dss_result (is used later on for RSS)
        self.dss_result = VDSSResult(dss_ports, self.maxtick, init=True)


        # calculate mu
        if len(ctx.flows) > 0:
            self.mu = self.getMu()

    def log(self, *msg):
        if self.verbose:
            print(*msg)


    def simulate_delegation_demand2(self, port_id, delegated_at_array, restrict=None):
        # wrapper for verification
        data = self._simulate_delegation_demand2(port_id, delegated_at_array, restrict=restrict)
        port = self.ports.get(port_id)
        check = self.ctx.scenario_data.simulate_delegation_demand2(port, delegated_at_array, restrict=restrict)
        assert(numpy.isclose(sum(data), sum(check)))
        return data


    def _simulate_delegation_demand2(self, port_id, delegated_at_array, restrict):
        """
        This is almost identical to simulate_delegation_demand
        The only difference is that this function does not assume that the port
        is delegated for all time slots defined by restrict;
        Used by solver 1
        """
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
            now = self.ports[port_id].epochs[tick-1]

            if tick in delegated_at_array:
                if tick-1 in delegated_at_array:
                    # continue with an active flowset
                    flowset += now.started[:]
                else:
                    flowset = now.started[:]
                    demand = 0    
            else:
                # create a new flowset
                flowset = []
                demand = 0

            remove_from_flowset = []
            for flow in flowset:
                if flow.start+flow.duration > tick:
                    # flow is still active
                    d = min(tick-flow.start, 1)*flow.demand_per_tick
                    demand += d
                else:
                    # we also have to handle the case where a flow rule is removed
                    # (compare also the getDemand() functions, they have to deal with the same issue)
                    d = (flow.start + flow.duration-(tick-1))*flow.demand_per_tick
                    demand += d
                    remove_from_flowset.append(flow)

            for flow in remove_from_flowset:
                flowset.remove(flow)  

            if tick >= start-1 and tick < stop-1:
                demands.append(demand)
        assert(len(demands) == stop-start)
        return demands

    def simulate_delegation_demand(self, port_id, delegated_at_array, restrict):
        """
        Calculates the delegation demand based on a history array. Assumes that
        the port is delegated from start to stop;
        Used only for debugging the functionality of "calculating demand" 
        within solver 2
        """
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
            now = self.ports[port_id].epochs[tick-1]

            if tick in delegated_at_array:
                # continue with an active flowset
                flowset += now.started[:]
            else:
                # create a new flowset
                flowset = now.started[:]

            remove_from_flowset = []
            for flow in flowset:
                if flow.start+flow.duration > tick:
                    # flow is still active
                    d = min(tick-flow.start, 1)*flow.demand_per_tick
                    demand += d
                    #print(flow.start, "is active at", tick, d)
                else:
                    # we also have to handle the case where a flow rule is removed
                    # (compare also the getDemand() functions, they have to deal with the same issue)
                    d = (flow.start + flow.duration-(tick-1))*flow.demand_per_tick
                    #print(flow.start, "is removed", tick, flow.start + flow.duration, d)
                    demand += d
                    remove_from_flowset.append(flow)

            for flow in remove_from_flowset:
                flowset.remove(flow)  

            if tick >= start-1:
                demands.append(demand)
        assert(len(demands) == stop-start+1)
        return demands



    def simulate_delegations(self, port_id, delegated_at_array, restrict=None):
        # wrapper for verification
        data = self._simulate_delegations(port_id, delegated_at_array, restrict=restrict)
        port = self.ports.get(port_id)
        check = self.ctx.scenario_data.simulate_delegations(port, delegated_at_array, restrict=restrict)
        assert data == check
        return data

    def _simulate_delegations(self, port_id, delegated_at_array, restrict=None):
        # transform delegated_at_array; this is necessary because
        # the simulator tick is shifted by one (otherwise the values
        # would not fit)
        if len(delegated_at_array) > 0:
            delegated_at_array = [x+1 for x in delegated_at_array]
        old = None
        flowset = []
        utilization = {}

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


        for tick in delegated_at_array:

            now = self.ports[port_id].epochs[tick-1]

            if tick-1 == old:
                # continue with an active flowset
                flowset += now.started[:]
            else:
                # create a new flowset
                flowset = now.started[:]

            reduce_cnt = 0
            remove_from_flowset = []
            for flow in flowset:
                if flow.start+flow.duration > tick:
                    # flow is still active
                    reduce_cnt += 1
                else:
                    remove_from_flowset.append(flow)

            for flow in remove_from_flowset:
                flowset.remove(flow)    

            # the +1 resembles the overhead that we have by using the port for
            # delegation!
            utilization[tick] = len(now.active) - reduce_cnt + 1
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
                    result_vector.append(len(self.ports[port_id].epochs[tick-1].active))
        else:  
            for tick in range(1, self.maxtick):
                if tick in delegated_at_array:
                    result_vector.append(utilization.get(tick))
                else:
                    result_vector.append(len(self.ports[port_id].epochs[tick-1].active))


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
                        without_delegation = len(self.ports[port_id].epochs[t1-1].active)
                        with_delegation = utilization.get(t1)
                        assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                        backdelegations += (without_delegation + 1 - with_delegation)


                t = toconsider[-1]
                without_delegation = len(self.ports[port_id].epochs[t-1].active)
                with_delegation = utilization.get(t)
                assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                backdelegations += (without_delegation + 1 - with_delegation)

                # tuning for small number of selected ports; in these cases, the above calculation
                # will usually return a cost of 0. As a result, the solver tends to choose isolated 
                # ports (delegate in one time slot and remove delegation directly afterwards) which 
                # is not really necessary and reduces performance; the below adjustments increase the
                # cost by the number of currently delegated flows to account for that.
                #if len(toconsider) == 1:
                #    backdelegations += 3*utilization.get(t) # using slightly higher cost (3 instead of 2) is useful here (empirical)
                #if len(toconsider) == 2:
                #    if toconsider[0] != toconsider[1]-1:
                #        backdelegations += 2*utilization.get(toconsider[0])               


            # delegations are only required if there is overutilization. In the backdelegations case,
            # it is usually cheaper for the solver to keep a port delegated if that port
            # was already delegated before regardless of whether delegation at all is still
            # required or not. That isbecause the cost to remove the delegation is equal to
            # the amount of flows that were delegated in the recent past which is >0 in these cases.
            # To avoid such scenarios, we check whether there is still a overutilization
            # and artifically increase the cost for all solutions that try to delegate. Note that
            # we do NOT want to consider the history tick (start-1) here!

            #mu = [self.mu[tick] for tick in range(restrict[0], restrict[1]+1)]
            #if sum(mu) == 0:
            #if self.mu[restrict[0]] == 0:
            for tick in range(restrict[0], restrict[1]+1):
                if tick in delegated_at_array and self.mu[tick] == 0: 
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
                        without_delegation = len(self.ports[port_id].epochs[t1-1].active)
                        with_delegation = utilization.get(t1)
                        assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                        backdelegations += (without_delegation + 1 - with_delegation) 

                t = delegated_at_array[-1]
                without_delegation = len(self.ports[port_id].epochs[t-1].active)
                with_delegation = utilization.get(t)
                assert(without_delegation + 1 >= with_delegation) # +1 because of overhead 
                backdelegations += (without_delegation + 1 - with_delegation)    


        return result_vector, backdelegations

    def size(self, tick):
        # wrapper for verification
        data = self._size(tick)
        check = self.ctx.scenario_data.getSize(tick)
        assert data == check
        return data

    def _size(self, tick):
        return sum([len(port.epochs[tick].active) for p, port in self.ports.items()])


    def get_first_overutilization(self):
        # wrapper for verification
        data = self._get_first_overutilization()
        check = self.ctx.scenario_data.get_first_overutilization()
        assert data == check
        return data

    def _get_first_overutilization(self):
        B = {}
        first_time_over = 0
        if len(self.ctx.flows) > 0:
            for t in range(0, len(self.ports[1].epochs)-1):
                table_size = sum([len(port.epochs[t].active) for p, port in self.ports.items()])
                B[t] = table_size
                if table_size > self.threshold:
                    if first_time_over == 0:
                        first_time_over = t
        return first_time_over

    def get_last_overutilization(self):
        # wrapper for verification
        data = self._get_last_overutilization()
        check = self.ctx.scenario_data.get_last_overutilization()
        assert data == check
        return data

    def _get_last_overutilization(self):
        B = {}
        last = 0
        if len(self.ctx.flows) > 0:
            for t in range(0, len(self.ports[1].epochs)-1):
                table_size = sum([len(port.epochs[t].active) for p, port in self.ports.items()])
                B[t] = table_size
                if table_size > self.threshold:
                    last = t
        return last

    def plot_table(self):
        utilization = []
        for t in range(0, self.maxtick-1):
            u = 0
            for p, port in self.ports.items():
                u += len(port.epochs[t].active)
            utilization.append(u)
        import matplotlib.pyplot as plt           
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(numpy.arange(len(utilization)), utilization)
        plt.show()

    # ==========================
    # Tables for solver2
    # ==========================
    #
    # overview usage A1 etc
    #
    #    [ ][ ] A1 --> do nothing (stay undelegated) (x=1)
    #    [ ][x] E1 --> add delegation (x=0)
    #    [x][x] A2 --> do nothing (stay delegated) (y=1)
    #    [x][ ] E2 --> remove delegation (y=0)  

    def getB(self):
        B = {}
        for t in range(0, len(self.ports[1].epochs)-1):
            B[t] = sum([len(port.epochs[t].active) for p, port in self.ports.items()])
        return B

    def getMu(self):
        # wrapper for verification
        data = self._getMu()
        check = self.ctx.scenario_data.getMu()
        #print(data)
        #print(check)
        assert data == check
        return data
  
    def _getMu(self):
        B = self.getB()
        mu = []
        for t, util in B.items(): 
            if util > self.threshold:
                mu.append(1)
            else:
                mu.append(0)
        return mu

    def getDemandRaw(self, tick):
        # wrapper for verification
        data = self._getDemandRaw(tick)
        check = self.ctx.scenario_data.getDemandRaw(tick, self.ports)
        assert(numpy.isclose(sum(data), sum(check)))
        return data
  
    def _getDemandRaw(self, tick):
        """used by updateDSSResultWithoutDelegation"""
        D = {}
        for p, port in self.ports.items():
            demand = 0
            for flow in port.epochs[tick].active:
                demand += min(tick+1-flow.start, 1)*flow.demand_per_tick
            # we also have to consider finished flows here because they
            # also contribute to the demand in the last slot
            for flow in port.epochs[tick].finished:
                demand += ((flow.start + flow.duration) - (tick))*flow.demand_per_tick    
            D[p] = demand 
        return D

    def getDemand(self, tick):
        # wrapper for verification
        data = self._getDemand(tick)
        check = self.ctx.scenario_data.getDemand(tick, self.ports)
        assert(numpy.isclose(sum(data), sum(check)))
        return data

    def _getDemand(self, tick):
        """used for dss_result.update_demand_per_port()"""
        D = {}
        for p, port in self.ports.items():
            demand = 0
            if port.is_delegated:
                pre_existing_set = port.epochs[port.delegated_at-1].active
                demand = 0
                for flow in port.epochs[tick].active:
                    # we are interested in all delegated flows here
                    if flow not in pre_existing_set: 
                        demand += min(tick+1-flow.start, 1)*flow.demand_per_tick
                # we also have to consider finished flows here because they
                # also contribute to the demand in the last slot
                for flow in port.epochs[tick].finished:
                    if flow not in pre_existing_set: 
                        demand += ((flow.start + flow.duration) - (tick))*flow.demand_per_tick
                D[p] = demand
            else:
                for flow in port.epochs[tick].active:
                    demand += min(tick+1-flow.start, 1)*flow.demand_per_tick
                # we also have to consider finished flows here because they
                # also contribute to the demand in the last slot
                for flow in port.epochs[tick].finished:
                    demand += ((flow.start + flow.duration) - (tick))*flow.demand_per_tick    
                D[p] = demand 
        return D

    def getDemand1(self, tick):
        # wrapper for verification
        data = self._getDemand1(tick)
        check = self.ctx.scenario_data.getDemand1(tick, self.ports)
        assert(numpy.isclose(sum(data), sum(check)))
        return data

    def _getDemand1(self, tick):
        D = {}
        for p, port in self.ports.items():
            D[p] = {}
            if port.is_delegated:
                D[p] = {}
                pre_existing_set = port.epochs[port.delegated_at-1].active
                for d in range(0, self.look_ahead):
                    demand = 0
                    for flow in port.epochs[tick+d-1].active:
                        # we are interested in all delegated flows here
                        if flow not in pre_existing_set: 
                            demand += min(tick+d-flow.start, 1)*flow.demand_per_tick

                    # we also have to consider finished flows here because they
                    # also contribute to the demand in the last slot
                    for flow in port.epochs[tick+d-1].finished:
                        if flow not in pre_existing_set: 
                            demand += ((flow.start + flow.duration) - (tick+d-1))*flow.demand_per_tick

                    D[p][d] = demand
            else:
                for d in range(0, self.look_ahead):
                    D[p][d] = 0   
        return D

    def getDemand2(self, tick):
        # wrapper for verification
        data = self._getDemand2(tick)
        check = self.ctx.scenario_data.getDemand2(tick, self.ports)
        assert(numpy.isclose(sum(data), sum(check)))
        return data

    def getDemand2(self, tick):
        D = {}
        for p, port in self.ports.items():
            D[p] = {}
            if not port.is_delegated:
                D[p] = {}
                pre_existing_set = port.epochs[tick-1].active
                for d in range(0, self.look_ahead):
                    demand = 0
                    for flow in port.epochs[tick+d].active:
                        # we are interested in all delegated flows here
                        if flow not in pre_existing_set: 
                            demand += min(tick+d+1-flow.start, 1)*flow.demand_per_tick
                    # we also have to consider finished flows here because they
                    # also contribute to the demand in the last slot
                    for flow in port.epochs[tick+d].finished:
                        if flow not in pre_existing_set: 
                            demand += ((flow.start + flow.duration) - (tick+d))*flow.demand_per_tick      
                    D[p][d] = demand  
        return D       

    def getA(self, tick):
        # wrapper for verification
        data = self._getA(tick)
        check = self.ctx.scenario_data.getA(tick, self.ports)
        assert data == check
        return data
    
    def _getA(self, tick):
        A = {}
        for p, port in self.ports.items():
            if not port.is_delegated:
                A[p] = {}
                for d in range(0, self.look_ahead):
                    A[p][d] = len(port.epochs[tick+d].active)
        return A

    def getA2(self, tick):
        # wrapper for verification
        data = self._getA2(tick)
        check = self.ctx.scenario_data.getA2(tick, self.ports)
        """
        print("data:")
        for p, slots in data.items():
            if sum(slots.values()) > 0:
                print(p, slots)
        print("check:")
        for p, slots in check.items():
            if sum(slots.values()) > 0:
                print(p, slots) 
        """  
        assert data == check
        return data

    def _getA2(self, tick):
        A2 = {}
        
        for p, port in self.ports.items():
            if port.is_delegated:
                A2[p] = {}
                flows = port.epochs[port.delegated_at-1].active
                for d in range(0, self.look_ahead):
                    cnt = 0
                    # note that epochs is shifted already by 1 and we need 
                    # the next tick (a little non-intuitive); thats why +2 is used
                    for flow in port.epochs[tick+d].active:
                       if flow in flows: cnt += 1;
                    A2[p][d] = cnt

        self.counters_evicted2[tick] = 0
        for p, port in self.ports.items():
            if port.is_delegated:
                if p == 6:
                    cnt = 0
                    self.counters_evicted2[tick] = A2[6][0]
                # note that epochs is shifted already by 1 and we need 
                # the next tick (a little non-intuitive); thats why +2 is used
                #for flow in port.epochs[tick-1].active:
                #    if flow in port.old_flows: 
                #        pass
                #    else: self.counters_evicted2[tick] += 1       

        # debugging
        self.counters_evicted[tick] = 0
        for p, port in self.ports.items():
            if port.is_delegated:  
                for i in range(port.delegated_at, tick):
                    self.counters_evicted[tick] += len(port.epochs[i].started);  
                #for flow in port.epochs[tick+1].active:
                #    if flow not in port.old_flows: 
                #        self.counters_evicted[tick] += 1;          
        return A2

    def getE(self, tick):
        # wrapper for verification
        data = self._getE(tick)
        check = self.ctx.scenario_data.getE(tick, self.ports)
        assert data == check
        return data

    def _getE(self, tick):
        # flow table development if a port is delegated
        E = {}
        for p, port in self.ports.items():
            if not port.is_delegated:
                E[p] = {}
                flows = port.epochs[tick-1].active
                for d in range(0, self.look_ahead):
                    cnt = 0
                    # we only consider flows here that are "not new", i.e., flows that were
                    # already present when the delegation started; all the other new flows will
                    # be handled by the delegation switch
                    for flow in port.epochs[tick+d].active:
                        if flow in flows: cnt += 1;
                    E[p][d] = cnt 
        return E

    def getE2(self, tick):
        # wrapper for verification
        data = self._getE2(tick)
        check = self.ctx.scenario_data.getE2(tick, self.ports)
        assert data == check
        return data

    def _getE2(self, tick):
        E2 = {}
        for p, port in self.ports.items():
            if port.is_delegated:
                E2[p] = {}
                for d in range(0, self.look_ahead):
                    E2[p][d] = len(port.epochs[tick+d].active) 
        return E2

    def getV(self):
        V = {}
        for p, port in self.ports.items():
            if port.is_delegated:
                V[p] = 0
            else:
                V[p] = 1
        return V       

    def updateDSSResultWithoutDelegation(self):
        """In case no delegation is required, this function will fill the dssResult object with raw data"""
        logger.info("prepare raw dssResult set")
        for t in range(0, self.maxtick-self.look_ahead-1):
            A1 = self.getA(t)
            DRaw = self.getDemandRaw(t)
            ASum = 0
            for p, port in self.ports.items():
                self.dss_result.update(t, p, 0) # nothing is delegated
                self.dss_result.update_util_per_port(t, p, A1[p][0], A1[p][0]) 
                self.dss_result.update_demand_per_port(t, p, DRaw[p], DRaw[p])
                ASum += A1[p][0]
            self.dss_result.update_util(t, ASum)    

    def statistics_append(self, name, value):
        """
        Helper function or statistics.
        """
        if not self.ctx.statistics.get(name):
            self.ctx.statistics[name] = []
        self.ctx.statistics[name].append(value)

    def statistics_model(self, m):
        # add some statistics for a solved model
        self.statistics_append('solver.stats_time_solving', m.Runtime)
        self.statistics_append('solver.stats_infeasible', m.status == 3)
        self.statistics_append('solver.stats_timelimit', m.status == 9)
        self.statistics_append('solver.stats_itercount', m.Itercount)
        self.statistics_append('solver.stats_nodecount', m.Nodecount)
        self.log("  solved model status=%d runtime=%.5f Itercount=%d Baritercount=%d Nodecount=%d" % (
            m.status, m.Runtime, m.Itercount, m.Baritercount, m.Nodecount))
        # parse results from printStats (there seem to be no other way to get this...)
        stats = io.StringIO()
        with redirect_stdout(stats):
            m.printStats()
        data = stats.getvalue().split()
        self.ctx.statistics['solver.stats_linear_constraint_matrix'] = int(data[9])
        self.ctx.statistics['solver.stats_constraints'] = int(data[11])
        self.ctx.statistics['solver.stats_nz_variables'] = int(data[13])
        if m.status == 3: self.ctx.statistics['solver.cnt_feasable'] += 1
        if m.status == 9: self.ctx.statistics['solver.cnt_timelimit'] += 1

    def print(self, table_name, table):
        print("%s" % (table_name))
        for p in table:
            print(p, "--", ''.join(['{:5}'.format(table[p][x]) for x in range(0, self.look_ahead)]))




    # ==========================
    # SOLVER 1
    # ==========================

    def solver1_wrapper(self):
        """
        Main wrapper for executing solver type 1 (non-iterative)
        """
        tick = self.get_first_overutilization()-(self.look_ahead*2)
        tick_last = self.get_last_overutilization()+(self.look_ahead*2)

        if self.get_first_overutilization() == 0 and self.get_last_overutilization() == 0:
            print("  skip solving, no overhead situation")
            self.ctx.statistics['solver.nothing_to_do'] = 1 
            self.commands = {}
            return

        history = {}
        for p, port in self.ports.items(): 
            history[p] = []   

        for t in range(tick, tick_last, self.skip_ahead):

            # start by skipping all cases where 
            """
            mu = self.getMu()
            allv_neg = sum([1-v for _, v in self.getV().items()])
            allmu_lookahead = sum(mu[t2] for t2 in range(t, t+self.look_ahead))
            if allv_neg + allmu_lookahead == 0:
                #print("--- Results for start=%d, look_ahead=%d : skipped, no overutil and no active delegations"  % (start, self.look_ahead))  
                continue
            """

            # modeling part
            stime = time.time()

            objective = self.ctx.config.get('param_dts_objective')
            m, C, combinations = (None, None, None)
            if objective == 1: 
                m, C, combinations = self.solver1_01_underutil(t, history=history)
            if objective == 2: 
                m, C, combinations = self.solver1_02_demand(t, history=history)
            if objective == 3: 
                m, C, combinations = self.solver1_03_overhead(t, history=history)
            if objective == 4: 
                m, C, combinations = self.solver1_04_backdelegations(t, history=history)
            if objective == 5: 
                m, C, combinations = self.solver1_05_multi_objective(t, history=history)

            if not m:
                raise RuntimeError('no objective defined for param_dts_objective==%d' % objective)

            self.statistics_append('solver.stats_time_modeling', time.time()-stime)
            self.log("  %d" % t, "build model in %.3f" % (time.time()-stime))

            # solving part
            opt = self.solver1_wrapper_runmodel(m, C, combinations)
            if opt:
                # extend history
                for p, port in self.ports.items(): 
                    for s in range(0, self.skip_ahead):
                        if opt.get(s).get(p) == 1:
                            if t+s not in history[p]:
                                history[p].append(t+s)

        # create commands for simulator based on history array
        for t in range(1, self.maxtick-1):
            self.commands[t] = dict(remove=[], add=[])          
        for p, port in self.ports.items(): 
            for t in range(1, self.maxtick):
                if t in history[p]:
                    self.dss_result.update(t, p, 1)
                    if not t-1 in history[p]:
                        self.commands[t]['add'].append((p, 'ES'))
                else:
                    self.dss_result.update(t, p, 0)
                    if t-1 in history[p]:
                        self.commands[t]['remove'].append((p, 'ES'))


        #self.dss_result.update_demand_per_port(start, p, DNew[p], DRaw[p])

        # verify with simulator
        arr = {}
        for p, port in self.ports.items(): 
            arr[p] = self.simulate_delegations(p, history[p])[0]
            
        for t in range(0, self.maxtick-2):
            summed = 0
            for p, port in self.ports.items(): 
                summed += arr[p][t]  
                self.dss_result.update_util_per_port(t, p, arr[p][t], len(port.epochs[t].active))

            self.counters_util[t] = summed 
            self.dss_result.update_util(t, summed)               

    def solver1_wrapper_runmodel(self, m, C, combinations):
        logging.disable(logging.INFO);
        
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 200.0)

        m.optimize()
        logging.disable(logging.NOTSET);

        self.statistics_model(m)

        if m.status == 3 or m.status == 9:
            # infeasable or timelimit
            print("  model was infeasable")
            self.ctx.statistics['solver.cnt_infeasable'] += 1
            return None
        else:
            self.ctx.statistics['solver.cnt_feasable'] += 1   

        SOpt = {}
        VAL = {}
        for p, port in self.ports.items():
            SOpt[p] = {}
            VAL[p] = {}

        # get results from the optimization
        for v in m.getVars():
            if v.varname[0] == 's':
                p = int(v.varname.split('_')[1])
                s = int(v.varname.split('_')[2])
                if v.x > 0.5:
                    #print(p, combinations[s])
                    for t, x in enumerate(combinations[s]):
                        SOpt[p][t] = x
                    for t, x in enumerate(C[p][s]):
                        VAL[p][t] = x

        result = {}
        for t in range(0, self.skip_ahead):
            result[t] = {} 
            for p, port in self.ports.items():
                result[t][p] = SOpt[p][t] 

        return result

    def solver1_05_multi_objective(self, tick, history=None):
        """
        Multi-objective formulation with three cost coefficients
        """
        m = Model("solver1_05_multi_objective")
        
        combinations = list(itertools.product([0, 1], repeat=self.look_ahead))

        S = {}
        C = {}
        P_CTRL = {}
        P_TABLE = {}
        P_LINK = {} # cost coefficients for link overhead
        for p, port in self.ports.items():
            C[p] = {}
            S[p] = {}
            P_CTRL[p] = {}
            P_TABLE[p] = {}
            P_LINK[p] = {}
            sum_subset = []
            for s, subset in enumerate(combinations):
                
                # create a binary variable for this subset
                S[p][s] = m.addVar(vtype=GRB.BINARY, name='s_%d_%d' % (p, s))
                sum_subset.append(S[p][s])

                # subset example --> [0,0,1,1,1,0,0,1,1]
                tick_arr = []
                if history:
                    tick_arr = history.get(p)[:]
                for i, c in enumerate(subset):
                    if c == 1: tick_arr.append(tick+i)

                coeff, backdelegations = self.simulate_delegations(p, tick_arr, restrict=(tick, tick+self.look_ahead))
                C[p][s] = coeff

                # rule overhead coefficients
                P_TABLE[p][s] = len(subset)-sum(subset)

                # control overhead coefficients
                P_CTRL[p][s] = backdelegations

                # link overhead coefficients
                profit = self.simulate_delegation_demand2(p, tick_arr, restrict=(tick+1, tick+self.look_ahead+1))
                P_LINK[p][s] = sum(profit)

            m.addConstr(quicksum(sum_subset) == 1)
            #print("port=%d, subset=%s, result=%s, coeff=%s" % (p, str(subset), str(tick_arr), str(coeff)))
       
        for t in range(0, self.look_ahead):
            sum_ports = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(C[p][s][t] * S[p][s])
                sum_ports.append(quicksum(sum_subset))

            m.addConstr(quicksum(sum_ports) <= self.threshold)

        # parameters for setObjectiveN
        # see https://www.gurobi.com/documentation/9.0/refman/py_model_setobjectiven.html
        obj_index = 1 # different per objective and >0
        obj_priority = 0 # all parts have the same priority

        w_table = self.ctx.config.get('param_dts_weight_table')
        w_link = self.ctx.config.get('param_dts_weight_link')
        w_ctrl = self.ctx.config.get('param_dts_weight_ctrl')

        # rule overhead
        if w_table > 0:
            obj_index += 1 # increase index if a new objective is added
            rule_overhead = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(P_TABLE[p][s] * S[p][s]) 
                rule_overhead.append(quicksum(sum_subset))  
            m.setObjectiveN(-quicksum(rule_overhead), obj_index, obj_priority, w_table)        

        # link overhead
        if w_link > 0:
            obj_index += 1 # increase index if a new objective is added
            link_overhead = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(P_LINK[p][s] * S[p][s]) 
                link_overhead.append(quicksum(sum_subset))  
            m.setObjectiveN(quicksum(link_overhead), obj_index, obj_priority, w_link)

        # ctrl overhead
        if w_ctrl > 0:
            obj_index += 1 # increase index if a new objective is added
            ctrl_overhead = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(P_CTRL[p][s] * S[p][s]) 
                ctrl_overhead.append(quicksum(sum_subset))  
            m.setObjectiveN(quicksum(ctrl_overhead), obj_index, obj_priority, w_ctrl)
            #m.setObjective(quicksum(ctrl_overhead), GRB.MINIMIZE)

        #m.update()
        #print(dir(m.getObjective()))
        #print(m.getObjective(2))

        return m, C, combinations

    def solver1_04_backdelegations(self, tick, history=None):
        """
        Alternative problem formulation based on the multiple-choice knapsack problem;
        WIP (currently no iteration)
        """
        m = Model("solver1_04_backdelegations")
        
        combinations = list(itertools.product([0, 1], repeat=self.look_ahead))

        """
        filtered = []
        for c in combinations:
            if sum(c) < 2: continue;
            filtered.append(c)
        filtered = combinations
        """

        S = {}
        C = {}
        P = {}
        P2 = {}
        for p, port in self.ports.items():
            C[p] = {}
            S[p] = {}
            P[p] = {}
            P2[p] = {}
            sum_subset = []
            for s, subset in enumerate(combinations):
                
                # create a binary variable for this subset
                S[p][s] = m.addVar(vtype=GRB.BINARY, name='s_%d_%d' % (p, s))
                sum_subset.append(S[p][s])

                # subset example --> [0,0,1,1,1,0,0,1,1]
                tick_arr = []
                if history:
                    tick_arr = history.get(p)[:]
                for i, c in enumerate(subset):
                    if c == 1: tick_arr.append(tick+i)

                coeff, backdelegations = self.simulate_delegations(p, tick_arr, restrict=(tick, tick+self.look_ahead))
                C[p][s] = coeff

                P2[p][s] = len(subset)-sum(subset)

                # create the profit for this port/subset
                P[p][s] = backdelegations


            m.addConstr(quicksum(sum_subset) == 1)
            #print("port=%d, subset=%s, result=%s, coeff=%s" % (p, str(subset), str(tick_arr), str(coeff)))
       
        for t in range(0, self.look_ahead):
            sum_ports = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(C[p][s][t] * S[p][s])
                sum_ports.append(quicksum(sum_subset))

            m.addConstr(quicksum(sum_ports) <= self.threshold)

        """
        objective_underutil = []
        for t in range(0, self.look_ahead):
            sum_ports = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(C[p][s][t] * S[p][s])
                sum_ports.append(quicksum(sum_subset))
            objective_underutil.append(self.threshold - quicksum(sum_ports))
        #m.setObjective(quicksum(objective_underutil), GRB.MINIMIZE)
        m.setObjectiveN(quicksum(objective_underutil), 1, 2)
        
        objective_overhead = []
        for p, port in self.ports.items():
            sum_subset = []
            for s, subset in enumerate(combinations):
                sum_subset.append(P2[p][s] * S[p][s]) 
            objective_overhead.append(quicksum(sum_subset))  
        m.setObjectiveN(-quicksum(objective_overhead), 1, 2)
        """

        objective_backdelegations = []
        for p, port in self.ports.items():
            sum_subset = []
            for s, subset in enumerate(combinations):
                sum_subset.append(P[p][s] * S[p][s]) 
            objective_backdelegations.append(quicksum(sum_subset))  
        m.setObjective(quicksum(objective_backdelegations), GRB.MINIMIZE)
        #m.setObjectiveN(quicksum(objective_backdelegations), 0, 3)




        return m, C, combinations

    def solver1_03_overhead(self, tick, history=None):
        """
        Alternative problem formulation based on the multiple-choice knapsack problem;
        WIP (currently no iteration)
        """
        m = Model("solver1_03_overhead")
        
        combinations = list(itertools.product([0, 1], repeat=self.look_ahead))

        S = {}
        C = {}
        P = {}
        for p, port in self.ports.items():
            C[p] = {}
            S[p] = {}
            P[p] = {}
            sum_subset = []
            for s, subset in enumerate(combinations):
                # create a binary variable for this subset
                S[p][s] = m.addVar(vtype=GRB.BINARY, name='s_%d_%d' % (p, s))
                sum_subset.append(S[p][s])

                # create the profit for this port/subset
                P[p][s] = sum(subset)-len(subset)

                # subset example --> [0,0,1,1,1,0,0,1,1]
                tick_arr = []
                if history:
                    tick_arr = history.get(p)[:]
                for i, c in enumerate(subset):
                    if c == 1: tick_arr.append(tick+i)

                coeff, backdelegations = self.simulate_delegations(p, tick_arr, restrict=(tick, tick+self.look_ahead))
                C[p][s] = coeff
            m.addConstr(quicksum(sum_subset) == 1)
            #print("port=%d, subset=%s, result=%s, coeff=%s" % (p, str(subset), str(tick_arr), str(coeff)))
       
        for t in range(0, self.look_ahead):
            sum_ports = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(C[p][s][t] * S[p][s])
                sum_ports.append(quicksum(sum_subset))

            m.addConstr(quicksum(sum_ports) <= self.threshold)

        sum_ports = []
        for p, port in self.ports.items():
            sum_subset = []
            for s, subset in enumerate(combinations):
                sum_subset.append(P[p][s] * S[p][s]) 
            sum_ports.append(quicksum(sum_subset))  
        m.setObjective(quicksum(sum_ports), GRB.MINIMIZE)

        return m, C, combinations

    def solver1_02_demand(self, tick, history=None):
        """
        Minimize delegated demand
        """
        m = Model("solver1_02_demand")
        
        combinations = list(itertools.product([0, 1], repeat=self.look_ahead))

        S = {}
        C = {}
        P = {}
        P2 = {}
        for p, port in self.ports.items():
            C[p] = {}
            S[p] = {}
            P[p] = {}
            P2[p] = {}
            sum_subset = []
            for s, subset in enumerate(combinations):
                
                # create a binary variable for this subset
                S[p][s] = m.addVar(vtype=GRB.BINARY, name='s_%d_%d' % (p, s))
                sum_subset.append(S[p][s])

                # subset example --> [0,0,1,1,1,0,0,1,1]
                tick_arr = []
                if history:
                    tick_arr = history.get(p)[:]
                for i, c in enumerate(subset):
                    if c == 1: tick_arr.append(tick+i)

                coeff, backdelegations = self.simulate_delegations(p, tick_arr, restrict=(tick, tick+self.look_ahead))
                C[p][s] = coeff

                # create the profit for this port/subset
                profit = self.simulate_delegation_demand2(p, tick_arr, restrict=(tick+1, tick+self.look_ahead+1))
                P[p][s] = sum(profit)


            m.addConstr(quicksum(sum_subset) == 1)
            #print("port=%d, subset=%s, result=%s, coeff=%s" % (p, str(subset), str(tick_arr), str(coeff)))
       
        for t in range(0, self.look_ahead):
            sum_ports = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(C[p][s][t] * S[p][s])
                sum_ports.append(quicksum(sum_subset))

            m.addConstr(quicksum(sum_ports) <= self.threshold)

        objective_demand = []
        for p, port in self.ports.items():
            sum_subset = []
            for s, subset in enumerate(combinations):
                sum_subset.append(P[p][s] * S[p][s]) 
            objective_demand.append(quicksum(sum_subset))  
        m.setObjective(quicksum(objective_demand), GRB.MINIMIZE)
        #m.setObjectiveN(quicksum(objective_backdelegations), 0, 3)
        return m, C, combinations

    def solver1_01_underutil(self, tick, history=None):
        """
        Optimize underutilization
        """
        m = Model("solver1_01_underutil")
        
        combinations = list(itertools.product([0, 1], repeat=self.look_ahead))

        S = {}
        C = {}
        P = {}
        for p, port in self.ports.items():
            C[p] = {}
            S[p] = {}
            P[p] = {}
            sum_subset = []
            for s, subset in enumerate(combinations):
                # create a binary variable for this subset
                S[p][s] = m.addVar(vtype=GRB.BINARY, name='s_%d_%d' % (p, s))
                sum_subset.append(S[p][s])

                # create the profit for this port/subset
                P[p][s] = sum(subset)-len(subset)

                # subset example --> [0,0,1,1,1,0,0,1,1]
                tick_arr = []
                if history:
                    tick_arr = history.get(p)[:]
                for i, c in enumerate(subset):
                    if c == 1: tick_arr.append(tick+i)

                coeff, backdelegations = self.simulate_delegations(p, tick_arr, restrict=(tick, tick+self.look_ahead))
                C[p][s] = coeff
            m.addConstr(quicksum(sum_subset) == 1)
            #print("port=%d, subset=%s, result=%s, coeff=%s" % (p, str(subset), str(tick_arr), str(coeff)))
       
        for t in range(0, self.look_ahead):
            sum_ports = []
            for p, port in self.ports.items():
                sum_subset = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(C[p][s][t] * S[p][s])
                sum_ports.append(quicksum(sum_subset))

            m.addConstr(quicksum(sum_ports) <= self.threshold)

        sum_all = []
        for t in range(0, self.look_ahead):
            sum_ports = []
            sum_sidecase = []
            for p, port in self.ports.items():
                sum_subset = []
                sum_2 = []
                for s, subset in enumerate(combinations):
                    sum_subset.append(C[p][s][t] * S[p][s])
                    sum_2.append(P[p][s] * S[p][s])
                sum_ports.append(quicksum(sum_subset))
                sum_sidecase.append(quicksum(sum_2))


            sidecase = (1-self.mu[tick+t])*quicksum(sum_sidecase)

            sum_all.append((self.mu[tick+t] * (self.threshold - quicksum(sum_ports))) + sidecase)
        m.setObjective(quicksum(sum_all), GRB.MINIMIZE)

        return m, C, combinations

    # ==========================
    # SOLVER 2
    # ==========================

    def solver2_wrapper(self):

        # this function adds raw data to dss results, i.e., the demand_per_port values; this is
        # required because the algorithm below is not necessarily executed (i.e., in case there
        # is no bottleneck) but RSS still needs the data
        self.updateDSSResultWithoutDelegation()

        tick = self.get_first_overutilization()-(self.look_ahead*2)
        tick_last = self.get_last_overutilization()+(self.look_ahead*2)

        if self.get_first_overutilization() == 0 and self.get_last_overutilization() == 0:
            logger.info("skip solving, no overhead situation")
            self.ctx.statistics['solver.nothing_to_do'] = 1 
            self.commands = {}
            
            return

        # run iterative solver
        for start in range(tick, tick_last):

            # init the empty command array for this tick
            self.commands[start] = dict(remove=[], add=[])

            # call the wrapper function for a single iteration (tick=start)
            result = self.solver2_wrapper_runmodel(start)
            if result:
                XOpt = result.get('XOpt')
                YOpt = result.get('YOpt')

                # add new delegations
                for p, v in XOpt.items():
                    port = self.ports[p]
                    if v == 0:
                        if port.is_delegated:
                            raise RuntimeError('port selected by XOpt is already delegated')
                        else:
                            self.dss_result.update(start, p, 1)
                            self.commands[start]['add'].append((p, 'ES'));
                            port.is_delegated = True # mark port as delegated
                            port.delegated_at = start
                            port.old_flows = []
                            epoch = port.epochs[start] # get the relevant epoch
                            # store all flows that are NOT actually delegated right now, i.e.,
                            # the flows that arrived prior to the delegation
                            for flow in epoch.active:
                                #if not flow in epoch.started:
                                port.old_flows.append(flow)
                    else:
                        self.dss_result.update(start, p, 0)   

                # remove delegations
                for p, v in YOpt.items():
                    port = self.ports[p]
                    if v == 0:
                        if not port.is_delegated:
                            raise RuntimeError('port selected by YOpt is not delegated')
                        else:
                            self.dss_result.update(start, p, 0)
                            self.commands[start]['remove'].append((p, 'ES'));
                            port.is_delegated = False
                            port.old_flows = []  
                    else:
                        self.dss_result.update(start, p, 1)  

                # update dss_result
                DNew = self.getDemand(start)
                DRaw = self.getDemandRaw(start)
                for p, port in self.ports.items():
                    self.dss_result.update_demand_per_port(start, p, DNew[p], DRaw[p])
 
    def solver2_wrapper_runmodel(self, start):

        # start by skipping all cases where 
        mu = self.getMu()
        V = self.getV()
        allv_neg = sum([1-v for _, v in V.items()])
        allmu_lookahead = sum(mu[t] for t in range(start, start+self.look_ahead))
        if allv_neg + allmu_lookahead == 0:
            #print("--- Results for start=%d, look_ahead=%d : skipped, no overutil and no active delegations"  % (start, self.look_ahead)) 
            #return None
            pass

        # prepare the inputs for the optimization problem
        stime = time.time()
        A1 = self.getA(start)
        E1 = self.getE(start)
        A2 = self.getA2(start)
        E2 = self.getE2(start)
        D1 = self.getDemand1(start)
        D2 = self.getDemand2(start)
        DRaw = self.getDemandRaw(start)

        self.statistics_append('solver.stats_time_modeling', time.time()-stime)

        #print("----- tick=%d" % start)
        #self.print('Table A1:', A1)
        #self.print('Table E1:', E1)
        #self.print('Table A2:', A2)
        #self.print('Table E2:', E2)

        # run optimization using the objective specified by the experiment
        objective = self.ctx.config.get('param_dts_objective')
        m = None
        if objective == 1: m = self.solver2_01_underutil(start, V, A1, E1, A2, E2, mu);
        if objective == 11: m = self.solver2_011_underutil_relaxed(start, V, A1, E1, A2, E2, mu);
        if objective == 2: m = self.solver2_02_demand(start, V, A1, E1, A2, E2, D1, D2, mu);
        if objective == 3: m = self.solver2_03_overhead(start, V, A1, E1, A2, E2, mu);
        if objective == 4: m = self.solver2_04_backdelegation(start, V, A1, E1, A2, E2, mu);
        if objective == 5: m = self.solver2_05_multi_objective(start, V, A1, E1, A2, E2, D1, D2, mu);

        if not m:
            raise RuntimeError('no objective defined for param_dts_objective==%d' % objective)

        logging.disable(logging.INFO);
        m.update() 
        #m.Params.Presolve = 0
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 200.0)
        status = m.optimize()
        logging.disable(logging.NOTSET);

        self.statistics_model(m)

        # now analyze the results
        XOpt = {}
        YOpt = {}
        delegated = []
        removed = []
        overutil = 0

        # 3 = model is infeasable
        if m.status == 3 or m.status == 9:
            # in case the model is not feasable we select/unselect delegation relationships
            # based on a simple greedy algorithm; the idea here is that the optimized algorithm
            # may step in later if the utilization has balanced out; note that the success/failure
            # of the optimized algorithm can be tracked via the cnt_infeasable variable
            self.ctx.statistics['solver.cnt_infeasable'] += 1

            # we have to initialize XOpt and YOpt because this data is used later on
            # to update dss_result; YOpt is only defined if the port is currently delegated
            # (because it models the "remove delegation" decision) while XOpt is only defined
            # if the port is not delegated
            for p, v in V.items():
                if v == 0:
                    # port is delegated
                    YOpt[p] = 1 
                if v == 1:
                    # port is not delegated
                    XOpt[p] = 1

            print("--- Results for start=%d, look_ahead=%d : model is infeasable; fallback to greedy algorithm ---"  % (start, self.look_ahead)) 
            overutil = sys.maxsize 
            target = self.size(start) - self.threshold
            used = 0
            if target > 0:
                for p, v in V.items():  
                    # abort if utilization is below threshold
                    if  used > target:
                        break;
                    if v == 1:
                        # currently not delegatged
                        save = A1[p][0] - E1[p][0]
                        if save > 0:
                            used += save
                            XOpt[p] = 0 # also update XOpt!
                            delegated.append(p)
            if target < 0:
                for p, v in V.items(): 
                    # abort if utilization is below threshold
                    if used > -target:
                        break;
                    if v == 0:
                        # currently not delegatged
                        free =  E2[p][0] - A2[p][0]
                        if free > 0:
                            used += free
                            YOpt[p] = 0 # also update YOpt!
                            removed.append(p)   
            print("    -> (greedy) target=%d used=%d" % (target, used), "delegated", delegated, "removed", removed)

        else:
            self.ctx.statistics['solver.cnt_feasable'] += 1
            for v in m.getVars():
                #print(v.varName, v.x)
                if v.varname[0] == 'x':
                    p = int(v.varname.split('_')[1])
                    XOpt[p] = int(v.x)
                    if XOpt[p] == 0:
                        delegated.append(p)
                if v.varname[0] == 'y':
                    p = int(v.varname.split('_')[1])
                    YOpt[p] = int(v.x)
                    if YOpt[p] == 0:
                        removed.append(p)


            # (debugging) calculate delegated demand            
            total_demand = 0
            for p, port in self.ports.items():
                #if p != 15: continue;
                if port.is_delegated:
                    self.counters_demand += D1[p][0]
                    # uncomment the code block below to verify that
                    # D1 actually contains the correct values
                    """
                    history = []
                    last = start-1
                    while last > 0: 
                        decision = self.counters_decisions.get(last)
                        if decision:
                            print(" has history", last, decision.get(p))
                            if decision.get(p) == 1:
                                history.insert(0, last)
                            else:
                                last = -1
                                break;
                        last -= 1
                    history.append(start)
                    verify_demand = self.simulate_delegation_demand(p, history, restrict=(start+1, start+1))
                    print("check", history,  verify_demand)
                    assert(math.isclose(D1[p][0],verify_demand[0]))
                    """
                else:
                    pass

            # defer delegations? --> not used atm
            if False:
                amount_used = 0
                print("*"*20, start, "*"*20)
                for p, port in self.ports.items():
   
                    # v==1 --> p is not delegated
                    if V[p] == 1 and XOpt[p] == 0:
                        print("--- delay?", p)      
                        delegated_at = []
                        for d in range(0, self.look_ahead+1):
                            delegated_at.append(start+d)

                        
                        test_e1 = []
                        for d in range(0, self.look_ahead):
                            test_e1.append(E1[p][d])
                        print("E1", test_e1)
                        # just to be sure that nothing is fancy, we first recalculate E1
                        # with the simulate_delegations() function. Because simulate_delegations()
                        # includes the overhead already, the values in verifyE1 should be exactly those
                        # of E1 only with +1 each.
                        verifyE1, _ = self.simulate_delegations(p, delegated_at, restrict=(start+1, start+self.look_ahead))
                        print("verifyE1", verifyE1)

                        for t1, t2 in zip(test_e1, verifyE1):
                            assert(t1+1 == t2)

                        allx_neg = sum([1-x for _, x in XOpt.items()])
                        ally_neg = sum([1-y for _, y in YOpt.items()])   
                        overhead_new = allv_neg + allx_neg - ally_neg - 1

                        delayed, _ = self.simulate_delegations(p, delegated_at[1:], restrict=(start+1, start+self.look_ahead))
                        utils_with_delayed_decision = [] 
                        utils_old = []     
                        for d in range(0, self.look_ahead):
                            add1 = sum([V[p2] * XOpt[p2] * A1[p2][d] for p2 in A1])
                            add2_old = sum([V[p] * (1-XOpt[p]) * E1[p][d] for p in E1])
                            add2 = 0
                            for p2 in E1:
                                if p2 == p:
                                    add2 += V[p2] * (1-XOpt[p2]) * delayed[d]    
                                else:
                                    add2 += V[p2] * (1-XOpt[p2]) * E1[p2][d]
                            rem1 = sum([(1-V[p2]) * YOpt[p2] * A2[p2][d] for p2 in A2])
                            rem2 = sum([(1-V[p2]) * (1-YOpt[p2]) * E2[p2][d] for p2 in E2])
                            utils_with_delayed_decision.append(add1 + add2 + rem1 + rem2 + overhead_new)
                            utils_old.append(add1 + add2_old + rem1 + rem2 + overhead_new)
                        # objective
                        if objective == 1:
                            old_underutil = [self.threshold - x for x in utils_old]
                            new_underutil = [self.threshold - x for x in utils_with_delayed_decision]
                            print("old", old_underutil)
                            print("new", new_underutil)
                        else:
                            raise("delayed decisions not implemented for objective=", objective)
                        print("delayed", delayed)
                        print("total", utils_with_delayed_decision)
                        if utils_with_delayed_decision[-1] <= self.threshold:
                            print("DELAY port", p)

                            A1[p] = delayed
                            XOpt[p] = 1
                            delegated.remove(p)

                    if V[p] == 0 and YOpt[p] == 1:
                        print("--- keep?", p)      
                        print ("r",p, A2[p][0], E2[p][0])


        # (debugging) store decisions 
        self.counters_decisions[start] = {}
        for p, port in self.ports.items():
            # v=0 --> is delegated, y=1 --> stay delegated
            if V[p] == 0 and YOpt[p] == 1:
                self.counters_decisions[start][p] = 1 
                self.dss_result.update_util_per_port(start, p, A2[p][0]+1, E2[p][0])
                #self.dss_result.update_demand_per_port(start, p, D1[p][1], DRaw[p])
            # v=0 --> is delegated, y=0 --> remove
            if V[p] == 0 and YOpt[p] == 0:
                self.counters_decisions[start][p] = 0  
                self.dss_result.update_util_per_port(start, p, E2[p][0], E2[p][0])
                #self.dss_result.update_demand_per_port(start, p, D1[p][1], DRaw[p])
            # v=1 --> is not delegated, x=0 --> add
            if V[p] == 1 and XOpt[p] == 0:
                self.counters_decisions[start][p] = 1   
                self.dss_result.update_util_per_port(start, p, E1[p][0]+1, A1[p][0])
                #self.dss_result.update_demand_per_port(start, p, D2[p][0], DRaw[p])
            # v=1 --> is not delegated, x=1 --> stay not delegated
            if V[p] == 1 and XOpt[p] == 1:
                self.counters_decisions[start][p] = 0  
                self.dss_result.update_util_per_port(start, p, A1[p][0], A1[p][0]) 
                #self.dss_result.update_demand_per_port(start, p, D2[p][0], DRaw[p])
        # (debugging) calculate utilization            
        new_utils = []      
        for d in range(0, self.look_ahead):
            add1 = sum([V[p] * XOpt[p] * A1[p][d] for p in A1])
            add2 = sum([V[p] * (1-XOpt[p]) * E1[p][d] for p in E1])
            rem1 = sum([(1-V[p]) * YOpt[p] * A2[p][d] for p in A2])
            rem2 = sum([(1-V[p]) * (1-YOpt[p]) * E2[p][d] for p in E2])
            new_utils.append(str(add1 + add2 + rem1 + rem2))

        # (debugging) calculate overutilization
        for d in range(0, self.look_ahead):
            add1 = sum([V[p] * XOpt[p] * A1[p][d] for p in A1])
            add2 = sum([V[p] * (1-XOpt[p]) * E1[p][d] for p in E1])
            rem1 = sum([(1-V[p]) * YOpt[p] * A2[p][d] for p in A2])
            rem2 = sum([(1-V[p]) * (1-YOpt[p]) * E2[p][d] for p in E2])
            overutil += mu[start+d] * (self.threshold - (add1 + add2 + rem1 + rem2))

        # (debugging) calculate overhead
        overhead = 0
        allx_neg = sum([1-x for _, x in XOpt.items()])
        ally_neg = sum([1-y for _, y in YOpt.items()])
        overhead = allv_neg + allx_neg - ally_neg

        print("--- t=%d, rawUtil=%d, LA=%d ---"  % (
            start, self.size(start), self.look_ahead), 
            ' - '.join(new_utils), "added", delegated, "removed", removed,
            "util=%d" % overutil, "overhead=%d" % overhead)

        # store the calculated value for the expected overall flowtable utilization to allow
        # a comparison with the simulator results
        self.counters_util[start] = int(new_utils[0]) + overhead
        
        # store utils in dss result (is used by RSS)
        self.dss_result.update_util(start, int(new_utils[0]) + overhead)
        
        return dict(
            feasable=m.status!=3,
            status=m.status,
            overutil=overutil,
            YOpt=YOpt,
            XOpt=XOpt)

    def solver2_05_multi_objective(self, start, V, A1, E1, A2, E2, D1, D2, mu):
        
        m = Model("solver2_05_multi_objective")
        allx = []
        allx_neg = []
        ally = []
        ally_neg = []
        allv_neg = []
        all_overhead = []
        X = {}
        Y = {}
        for p, port in self.ports.items():
            allv_neg.append(1-V[p])
            if port.is_delegated:
                assert(V[p] == 0)
                Y[p] = m.addVar(vtype=GRB.BINARY, name='y_%.4d' % (p))
                ally.append(Y[p])
                ally_neg.append(1-Y[p])
            else:
                assert(V[p] == 1)
                X[p] = m.addVar(vtype=GRB.BINARY, name='x_%.4d' % (p))
                allx.append(X[p])
                allx_neg.append(1-X[p])

        overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)
        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            m.addConstr(add1 + add2 + rem1 + rem2 + overhead <= self.threshold)

        # parameters for setObjectiveN
        # see https://www.gurobi.com/documentation/9.0/refman/py_model_setobjectiven.html
        obj_index = 1 # different per objective and >0
        obj_priority = 0 # all parts have the same priority

        w_table = self.ctx.config.get('param_dts_weight_table')
        w_link = self.ctx.config.get('param_dts_weight_link')
        w_ctrl = self.ctx.config.get('param_dts_weight_ctrl')

        # rule overhead
        if w_table > 0:
            obj_index += 1 # increase index if a new objective is added 
            m.setObjectiveN(overhead, obj_index, obj_priority, w_table)        

        # link overhead
        if w_link > 0:
            obj_index += 1 # increase index if a new objective is added
            link_overhead = []
            for p, port in self.ports.items():
                if p == 0: continue;
                # port is delegated
                if V[p] == 0:
                    # stay delegated 
                    cv0y1 = D1[p][0] +1
                    if len(D1[p]) > 1:
                        cv0y1 = D1[p][0] + D1[p][1]  +1
                    #cv0y1 = sum(D1[p])+1
                    # remove delegation 
                    cv0y0 = 0
                    # add cost
                    link_overhead.append(Y[p]*cv0y1)
                    
                # port is not delegated
                if V[p] == 1:
                    # stay undelegated
                    cv1x1 = 0
                    # add delegation
                    cv1x0 = D2[p][0] + 1 # this could also be replaced with static 1 (similar results, marginal worse)
                    #cv1x0 = sum(D2[p]) + 1
                    # add cost
                    link_overhead.append((1-X[p])*cv1x0)
            m.setObjectiveN(quicksum(link_overhead), obj_index, obj_priority, w_link)

        # ctrl overhead
        if w_ctrl > 0:
            obj_index += 1 # increase index if a new objective is added
            ctrl_overhead = []
            for p, port in self.ports.items():
                if p == 0: continue;
                # port is delegated
                if V[p] == 0:
                    # stay delegated 
                    cv0x1 = E2[p][self.look_ahead-1] - A2[p][self.look_ahead-1]
                    # remove delegation 
                    cv0x0 = E2[p][0] - A2[p][0]
                    # add cost
                    # the mu[start] part is there to reduce cost to zero if no delegation is required
                    ctrl_overhead.append(Y[p]*cv0x1 + (1-Y[p])*cv0x0 * mu[start])
                # port is not delegated
                if V[p] == 1:
                    # stay undelegated
                    cv1y1 = 0 # just mentioned here for completeness
                    # add delegation
                    cv1y0 = A1[p][self.look_ahead-1] - E1[p][self.look_ahead-1] + (1-mu[start])*10
                    # add cost
                    ctrl_overhead.append((1-X[p])*cv1y0)
            m.setObjectiveN(quicksum(ctrl_overhead), obj_index, obj_priority, w_ctrl)

        return m

    def solver2_04_backdelegation(self, start, V, A1, E1, A2, E2, mu):
        """
        Optimize the number of flows that are "backdelegated". Backdelegations
        occur if a delegation status is changed from active to inactive and the
        remaining delegated flows have to be moved back to the delegation switch.
 
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        Input variables:
        [ ][ ] A1 --> do nothing (stay undelegated)
        [ ][x] E1 --> add delegation
        [x][x] A2 --> do nothing (stay delegated)
        [x][ ] E2 --> remove delegation

        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        m = Model("solver2_04_backdelegation")
        allx = []
        allx_neg = []
        ally = []
        ally_neg = []
        allv_neg = []
        all_overhead = []
        X = {}
        Y = {}
        for p, port in self.ports.items():
            allv_neg.append(1-V[p])
            if port.is_delegated:
                assert(V[p] == 0)
                Y[p] = m.addVar(vtype=GRB.BINARY, name='y_%.4d' % (p))
                ally.append(Y[p])
                ally_neg.append(1-Y[p])
            else:
                assert(V[p] == 1)
                X[p] = m.addVar(vtype=GRB.BINARY, name='x_%.4d' % (p))
                allx.append(X[p])
                allx_neg.append(1-X[p])

        overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)
        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            m.addConstr(add1 + add2 + rem1 + rem2 + overhead <= self.threshold)


        
 

        # todo: rename cv0x1 -> cv0y1 (x and y switched); only naming issue
        cost_backdelegation = []
        for p, port in self.ports.items():
            if p == 0: continue;
            # port is delegated
            if V[p] == 0:
                # stay delegated 
                cv0x1 = E2[p][self.look_ahead-1] - A2[p][self.look_ahead-1]
                # remove delegation 
                cv0x0 = E2[p][0] - A2[p][0]
                # add cost
                # the mu[start] part is there to reduce cost to zero if no delegation is required
                cost_backdelegation.append(Y[p]*cv0x1 + (1-Y[p])*cv0x0 * mu[start])
                
            # port is not delegated
            if V[p] == 1:
                # stay undelegated
                cv1y1 = 0 # just mentioned here for completeness
                # add delegation
                cv1y0 = A1[p][self.look_ahead-1] - E1[p][self.look_ahead-1] + (1-mu[start])*10
                # add cost
                cost_backdelegation.append((1-X[p])*cv1y0)

        m.setObjective(quicksum(cost_backdelegation), GRB.MINIMIZE)
        #m.setObjectiveN(overhead, 1, 2)
        #m.setObjectiveN(quicksum(cost_backdelegation), 0, 3)
        return m

    def solver2_03_overhead(self, start, V, A1, E1, A2, E2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        Input variables:
        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        m = Model("solver2_03_overhead")
        allx = []
        allx_neg = []
        ally = []
        ally_neg = []
        allv_neg = []
        all_overhead = []
        X = {}
        Y = {}
        for p, port in self.ports.items():
            allv_neg.append(1-V[p])
            if port.is_delegated:
                Y[p] = m.addVar(vtype=GRB.BINARY, name='y_%.4d' % (p))
                ally.append(Y[p])
                ally_neg.append(1-Y[p])
            else:
                X[p] = m.addVar(vtype=GRB.BINARY, name='x_%.4d' % (p))
                allx.append(X[p])
                allx_neg.append(1-X[p])

        overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)
        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            m.addConstr(add1 + add2 + rem1 + rem2 + overhead <= self.threshold)

        m.setObjective(overhead, GRB.MINIMIZE)
        return m
        #m.setObjectiveN(quicksum(arr), 1, 1)


        # x -> intended value is 1 for no delegation, i.e., minus
        # y -> intended value is 0 for remove all delegations, i.e., plus
        # todo: replace 10 with #ports
        #m.setObjective(quicksum(arr) - quicksum(allx)/10 + quicksum(ally)/10, GRB.MINIMIZE)


        #m.setObjective(-quicksum(allx), GRB.MINIMIZE)
        #m.setObjectiveN(quicksum(arr), 1, 1)

    def solver2_02_demand(self, start, V, A1, E1, A2, E2, D1, D2, mu):
        """
        Minimize the demand that is delegated.

        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        Input variables:
        [ ][ ] A1 --> do nothing (stay undelegated)
        [ ][x] E1 --> add delegation
        [x][x] A2 --> do nothing (stay delegated)
        [x][ ] E2 --> remove delegation

        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        m = Model("solver2_02_demand")
        allx = []
        allx_neg = []
        ally = []
        ally_neg = []
        allv_neg = []
        all_overhead = []
        X = {}
        Y = {}
        for p, port in self.ports.items():
            allv_neg.append(1-V[p])
            if port.is_delegated:
                assert(V[p] == 0)
                Y[p] = m.addVar(vtype=GRB.BINARY, name='y_%.4d' % (p))
                ally.append(Y[p])
                ally_neg.append(1-Y[p])
            else:
                assert(V[p] == 1)
                X[p] = m.addVar(vtype=GRB.BINARY, name='x_%.4d' % (p))
                allx.append(X[p])
                allx_neg.append(1-X[p])

        overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)
        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            m.addConstr(add1 + add2 + rem1 + rem2 + overhead <= self.threshold)

        # the +1 is required to make sure that delegations are removed at the
        # end if no delegation is required any more; besides that, it is possible
        # to use sum_t(Dx[p][t]) with t=1,...,L here which will give different 
        # results
        #
        # TODO: using 2 values for D1 and only the first value for D2 is 
        # an empirical solution --> systematic tests required but this seems to lead to
        # the best results
        cost_demand = []
        for p, port in self.ports.items():
            if p == 0: continue;
            # port is delegated
            if V[p] == 0:
                # stay delegated 
                cv0y1 = D1[p][0] +1
                if len(D1[p]) > 1:
                    cv0y1 = D1[p][0] + D1[p][1]  +1
                #cv0y1 = sum(D1[p])+1
                # remove delegation 
                cv0y0 = 0
                # add cost
                cost_demand.append(Y[p]*cv0y1)
                
            # port is not delegated
            if V[p] == 1:
                # stay undelegated
                cv1x1 = 0
                # add delegation
                cv1x0 = D2[p][0] + 1 # this could also be replaced with static 1 (similar results, marginal worse)
                #cv1x0 = sum(D2[p]) + 1
                # add cost
                cost_demand.append((1-X[p])*cv1x0)

        m.setObjective(quicksum(cost_demand), GRB.MINIMIZE)
        #m.setObjectiveN(overhead, 1, 2)
        #m.setObjectiveN(quicksum(cost_demand), 0, 3)
        return m

    def solver2_01_underutil(self, start, V, A1, E1, A2, E2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        Input variables:
        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        m = Model("solver2_01_underutil")
        allx = []
        allx_neg = []
        ally = []
        ally_neg = []
        allv_neg = []
        X = {}
        Y = {}
        for p, port in self.ports.items():
            allv_neg.append(1-V[p])
            if port.is_delegated:
                Y[p] = m.addVar(vtype=GRB.BINARY, name='y_%.4d' % (p))
                ally.append(Y[p])
                ally_neg.append(1-Y[p])
            else:
                X[p] = m.addVar(vtype=GRB.BINARY, name='x_%.4d' % (p))
                allx.append(X[p])
                allx_neg.append(1-X[p])

        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)
            m.addConstr(add1 + add2 + rem1 + rem2 + overhead <= self.threshold)


        # note that this is the invertion of the overhead case
        sidecase = quicksum(allx) + quicksum(ally_neg)

        arr2 = []
        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)

            z = self.threshold - (add1 + add2 + rem1 + rem2 + overhead)
 
            # we need another differentiation here because we minize absolute values with respect
            # to the threshold. Without overutilization, the problem would try to artificially 
            # increase the load on the flow table of the delegation switch in order to improve the
            # objective (because higher load in the delegation switch means that the absolute difference
            # between threshold and utilization smaller). We therefore differentiate between two cases:
            # 1) there is overutilization (mu=1) --> minimize |util(DS)-threshold|
            # 2) no overutilization (mu=0) --> minimize new delegations, maximize remove delegation
            # the latter one is modeled in the sidecase variable
            arr2.append((mu[start+d] * z) - ((1-mu[start+d]) * sidecase ))

        m.setObjective(quicksum(arr2), GRB.MINIMIZE)
        return m

    def solver2_011_underutil_relaxed(self, start, V, A1, E1, A2, E2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        Input variables:
        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        m = Model("solver2_011_underutil_relaxed")
        allx = []
        allx_neg = []
        ally = []
        ally_neg = []
        allv_neg = []
        X = {}
        Y = {}
        for p, port in self.ports.items():
            allv_neg.append(1-V[p])
            if port.is_delegated:
                Y[p] = m.addVar(vtype=GRB.BINARY, name='y_%.4d' % (p))
                ally.append(Y[p])
                ally_neg.append(1-Y[p])
            else:
                X[p] = m.addVar(vtype=GRB.BINARY, name='x_%.4d' % (p))
                allx.append(X[p])
                allx_neg.append(1-X[p])

        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)
            m.addConstr(add1 + add2 + rem1 + rem2 + overhead <= self.threshold + 5)


        # note that this is the invertion of the overhead case
        sidecase = quicksum(allx) + quicksum(ally_neg)

        arr2 = []
        for d in range(0, self.look_ahead):
            add1 = quicksum([V[p] * X[p] * A1[p][d] for p in A1])
            add2 = quicksum([V[p] * (1-X[p]) * E1[p][d] for p in E1])
            rem1 = quicksum([(1-V[p]) * Y[p] * A2[p][d] for p in A2])
            rem2 = quicksum([(1-V[p]) * (1-Y[p]) * E2[p][d] for p in E2])
            overhead = quicksum(allv_neg) + quicksum(allx_neg) - quicksum(ally_neg)

            # see http://lpsolve.sourceforge.net/5.1/absolute.htm if you are not familiar with 
            # modeling absolute values
            z1 = m.addVar(vtype=GRB.INTEGER, name='z1')
            z1 = self.threshold - (add1 + add2 + rem1 + rem2 + overhead)
            z2 = m.addVar(vtype=GRB.INTEGER, name='z2')
            m.addConstr(z1 <= z2)
            m.addConstr(-z1 <= z2)

            # we need another differentiation here because we minimize absolute values with respect
            # to the threshold. Without overutilization, the problem would try to artificially 
            # increase the load on the flow table of the delegation switch in order to improve the
            # objective (because higher load in the delegation switch means that the absolute difference
            # between threshold and utilization smaller). We therefore differentiate between two cases:
            # 1) there is overutilization (mu=1) --> minimize |util(DS)-threshold|
            # 2) no overutilization (mu=0) --> minimize new delegations, maximize remove delegation
            # the latter one is modeled in the sidecase variable
            arr2.append((mu[start+d] * z2) - ((1-mu[start+d]) * sidecase ))

        m.setObjective(quicksum(arr2), GRB.MINIMIZE)
        return m

    # ==========================
    # SOLVER 3
    # ==========================

    def solver3_wrapper(self):

        self.updateDSSResultWithoutDelegation()

        tick = self.get_first_overutilization()-(self.look_ahead*2)
        tick_last = self.get_last_overutilization()+(self.look_ahead*4)

        if self.get_first_overutilization() == 0 and self.get_last_overutilization() == 0:
            print("skip solving, no overhead situation")
            self.ctx.statistics['solver.nothing_to_do'] = 1 
            self.commands = {}
            return

        # default iterative solver
        for start in range(tick, self.maxtick-self.look_ahead-1):

            # init the empty command array for this tick
            self.commands[start] = dict(remove=[], add=[])

            result = self.solver3_wrapper_runmodel(start)
            if result:
                XOpt = result.get('XOpt')
                YOpt = result.get('YOpt')

                # add new delegations
                for p, v in XOpt.items():
                    port = self.ports[p]
                    if v == 0:
                        if port.is_delegated:
                            raise RuntimeError('port selected by XOpt is already delegated')
                        else:
                            self.dss_result.update(start, p, 1)
                            self.commands[start]['add'].append((p, 'ES'));
                            port.is_delegated = True # mark port as delegated
                            port.delegated_at = start
                            port.old_flows = []
                            epoch = port.epochs[start] # get the relevant epoch
                            # store all flows that are NOT actually delegated right now, i.e.,
                            # the flows that arrived prior to the delegation
                            for flow in epoch.active:
                                #if not flow in epoch.started:
                                port.old_flows.append(flow)
                    else:
                        self.dss_result.update(start, p, 0)    

                # remove delegations
                for p, v in YOpt.items():
                    port = self.ports[p]
                    if v == 0:
                        if not port.is_delegated:
                            raise RuntimeError('port selected by YOpt is not delegated')
                        else:
                            self.dss_result.update(start, p, 0)
                            self.commands[start]['remove'].append((p, 'ES'));
                            port.is_delegated = False
                            port.old_flows = []
                    else:
                        self.dss_result.update(start, p, 1) 

                for p, port in self.ports.items():
                    if port.is_delegated:
                        self.dss_result.update(start, p, 1)
                    else:
                        self.dss_result.update(start, p, 0) 

                # update dss_result
                DNew = self.getDemand(start)
                DRaw = self.getDemandRaw(start)
                for p, port in self.ports.items():
                    self.dss_result.update_demand_per_port(start, p, DNew[p], DRaw[p])
    
    def solver3_wrapper_runmodel(self, start):

        # start by skipping all cases where 
        mu = self.getMu()
        V = self.getV()
        allv_neg = sum([1-v for _, v in V.items()])
        allmu_lookahead = sum(mu[t] for t in range(start, start+self.look_ahead))
        if allv_neg + allmu_lookahead == 0:
            #print("--- Results for start=%d, look_ahead=%d : skipped, no overutil and no active delegations"  % (start, self.look_ahead)) 
            #return None
            pass

        # prepare the inputs for the optimization problem
        stime = time.time()
        A1 = self.getA(start)
        E1 = self.getE(start)
        A2 = self.getA2(start)
        E2 = self.getE2(start)
        D1 = self.getDemand1(start)
        D2 = self.getDemand2(start)
        self.statistics_append('solver.stats_time_modeling', time.time()-stime)

        #print("----- tick=%d" % start)
        #self.print('Table A1:', A1)
        #self.print('Table E1:', E1)
        #self.print('Table A2:', A2)
        #self.print('Table E2:', E2)

        # run optimization using the objective specified by the experiment
        objective = self.ctx.config.get('param_dts_objective')
        stime = time.time()
        result = None
        if objective == 1: result = self.solver3_03_overhead(start, V, A1, E1, A2, E2, mu);
        if objective == 2: result = self.solver3_02_demand(start, V, A1, E1, A2, E2, D1, D2, mu);
        if objective == 3: result = self.solver3_03_overhead(start, V, A1, E1, A2, E2, mu);
        if objective == 4: result = self.solver3_04_backdelegation(start, V, A1, E1, A2, E2, mu);
        if objective == 5: result = self.solver3_05_multi_objective(start, V, A1, E1, A2, E2, D1, D2, mu);

        self.statistics_append('solver.stats_time_solving', time.time()-stime)


        # now analyze the results
        XOpt = result.get('XOpt')
        YOpt = result.get('YOpt')
        delegated = [self.ports.get(p).label for p in result.get('delegated')]
        removed = [self.ports.get(p).label for p in result.get('removed')]
        overutil = 0

        # (debugging) calculate delegated demand            
        total_demand = 0
        for p, port in self.ports.items():
            if port.is_delegated:
                self.counters_demand += D1[p][0]

        # (debugging) store decisions 
        self.counters_decisions[start] = {}
        for p, port in self.ports.items():
            # v=0 --> is delegated, y=1 --> stay delegated
            if V[p] == 0 and YOpt[p] == 1:
                self.counters_decisions[start][p] = 1 
                self.dss_result.update_util_per_port(start, p, A2[p][0], E2[p][0])
            # v=0 --> is delegated, y=0 --> remove
            if V[p] == 0 and YOpt[p] == 0:
                self.counters_decisions[start][p] = 0  
                self.dss_result.update_util_per_port(start, p, E2[p][0], E2[p][0])
            # v=1 --> is not delegated, x=0 --> add
            if V[p] == 1 and XOpt[p] == 0:
                self.counters_decisions[start][p] = 1   
                self.dss_result.update_util_per_port(start, p, E1[p][0], A1[p][0])
            # v=1 --> is not delegated, x=1 --> stay not delegated
            if V[p] == 1 and XOpt[p] == 1:
                self.counters_decisions[start][p] = 0
                self.dss_result.update_util_per_port(start, p, A1[p][0], A1[p][0])  

        # (debugging) calculate utilization            
        new_utils = []      
        for d in range(0, self.look_ahead):
            add1 = sum([V[p] * XOpt[p] * A1[p][d] for p in A1])
            add2 = sum([V[p] * (1-XOpt[p]) * E1[p][d] for p in E1])
            rem1 = sum([(1-V[p]) * YOpt[p] * A2[p][d] for p in A2])
            rem2 = sum([(1-V[p]) * (1-YOpt[p]) * E2[p][d] for p in E2])
            new_utils.append(str(add1 + add2 + rem1 + rem2))


        # (debugging) calculate overutilization
        for d in range(0, self.look_ahead):
            add1 = sum([V[p] * XOpt[p] * A1[p][d] for p in A1])
            add2 = sum([V[p] * (1-XOpt[p]) * E1[p][d] for p in E1])
            rem1 = sum([(1-V[p]) * YOpt[p] * A2[p][d] for p in A2])
            rem2 = sum([(1-V[p]) * (1-YOpt[p]) * E2[p][d] for p in E2])
            overutil += mu[start+d] * (self.threshold - (add1 + add2 + rem1 + rem2))

        # (debugging) calculate overhead
        overhead = 0
        allx_neg = sum([1-x for _, x in XOpt.items()])
        ally_neg = sum([1-y for _, y in YOpt.items()])
        overhead = allv_neg + allx_neg - ally_neg

        print("--- t=%d, rawUtil=%d, LA=%d ---"  % (
            start, self.size(start), self.look_ahead), 
            ' - '.join(new_utils), "added", delegated, "removed", removed,
            "util=%d" % overutil, "overhead=%d" % overhead)

        # store the calculated value for the expected overall flowtable utilization to allow
        # a comparison with the simulator results
        self.counters_util[start] = int(new_utils[0]) + overhead

        # store utils in dss result (is used by RSS)
        self.dss_result.update_util(start, int(new_utils[0]) + overhead)
       

        """
        current_util = 0
        for p, v in V.items():  
            if v == 1: # not delegated
                A1[p][0] 
            if v == 0:
                A2[p][0]+1  # +1 for overhead!  
        """

        return dict(
            overutil=overutil,
            YOpt=YOpt,
            XOpt=XOpt)

    def solver3_05_multi_objective(self, start, V, A1, E1, A2, E2, D1, D2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        [ ][ ] A1 --> do nothing (stay undelegated)
        [ ][x] E1 --> add delegation
        [x][x] A2 --> do nothing (stay delegated)
        [x][ ] E2 --> remove delegation

        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        XOpt = {}
        YOpt = {}
        delegated = []
        removed = []
        for p, port in self.ports.items():
            XOpt[p] = 1
            YOpt[p] = 1

        # calculate expected utilization
        U = [0]*self.look_ahead
        for p, v in V.items():  
            if v == 1: # not delegated
                for i, val in enumerate(A1[p]):
                    U[i] += A1[p][i] 
            if v == 0:
                for i, val in enumerate(A2[p]):
                    U[i] += A2[p][i]+1  # +1 for overhead!            

        # get maximum value out of lookahead
        util = 0
        utilindex = -1
        for i, u in enumerate(U):
            if u > util:
                util = u
                utilindex = i

        w_table = self.ctx.config.get('param_dts_weight_table')
        w_link = self.ctx.config.get('param_dts_weight_link')
        w_ctrl = self.ctx.config.get('param_dts_weight_ctrl')

        if util > self.threshold:
            # we have to add delegations
            savesum = 0
            # presort ports according to objective
            sorted_ports = []
            for p, v in V.items():
                if v == 1:

                    rating = 0 # represents the sorting weight

                    # rule overhead
                    if w_table > 0:
                        rating += w_table * -1 * ((A1[p][utilindex] - E1[p][utilindex]) - 1)
  
                    # link overhead
                    if w_link > 0:
                        if len(D2[p]) > 1:
                            rating += w_link * (D2[p][0]+D2[p][1])
                        else:
                            rating += w_link * (D2[p][0])
                            
                    # ctrl overhead
                    if w_ctrl > 0:
                        rating += w_ctrl * (A1[p][self.look_ahead-1] - E1[p][self.look_ahead-1])

                    sorted_ports.append((rating, p))

            for rating, p in sorted(sorted_ports, key=lambda e: e[0]):
                if util - savesum <= self.threshold:
                    break;
                save = (A1[p][utilindex] - E1[p][utilindex]) - 1 
                if save > 0:
                    savesum += save
                    XOpt[p] = 0
                    delegated.append(p) 
        else:
            skip = False
            for u in U:
                # all values in u should be below the threshold to remove a delegation
                if u > self.threshold: skip=True
            if not skip:
                # we have to remove delegations
                freesum = 0
                # presort ports according to objective
                sorted_ports = []
                for p, v in V.items():
                    if v == 0:

                        rating = 0 # represents the sorting weight

                        # rule overhead
                        if w_table > 0:
                            rating += w_table * -1 * (E2[p][utilindex] - A2[p][utilindex] + 1)
      
                        # link overhead
                        if w_link > 0:
                            if len(D1[p]) > 1:
                                rating += w_link * (D1[p][0]+D1[p][1])
                            else:
                                rating += w_link * (D1[p][0])
                                
                        # ctrl overhead
                        if w_ctrl > 0:
                            rating += w_ctrl * (E2[p][0] - A2[p][0])

                        sorted_ports.append((rating, p))

                for rating, p in sorted(sorted_ports, key=lambda e: e[0], reverse=True):
                    # abort if utilization is below threshold
                    if util + freesum >= self.threshold:
                        break;
                    # currently not delegatged
                    free =  E2[p][utilindex] - A2[p][utilindex] + 1
                    if free >= 1 and free+freesum+util <= self.threshold:
                        freesum += free
                        YOpt[p] = 0
                        removed.append(p)      

        return dict(
            YOpt=YOpt,
            XOpt=XOpt,
            delegated=delegated,
            removed=removed)
  
    def solver3_04_backdelegation(self, start, V, A1, E1, A2, E2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        [ ][ ] A1 --> do nothing (stay undelegated)
        [ ][x] E1 --> add delegation
        [x][x] A2 --> do nothing (stay delegated)
        [x][ ] E2 --> remove delegation

        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        XOpt = {}
        YOpt = {}
        delegated = []
        removed = []
        for p, port in self.ports.items():
            XOpt[p] = 1
            YOpt[p] = 1

        # calculate expected utilization
        U = [0]*self.look_ahead
        for p, v in V.items():  
            if v == 1: # not delegated
                for i, val in enumerate(A1[p]):
                    U[i] += A1[p][i] 
            if v == 0:
                for i, val in enumerate(A2[p]):
                    U[i] += A2[p][i]+1  # +1 for overhead!            

        # get maximum value out of lookahead
        util = 0
        utilindex = -1
        for i, u in enumerate(U):
            if u > util:
                util = u
                utilindex = i

        if util > self.threshold:
            # we have to add delegations
            savesum = 0
            # presort ports according to objective
            sorted_ports = []
            for p, v in V.items():
                if v == 1:
                    backdelegations = A1[p][self.look_ahead-1] - E1[p][self.look_ahead-1]
                    sorted_ports.append((backdelegations, p))

            # reverse=true is set because the highest values should be chose first;
            # higher values first = less delegated rules = objective minimized
            for value, p in sorted(sorted_ports, key=lambda e: e[0], reverse=True):
                if util - savesum <= self.threshold:
                    break;
                save = (A1[p][utilindex] - E1[p][utilindex]) - 1 
                if save > 0:
                    savesum += save
                    XOpt[p] = 0
                    delegated.append(p) 
        else:
            skip = False
            for u in U:
                # all values in u should be below the threshold to remove a delegation
                if u > self.threshold: skip=True
            if not skip:
                # we have to remove delegations
                freesum = 0
                # presort ports according to objective
                sorted_ports = []
                for p, v in V.items():
                    if v == 0:
                        backdelegations = E2[p][0] - A2[p][0]
                        sorted_ports.append((backdelegations, p))

                # normal sort (i.e., lowest values first is fine) because
                # we want to remove as many ports as possible
                for value, p in sorted(sorted_ports, key=lambda e: e[0]):
                    # abort if utilization is below threshold
                    if util + freesum >= self.threshold:
                        break;
                    # currently not delegatged
                    free =  E2[p][utilindex] - A2[p][utilindex] + 1
                    # TODO: >=1
                    if free > 1 and free+freesum+util <= self.threshold:
                        freesum += free
                        YOpt[p] = 0
                        removed.append(p)      

        return dict(
            YOpt=YOpt,
            XOpt=XOpt,
            delegated=delegated,
            removed=removed)

    def solver3_03_overhead(self, start, V, A1, E1, A2, E2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        [ ][ ] A1 --> do nothing (stay undelegated) Xp=1 
        [ ][x] E1 --> add delegation Xp=0
        [x][x] A2 --> do nothing (stay delegated)
        [x][ ] E2 --> remove delegation

        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        XOpt = {}
        YOpt = {}
        delegated = []
        removed = []
        for p, port in self.ports.items():
            XOpt[p] = 1
            YOpt[p] = 1



        # calculate expected utilization
        U = [0]*self.look_ahead
        for p, v in V.items():  
            if v == 1: # not delegated
                for i, val in enumerate(A1[p]):
                    U[i] += A1[p][i] 
            if v == 0:
                for i, val in enumerate(A2[p]):
                    U[i] += A2[p][i]+1  # +1 for overhead!            

        # get maximum value out of lookahead
        util = 0
        utilindex = -1
        for i, u in enumerate(U):
            if u > util:
                util = u
                utilindex = i

        if util > self.threshold:
            # we have to add delegations
            savesum = 0
            # presort ports according to objective
            sorted_ports = []
            for p, v in V.items():
                if v == 1:
                    save = (A1[p][utilindex] - E1[p][utilindex]) - 1
                    sorted_ports.append((save, p))

            # reverse=true is set because the highest values should be chose first;
            # higher values first = less delegated rules = objective minimized
            for value, p in sorted(sorted_ports, key=lambda e: e[0], reverse=True):
                if util - savesum <= self.threshold:
                    break;
                save = (A1[p][utilindex] - E1[p][utilindex]) - 1 
                if save > 0:
                    savesum += save
                    XOpt[p] = 0
                    delegated.append(p) 
        else:
            skip = False
            for u in U:
                # all values in u should be below the threshold to remove a delegation
                if u > self.threshold: skip=True
            if not skip:
                # we have to remove delegations
                freesum = 0
                # presort ports according to objective
                sorted_ports = []
                for p, v in V.items():
                    if v == 0:
                        free =  E2[p][utilindex] - A2[p][utilindex] + 1  
                        sorted_ports.append((free, p))

                # normal sort (i.e., lowest values first is fine) because
                # we want to remove as many ports as possible
                for value, p in sorted(sorted_ports, key=lambda e: e[0]):
                    # abort if utilization is below threshold
                    if util + freesum >= self.threshold:
                        break;
                    # currently not delegatged
                    free =  E2[p][utilindex] - A2[p][utilindex] + 1
                    # TODO: >=1
                    if free > 1 and free+freesum+util <= self.threshold:
                        freesum += free
                        YOpt[p] = 0
                        removed.append(p)      

        return dict(
            YOpt=YOpt,
            XOpt=XOpt,
            delegated=delegated,
            removed=removed)

    def solver3_02_demand(self, start, V, A1, E1, A2, E2, D1, D2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        [ ][ ] A1 --> do nothing (stay undelegated)
        [ ][x] E1 --> add delegation
        [x][x] A2 --> do nothing (stay delegated)
        [x][ ] E2 --> remove delegation

        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        XOpt = {}
        YOpt = {}
        delegated = []
        removed = []
        for p, port in self.ports.items():
            XOpt[p] = 1
            YOpt[p] = 1

        # calculate expected utilization
        U = [0]*self.look_ahead
        for p, v in V.items():  
            if v == 1: # not delegated
                for i, val in enumerate(A1[p]):
                    U[i] += A1[p][i] 
            if v == 0:
                for i, val in enumerate(A2[p]):
                    U[i] += A2[p][i]+1  # +1 for overhead!            

        # get maximum value out of lookahead
        util = 0
        utilindex = -1
        for i, u in enumerate(U):
            if u > util:
                util = u
                utilindex = i

        if util > self.threshold:
            # we have to add delegations
            savesum = 0
            # presort ports according to objective
            sorted_ports = []
            for p, v in V.items():
                if v == 1:
                    if len(D2[p]) > 1:
                        sorted_ports.append((D2[p][0]+D2[p][1], p))
                    else:
                        sorted_ports.append((D2[p][0], p))

            for value, p in sorted(sorted_ports, key=lambda e: e[0]):
                if util - savesum <= self.threshold:
                    break;
                save = (A1[p][utilindex] - E1[p][utilindex]) - 1 
                if save > 0:
                    savesum += save
                    XOpt[p] = 0
                    delegated.append(p) 
        else:
            skip = False
            for u in U:
                # all values in u should be below the threshold to remove a delegation
                if u > self.threshold: skip=True
            if not skip:
                # we have to remove delegations
                freesum = 0
                # presort ports according to objective
                sorted_ports = []
                for p, v in V.items():
                    if v == 0:
                        if len(D1[p]) > 1:
                            sorted_ports.append((D1[p][0]+D1[p][1], p))
                        else:
                            sorted_ports.append((D1[p][0], p))

                for value, p in sorted(sorted_ports, key=lambda e: e[0], reverse=True):
                    # abort if utilization is below threshold
                    if util + freesum >= self.threshold:
                        break;
                    # currently not delegatged
                    free =  E2[p][utilindex] - A2[p][utilindex] + 1
                    if free >= 1 and free+freesum+util <= self.threshold:
                        freesum += free
                        YOpt[p] = 0
                        removed.append(p)      

        return dict(
            YOpt=YOpt,
            XOpt=XOpt,
            delegated=delegated,
            removed=removed)

    # ==========================
    # SOLVER 9
    # ==========================

    def solver9_wrapper(self):

        tick = self.get_first_overutilization()-(self.look_ahead*2)
        tick_last = self.get_last_overutilization()+(self.look_ahead*2)

        if self.get_first_overutilization() == 0 and self.get_last_overutilization() == 0:
            print("skip solving, no overhead situation")
            self.ctx.statistics['solver.nothing_to_do'] = 1 
            self.commands = {}
            return

        # default iterative solver
        for start in range(tick, tick_last):

            # init the empty command array for this tick
            self.commands[start] = dict(remove=[], add=[])

            result = self.solver9_wrapper_runmodel(start)
            if result:
                XOpt = result.get('XOpt')
                YOpt = result.get('YOpt')

                # add new delegations
                for p, v in XOpt.items():
                    port = self.ports[p]
                    if v == 0:
                        if port.is_delegated:
                            raise RuntimeError('port selected by XOpt is already delegated')
                        else:
                            self.commands[start]['add'].append((p, 'ES'));
                            port.is_delegated = True # mark port as delegated
                            port.delegated_at = start
                            port.old_flows = []
                            epoch = port.epochs[start] # get the relevant epoch
                            # store all flows that are NOT actually delegated right now, i.e.,
                            # the flows that arrived prior to the delegation
                            for flow in epoch.active:
                                #if not flow in epoch.started:
                                port.old_flows.append(flow)

                # remove delegations
                for p, v in YOpt.items():
                    port = self.ports[p]
                    if v == 0:
                        if not port.is_delegated:
                            raise RuntimeError('port selected by YOpt is not delegated')
                        else:
                            self.commands[start]['remove'].append((p, 'ES'));
                            port.is_delegated = False
                            port.old_flows = []                       
    
    def solver9_wrapper_runmodel(self, start):

        # start by skipping all cases where 
        mu = self.getMu()
        V = self.getV()
        allv_neg = sum([1-v for _, v in V.items()])
        allmu_lookahead = sum(mu[t] for t in range(start, start+self.look_ahead))
        if allv_neg + allmu_lookahead == 0:
            #print("--- Results for start=%d, look_ahead=%d : skipped, no overutil and no active delegations"  % (start, self.look_ahead)) 
            #return None
            pass

        # prepare the inputs for the optimization problem
        stime = time.time()
        A1 = self.getA(start)
        E1 = self.getE(start)
        A2 = self.getA2(start)
        E2 = self.getE2(start)
        D1 = self.getDemand1(start)
        D2 = self.getDemand2(start)
        self.statistics_append('solver.stats_time_modeling', time.time()-stime)

        #print("----- tick=%d" % start)
        #self.print('Table A1:', A1)
        #self.print('Table E1:', E1)
        #self.print('Table A2:', A2)
        #self.print('Table E2:', E2)

        # run optimization using the objective specified by the experiment
        objective = self.ctx.config.get('param_dts_objective')
        stime = time.time()
        result = None
        if objective == 1: 
            raise(RuntimeError('not implemented'))
        if objective == 2:
            raise(RuntimeError('not implemented'))
        if objective == 3: 
            result = self.solver9_03_overhead(start, V, A1, E1, A2, E2, mu);
        if objective == 4: 
            raise(RuntimeError('not implemented'))

        self.statistics_append('solver.stats_time_solving', time.time()-stime)


        # now analyze the results
        XOpt = result.get('XOpt')
        YOpt = result.get('YOpt')
        delegated = result.get('delegated')
        removed = result.get('removed')
        overutil = 0

        # (debugging) calculate delegated demand            
        total_demand = 0
        for p, port in self.ports.items():
            if port.is_delegated:
                self.counters_demand += D1[p][0]

        # (debugging) store decisions 
        self.counters_decisions[start] = {}
        for p, port in self.ports.items():
            # v=0 --> is delegated, y=1 --> stay delegated
            if V[p] == 0 and YOpt[p] == 1:
                self.counters_decisions[start][p] = 1 
            # v=0 --> is delegated, y=0 --> remove
            if V[p] == 0 and YOpt[p] == 0:
                self.counters_decisions[start][p] = 0  
            # v=1 --> is not delegated, x=0 --> add
            if V[p] == 1 and XOpt[p] == 0:
                self.counters_decisions[start][p] = 1   
            # v=1 --> is not delegated, x=1 --> stay not delegated
            if V[p] == 1 and XOpt[p] == 1:
                self.counters_decisions[start][p] = 0  

        # (debugging) calculate utilization            
        new_utils = []      
        for d in range(0, self.look_ahead):
            add1 = sum([V[p] * XOpt[p] * A1[p][d] for p in A1])
            add2 = sum([V[p] * (1-XOpt[p]) * E1[p][d] for p in E1])
            rem1 = sum([(1-V[p]) * YOpt[p] * A2[p][d] for p in A2])
            rem2 = sum([(1-V[p]) * (1-YOpt[p]) * E2[p][d] for p in E2])
            new_utils.append(str(add1 + add2 + rem1 + rem2))

        # (debugging) calculate overutilization
        for d in range(0, self.look_ahead):
            add1 = sum([V[p] * XOpt[p] * A1[p][d] for p in A1])
            add2 = sum([V[p] * (1-XOpt[p]) * E1[p][d] for p in E1])
            rem1 = sum([(1-V[p]) * YOpt[p] * A2[p][d] for p in A2])
            rem2 = sum([(1-V[p]) * (1-YOpt[p]) * E2[p][d] for p in E2])
            overutil += mu[start+d] * (self.threshold - (add1 + add2 + rem1 + rem2))

        # (debugging) calculate overhead
        overhead = 0
        allx_neg = sum([1-x for _, x in XOpt.items()])
        ally_neg = sum([1-y for _, y in YOpt.items()])
        overhead = allv_neg + allx_neg - ally_neg

        print("--- t=%d, rawUtil=%d, LA=%d ---"  % (
            start, self.size(start), self.look_ahead), 
            ' - '.join(new_utils), "added", delegated, "removed", removed,
            "util=%d" % overutil, "overhead=%d" % overhead)

        # store the calculated value for the expected overall flowtable utilization to allow
        # a comparison with the simulator results
        self.counters_util[start] = int(new_utils[0]) + overhead


        return dict(
            overutil=overutil,
            YOpt=YOpt,
            XOpt=XOpt)

    def solver9_03_overhead(self, start, V, A1, E1, A2, E2, mu):
        """
        Overview over the decision variables:
        x(p) = 0 --> add a new delegation for p
        x(p) = 1 --> don't add delegation for p
        y(p) = 0 --> remove delegation from p
        y(p) = 1 --> don't remove delegation from p 

        [ ][ ] A1 --> do nothing (stay undelegated)
        [ ][x] E1 --> add delegation
        [x][x] A2 --> do nothing (stay delegated)
        [x][ ] E2 --> remove delegation

        v(p) == 0 --> port p is currently delegated
        v(p) == 1 --> port p is currently not delegated
        mu(t) == 0 --> overutil(t) < threshold
        mu(t) == 1 --> overutil(t) > threshold
        """
        XOpt = {}
        YOpt = {}
        delegated = []
        removed = []
        for p, port in self.ports.items():
            XOpt[p] = 1
            YOpt[p] = 1


        return dict(
            YOpt=YOpt,
            XOpt=XOpt,
            delegated=delegated,
            removed=removed)


    # ==========================
    # OTHER
    # ==========================

    def debug_backdelegations(self):
        """
        objective 222
        """

        tick = self.get_first_overutilization() - 1

        combinations = list(itertools.product([0, 1], repeat=self.look_ahead))
        for p, port in self.ports.items():
            print("P=%d" % p)
            for s, subset in enumerate(combinations):
                tick_arr = []
                for i, c in enumerate(subset):
                    if c == 1: tick_arr.append(tick+i)
                coeff, backdelegations = self.simulate_delegations(p, tick_arr, restrict=(tick, tick+self.look_ahead))
                print(subset, "-->", coeff, backdelegations)




        pass

    def debug_vtable(self):
        """
        objective 111 is only for debugging; It will check that
        the "simulated" results from the VTable with regard to 
        delegation and the actual results from the simulator match 1:1;
        For this, the objective will create 1000 random delegations,
        calculate the estimated flowtable impact in the VTable using the
        simulate_delegations() function and compare the results to the
        simulator run that is automatically executed afterwards.
        """
        print("!!!!!!!!!!!!!!!!!")
        print("This objective (111) runs a sanity check for the " + 
            "simulate_delegations() function. If this run does not raise an " +
            "assertion error, the test has passed successfully.")
        print("!!!!!!!!!!!!!!!!!")


        class FakeDel:
            def __init__(self, port, start, end):
                self.port = port
                self.start = start
                self.end = end

        allports = [p for p, port in self.ports.items()]

        # create 1000 random delegations to see whether the calculated utilization via
        # simulate_delegations and the actual simulator results match on each other; the
        # counter_port stuff is required to avoid overlapping delegations
        fake_dels = []
        counter_port = {}
        for i in range(1000):
            port = random.choice(allports)
            if not counter_port.get(port): counter_port[port] = 1
            start = counter_port.get(port) + random.randint(1, 10)
            if start > self.maxtick-50: continue
            end = start + random.randint(1, 10)
            counter_port[port] = end+1
            fake_dels.append(FakeDel(port,start,end))

        #fake_dels = [FakeDel(6,1,80), FakeDel(4,90,150), FakeDel(5,90,150), FakeDel(6,90,150), FakeDel(7,204,204)]
        vectors = []
        cnt_backdelegations = 0
        overhead = [0]*self.maxtick
        for p, port in self.ports.items():
            delegated_ticks = []
            # run through all fake delegations and create one array per port
            # that contains the ticks that have to be delegated, e.g., 
            # [4,5,6,7] if the port should be delegated at tick 4,5,6 and 7.
            for fdel in filter(lambda f: f.port == p, fake_dels):
                delegated_ticks += range(fdel.start, fdel.end+1)

            if len(delegated_ticks) > 0:
                print("port=%d delegated=%s" % (p, str(delegated_ticks)))
                coeff, backdelegations = self.simulate_delegations(p, delegated_ticks)
                vectors.append(coeff)
                cnt_backdelegations += backdelegations
            else:
                coeff, backdelegations = self.simulate_delegations(p, [])
                vectors.append(coeff)
                cnt_backdelegations += backdelegations
        print("cnt_backdelegations", cnt_backdelegations)

        # now create the sum of all port-based vectors which will be the 
        # flow table utilization array
        vectors = np.array(vectors).sum(axis=0)
        
        # prepare the commands and counters that are used to verify the results
        for t in range(1, self.maxtick-1):
            self.commands[t] = dict(remove=[], add=[])
            self.counters_util[t] = vectors[t] 

        # enable the fake delegations in the simulation by setting commands
        # array accordingly
        for fdel in fake_dels:
            self.commands[fdel.start]['add'].append((fdel.port, 'ES'))
            self.commands[fdel.end+1]['remove'].append((fdel.port, 'ES'))

class PBCESolver:

    def __init__(self, ctx):
        self.table = None
        self.ctx = ctx
        self.ctx.statistics['solver.cnt_timelimit'] = 0
        self.ctx.statistics['solver.cnt_infeasable'] = 0
        self.ctx.statistics['solver.cnt_feasable'] = 0
        self.ctx.statistics['solver.class'] = os.path.basename(__file__)

    def check_results(self, ctx, switch):
        """
        Compares analytical results with simulator results; Expected
        totaldiff is 0, but there can be small differences due to timing issues
        of the periodic polling function (see simulator.py stats offset solution)
        """
        if not switch:
            # single switch scenarios are hardcoded with DS and ES
            switch = ctx.topo.get_switch_by_label('DS')
        data = {}
        for r in switch.reports:
            data[r.tick] = r # get the switch reports where the flow table data is stored

        print("on_sim_done()")
        cnt = 0
        totaldiff = 0
        table_data = []
        try:
            table_data = self.table.ports[1].epochs
        except KeyError:
            logger.info('solver.check_results(): no data, probably a no-bottleneck scenario')

        if len(table_data) > 0:
            for t in range(0, len(table_data)-1):
                cnt += 1
                r = data.get(t)
                calculated = self.table.counters_util.get(t)
                if calculated is not None:
                    expected = data.get(t+1).cnt_active_flows
                    ports = data.get(t+1).cnt_ports_delegated
                    diff = abs(expected-calculated)
                    totaldiff += diff
                    if diff > 0:
                        print(t, expected, "-", calculated, expected==calculated, ports)
        
        if cnt > 0:
            result = totaldiff / float(cnt)
        else:
            result = 0
        logger.info("solver.check_results(): total=%d relative=%f" % (totaldiff, result))
        return result

    def on_sim_done(self, ctx, switch):
        """
        Same as check_results but with some additional debugging 
        """        
        if not switch:
            switch = ctx.topo.get_switch_by_label('DS')
        data = {}
        for r in switch.reports:
            data[r.tick] = r # get the switch reports where the flow table data is stored

        print("on_sim_done()")
        for t in range(0, len(self.table.ports[1].epochs)-1):
            r = data.get(t)
            calculated = self.table.counters_util.get(t)
            if calculated is not None:
                expected = data.get(t+1).cnt_active_flows
                ports = data.get(t+1).cnt_ports_delegated
                if abs(expected-calculated) > 0:
                    print(t, expected, "-", calculated, expected==calculated, ports)

        # 111 solver is different (non-incremental)
        if self.ctx.config.get('param_dts_objective') >= 100:
            for t in range(0, len(self.table.ports[1].epochs)-1):
                r = data.get(t)
                calculated = self.table.counters_util.get(t)
                if calculated is not None:
                    expected = data.get(t+1).cnt_active_flows
                    ports = data.get(t+1).cnt_ports_delegated
                    print(t, expected, "-", calculated, expected==calculated, ports)


        # The calculation of the coefficients is pretty tricky. To assure that
        # the calculated values are correct, we check whether the predicted 
        # value from time=t is identical to the actual value from the simulation at time=t+1 
        passed_checks = 0
        for t in range(0, len(self.table.ports[1].epochs)-1):
            r = data.get(t)
            # calculated is the flow table utilization predicted after optimization
            # expected is the actual flow table utilization taken from simulator results
            calculated = self.table.counters_util.get(t)
            if calculated is not None:
                expected = data.get(t+1).cnt_active_flows
                assert(int(calculated) == int(expected))
                passed_checks += 1

        logger.info("cross check with simulator: ALL FINE! passed checks: %d" % passed_checks)
        logger.info("self.counters_demand=%f" % self.table.counters_demand)
    
    def run(self, ctx):

        solver = ctx.config.get('param_dts_algo')
        table = VTable(ctx)
        self.table = table

        for i in range(0, self.table.maxtick-1):
            table.dss_result.update_util_raw(i, table.size(i))
        
        print("")
        print("run solver %d" % solver)
        print("  first_time_over: %d" % table.get_first_overutilization())
                    
        stime = time.time()

        # testing / debugging
        if self.ctx.config.get('param_dts_objective') == 111:
            print("using debug_vtable -> run vtable test")
            table.debug_vtable()
            return table.commands

        if self.ctx.config.get('param_dts_objective') == 222:
            print("using debug_backdelegations -> run backdelegation test")
            table.debug_backdelegations()
            return table.commands

        # solver selection
        if solver == 1:
            table.solver1_wrapper()

        if solver == 2:
            table.solver2_wrapper()

        if solver == 3:
            table.solver3_wrapper()

        if solver == 9:
            table.solver9_wrapper()

        self.ctx.statistics['solver.time_total'] = time.time()-stime

        print("  cnt_feasable", self.ctx.statistics['solver.cnt_feasable'] )
        print("  cnt_infeasable", self.ctx.statistics['solver.cnt_infeasable'] )
        print("  cnt_timelimit", self.ctx.statistics['solver.cnt_timelimit'] )
        print("  time_total", self.ctx.statistics['solver.time_total'] )

        # store DSS results to be used in RSS 
        # (can be disabled because this takes a huge amount of disc space)
        if ctx.config.get("param_include_dss_result_in_statistics") == 1:
            self.ctx.statistics['dss_result'] = table.dss_result.serialize()
        if ctx.scenario:
            ctx.scenario_result = table.dss_result







        """
        for t in range(1, self.table.maxtick-1):
            v = 0
            for p, port in self.table.ports.items():
                v += len(self.table.ports[p].epochs[t].active)

            if v > 0:
                logger.info('  %.3d  %d' % (t,v))
        """


        return table.commands

