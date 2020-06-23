# Implementation of the remote switch allocation (RSAlloc) algorithms

from gurobipy import *

import itertools
import random
import logging
import time
import sys
import os
import numpy
import io
import json
import hashlib
import statistics
import time

import matplotlib.pyplot as plt
from heapq import heappush, heappop, nsmallest
from contextlib import redirect_stdout

from topo.topology import TopologyFromDTSResult
from core.context import Context
from core.simulator import Simulator
from core.flow import Flow
from core.statistics import Timer

from engine.errors import TimeoutError
from engine.solve_util import VDelegationStatus, VDSSResult
from engine.solve_dts_data import DTSData
from engine.solve_rsa_data import RSAData

from engine.assignment_builder import AssignmentBuilder
from engine.solve_dts import DTSSolver

logger = logging.getLogger(__name__)

class VJobAssignment(object):
    """
    example for a demo scenario with 15 time slots where all time slots
    could be handled by S1 and S2 except for time slot 10 that can only
    be handled by S1: see final result marked entries with (*)

    ----- level=0
    indices: [(0, 14)]
    intersection of (0, 14): ['S1']
    final result: [('S1',)]
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']

    ----- level=1
    indices: [(0, 7), (8, 14)]
    intersection of (0, 7): ['S2', 'S1']
    intersection of (8, 14): ['S1'] <-- because it contains time slot 10!
    final result: [('S2', 'S1'), ('S1', 'S1')]
    ['S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']

    ----- level=2
    indices: [(0, 3), (4, 7), (8, 11), (12, 14)]
    intersection of (0, 3): ['S2', 'S1']
    intersection of (4, 7): ['S2', 'S1']
    intersection of (8, 11): ['S1']
    intersection of (12, 14): ['S2', 'S1']
    final result: [('S2', 'S2', 'S1', 'S2'), ('S2', 'S2', 'S1', 'S1'), ('S2', 'S1', 'S1', 'S2'), ('S2', 'S1', 'S1', 'S1'), ('S1', 'S2', 'S1', 'S2'), ('S1', 'S2', 'S1', 'S1'), ('S1', 'S1', 'S1', 'S2'), ('S1', 'S1', 'S1', 'S1')]

    ----- result alltogether:
                                                      (*)   (*)   (*)
    ['S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2']
    ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1']
    """
    def __init__(self, vjob, id, assignment):
        self.id = id # id of an assignment (its position index in the array)
        self.vjob = vjob # reference to the parent vjob object this assignment belongs to
        self.assignment = assignment # the assignment data (array of switches)
        self.rating = 1000000 # plain rating for this assignment, higher ratings are bad (initial=worst case)

    def get_weights(self):
        util = 0
        demand = 0
        for t, job in self.vjob.jobs.items():
            util += job.util_raw - job.util
            demand += job.demand
        return util, demand

    def get_delegations(self):
        """ return a list of tuples (s,t,v) where s is the switch
        this item of the assignment will delegate to, t is the time slot
        and v is the amount of rules that are delegated"""
        result = []
        idx = 0
        for t, job in self.vjob.jobs.items():
            new_tuple = (self.assignment[idx], t, job.util_raw - job.util)
            result.append(new_tuple)
            idx += 1
        return result

    def get_link_demand_constraints(self):
        """ return a list of tuples (s,t,v) where s is the switch
        this item of the assignment will delegate to, t is the time slot
        and v is the demand that will be delegated"""
        result = []
        idx = 0
        for t, job in self.vjob.jobs.items():
            new_tuple = (self.assignment[idx], t, job.demand)
            result.append(new_tuple)
            idx += 1
        return result              

class VJob(object):

    def __init__(self, rss, dstats):
        self.rss = rss # previously vnet; core rss data structure (gives access to global data)
        self.jobs = dstats
        self.job_id = None
        self.switch = None
        ticks = []
        for k, dstat in dstats.items():
            ticks.append(k)
            if not self.switch:
                self.switch = dstat.switch
            if not self.job_id:
                self.job_id = dstat.job_id
            assert(dstat.switch == self.switch)
            assert(dstat.job_id == self.job_id)
        self.assignments = []
        self.assignments_by_id = {}
        self.ticks = ticks
        self.start = min(ticks)
        self.end = max(ticks)
        self.length = len(ticks)

    def __lt__(self, other):
        """required because vjobs are added to a minheap"""
        return self.job_id < other.job_id

    def iterate(self, start, stop, data, lvl, datamap, thresh):
        try:
            datamap[lvl].append((start, stop))
        except KeyError:
            datamap[lvl] = [(start, stop)]
        if stop-start > thresh:
            split = int(start + (stop-start)/2)
            self.iterate(start, split, data, lvl+1, datamap, thresh)
            self.iterate(split+1, stop, data,  lvl+1, datamap, thresh)

    def update_ratings_default(self, assignment):
        PENALITY_FLOW_MIGRATION = 5
        rating = []
        idx = 0
        for t, job in self.jobs.items():
            try:
                switch = assignment[idx]
            except IndexError as e:
                logger.warn('self.jobs (len=%d)' % len(self.jobs))
                logger.warn('assignment (len=%d) ' % len(assignment))
                raise e
            
            if switch == 9999:
                rating.append(100)
            else:
                delegate = job.util_raw - job.util
                # minimization problem, so the values are stored as 
                # negative values (except for the value of ES which
                # is stored as a static positive)
                try:
                    rating.append((self.rss.free_capacity[switch][t] - delegate)*-1)
                except KeyError as e:
                    print("KeyError", "switch=", switch, "t=", t)
                    print("assignment", assignment)
                    print("self.rss.free_capacity", self.rss.free_capacity)
                    raise e


            # add a penalty if the remote switch is changed within the assignment,
            # i.e. [s1, s1, s2, s2, s1, s1] will receive 2 penalties for (s1,s2) and (s2,s1)
            if idx > 0:
                if assignment[idx] is not assignment[idx-1]:
                    rating.append((job.util_raw - job.util)* PENALITY_FLOW_MIGRATION)

            idx += 1

        return sum(rating)

    def get_assignment(self, id):
        return self.assignments_by_id[id]

    def update_ratings_iterative2(self, tick, history_remote_utilization_used, 
        history_remote_from, history_allocation, history_relocated, history_util_free,
        history_link_used, buckets,
        w_table, w_link, w_ctrl, w_backup, export_ratings=False):

        #print("--- t=%d job=%d" % (tick, self.job_id))

        stats_ratings = []
        stats_table = []
        stats_ctrl = []
        stats_link = []
        stats_static = []

        best_rating = 1000000000000

        for v_assignment in self.assignments:  
            assignment = v_assignment.assignment
            rating = []
            idx = 0

            rating_table = 0
            rating_ctrl = 0
            rating_link = 0
            rating_static = 0

            debug_rating_table = []
            debug_rating_ctrl = []
            debug_rating_link = []
            debug_rating_static = []
            
            last_allocation = history_allocation.get(self.job_id)
            last_link = (self.switch, last_allocation)
                   
            # go trough all the switches in the assignment; this is done with 
            # self.jobs because this provides access to the dstat objects where
            # the utilization is stored
            for t, dstat in self.jobs.items():
        
                remote_switch = assignment[idx]
                link = (self.switch, remote_switch)
                assert(remote_switch >= 0)

                my_relocated_rules = dstat.util_raw - dstat.util   
                my_relocated_demand = dstat.demand

                my_relocated_rules_last_tick = history_remote_from.get(dstat.port_label, 0)
                my_relocated_demand_last_tick = history_relocated.get(last_link, 0)
                
                demand_diff = my_relocated_demand-my_relocated_demand_last_tick
                rules_diff =  my_relocated_rules-my_relocated_rules_last_tick 

                # ctrl overhead cost
                rules_moved = 0
                if remote_switch != last_allocation:
                    rules_moved = max(0, my_relocated_rules_last_tick) + max(0, my_relocated_rules)
                rating_ctrl = w_ctrl * rules_moved  
                rating_ctrl = rating_ctrl / (self.rss.ctx.statistics['scenario.table_util_max_total_per_port'] * 2)
                rating.append(rating_ctrl)
                
                if remote_switch == 9999:
                    # static cost for virtual backup switch
                    rating_static = w_backup
                    rating.append(rating_static)
                else:
                    # link overhead cost
                    rating_link = w_link * min(self.rss.link_bandwidth, history_link_used.get(link) + my_relocated_demand )
                    rating_link = rating_link / self.rss.link_bandwidth
                    rating.append(rating_link)
                              
                    # table overhead cost
                    table_free_new = history_util_free.get(remote_switch) - my_relocated_rules
                    rating_table = (-1 * table_free_new * w_table ) / self.rss.ctx.statistics['scenario.table_util_max_total']
                    rating.append(rating_table)
                
                if export_ratings:
                    debug_rating_link.append(rating_link)  
                    debug_rating_ctrl.append(rating_ctrl)
                    debug_rating_table.append(rating_table)
                    debug_rating_static.append(rating_static)

                # prepare for next tick
                idx += 1
                last_allocation = remote_switch
                last_link = link

            # update the rating in the v_assignment object
            rating_total = sum(rating)
            v_assignment.rating = rating_total

            if export_ratings:
                print("")
                print("  ", assignment, " ===>", v_assignment.rating)
                print("    table :", [round(x,2) for x in debug_rating_table])
                print("    ctrl  :", [round(x,2) for x in debug_rating_ctrl])
                print("    link  :", [round(x,2) for x in debug_rating_link])
                print("    static:", debug_rating_static)
                stats_ratings.append(v_assignment.rating)
                stats_table +=  debug_rating_table
                stats_link +=  debug_rating_link
                stats_ctrl +=  debug_rating_ctrl
                stats_static += debug_rating_static

        if export_ratings:
            try:
                summary_ratings = (min(stats_ratings), max(stats_ratings), statistics.mean(stats_ratings))
            except:
                summary_ratings = (0,0,0)
            try:
                summary_table = (min(stats_table), max(stats_table), statistics.mean(stats_table))
            except:
                summary_table = (0,0,0)
            try:
                summary_link = (min(stats_link), max(stats_link), statistics.mean(stats_link))
            except:
                summary_link = (0,0,0)
            try:
                summary_ctrl = (min(stats_ctrl), max(stats_ctrl), statistics.mean(stats_ctrl))
            except:
                summary_ctrl = (0,0,0)
            try:
                summary_static = (min(stats_static), max(stats_static), statistics.mean(stats_static))
            except:
                summary_static = (0,0,0)
            return (summary_ratings, summary_table, summary_link, summary_ctrl, summary_static)

    def update_ratings_iterative(self, last_decision):

        for v_assignment in self.assignments:  
            assignment = v_assignment.assignment
            rating = []
            idx = 0
            for t, job in self.jobs.items():
                try:
                    switch = assignment[idx]
                except IndexError as e:
                    logger.warn('self.jobs (len=%d)' % len(self.jobs))
                    logger.warn('assignment (len=%d) ' % len(assignment))
                    raise e

                # check history
                modifier = 0
                for last_job, last_t, last_switch in last_decision: 
                    if last_job.job_id == job.job_id:
                        #logger.info("  there was an old value %d %d %s" % (last_job.job_id, last_t, last_switch.label))
                        # there was already a decision for this job
                        if switch == last_switch:
                            # the same as last one --> better
                            modifier -= 200

                idx += 1
                if switch == 9999:
                    rating.append(100 + modifier)
                else:
                    delegate = job.util_raw - job.util
                    # minimization problem, so the values are stored as 
                    # negative values (except for the value of ES which
                    # is stored as a static positive) 
                    rating.append(((self.rss.free_capacity[switch][t] - delegate)*-1)+modifier)

            # update the rating in the v_assignment object
            v_assignment.rating = sum(rating)

    def create_assignments(self):

        ts = time.time()
        ab = AssignmentBuilder(max_assignments=self.rss.ctx.config.get('param_rsa_max_assignments'))
        switches = {}
        free = {}
        demand = []
        #free[60] = [] # backup switch
        for sw in self.rss.ctx.scenario.g.nodes():
            switches[sw] = sw 
            free[sw] = []
        for t, dstat in self.jobs.items():
            #ree[60].append(250)
            demand.append(dstat.util_raw - dstat.util)
            for sw in self.rss.ctx.scenario.g.nodes():
                if sw in dstat.es_options:
                    free[sw].append(self.rss.free_capacity[sw][t])
                else:
                    free[sw].append(-9)

        ab.add_demand(demand)
        for sw, data in free.items():
            ab.add_switch(sw, data)
        
        assignments_by_label = ab.run()
        assignments = []
        for a in assignments_by_label:
            a2 = []
            for switch in a:
                # the assignment builder still works with labels instead of integer ids,
                # not changed here for compatability ('ES' is translated to 9999, the other
                # ids work fine)
                if switch == 'ES':
                    a2.append(9999)
                else:   
                    a2.append(switch)
            assignments.append(a2)

        self.rss.statistics_append('rsa.solver.stats_time_create_assignments', time.time()-ts)
        self.rss.statistics_append('rsa.solver.jobs_assignments_lengths', len(assignments))

        for assignment in assignments:  

            #logger.info("len=%d" % len(assignment))
            vAssignment = VJobAssignment(self, len(self.assignments), assignment)
            self.assignments_by_id[vAssignment.id] = vAssignment
            self.assignments.append(vAssignment)

            rating = self.update_ratings_default(assignment)  
            vAssignment.rating = rating
            #logger.info([sw.label for sw in assignment])
            #logger.info('  update_ratings_default: %d' % rating)
   
class RSSSolver:

    def __init__(self, ctx):

        self.ctx = ctx
        self.timer = Timer(self.ctx, 'rsa.solver')
        self.timelimit = self.ctx.config.get('param_debug_total_time_limit', -1)
        self.test_utils = None # for debugging
        self.maxtick = 449
        self.look_ahead = self.ctx.config.get('param_rsa_look_ahead')
        self.link_bandwidth = 1000000000 # 1gbit/s as default

        self.w_table = self.ctx.config.get('param_rsa_weight_table')
        self.w_link = self.ctx.config.get('param_rsa_weight_link')
        self.w_ctrl = self.ctx.config.get('param_rsa_weight_ctrl')
        self.w_backup = self.ctx.config.get('param_rsa_weight_backup')
        # ---------
        # data structures required by the algorithms
        # ---------
        self.switches = [] # a list of all switches consered by this RSS problem, including the generic "backup extension switch"
        self.vjobs = [] # global list of all vjobs (i.e., for all time slots / the whole scenario)

        self.dss = {} # dss[s][t][p] -> dstat item for switch s time slot t and port p
        self.free_util = {}  # free_util[s1][s2][t] -> free link capacity between s1 and s2 at time slot t
        self.free_capacity = {} # free_capacity[s][t] -> free table capacity at s at time slot t
        self.active_jobs = {} # stores a tuple (job_cnt, util_cnt) per time slot
        
        self.ctx.statistics['rsa.solver.cnt_timelimit'] = 0
        self.ctx.statistics['rsa.solver.cnt_infeasable'] = 0
        self.ctx.statistics['rsa.solver.cnt_feasable'] = 0


    def run(self):
        self.split_traffic()          

    def get_cacheid(self, switch):
        """
        DSS is executed seperately for each switch here. DSS results are stored
        in a cache folder. The uuid to access the cache consists of the main
        input paramters (solver, objective etc) and a hash over the flows. Only the exact
        same traffic profile will result in the same uuid and thus a cache hit!
        """
        solver = self.ctx.config.get('param_dts_algo')
        look_ahead = self.ctx.config.get('param_dts_look_ahead')
        skip_ahead = self.ctx.config.get('param_dts_skip_ahead')
        threshold = self.ctx.config.get('param_topo_switch_capacity')
        obj = self.ctx.config.get('param_dts_objective')

        uuid_prefix = 'switch-%s-s%d-o%d-L%d-S%d-c%d' % (
            switch.label, solver, obj, look_ahead, skip_ahead, threshold)

        m = hashlib.md5()      
        m.update(uuid_prefix.encode('utf-8'))  
        rawhash = ''
        for flow in self.ctx.flows:
            # all important flow generation parameters are included in the hash
            # (start, demand, duration, flowlabel); if only one parameter is changed,
            # the hash gets invalid and the dss results are recalculated
            h = '%s%.4f%.4f%.4f' % (
                flow.label,flow.start, flow.demand_per_tick,flow.duration)
            m.update(h.encode('utf-8'))
            rawhash += h
        return str(m.hexdigest())

    def statistics_append(self, name, value):
        """
        Helper function or statistics.
        """
        if not self.ctx.statistics.get(name):
            self.ctx.statistics[name] = []
        self.ctx.statistics[name].append(value)

    def statistics_model(self, m):
        # add some statistics for a solved model
        self.statistics_append('rsa.solver.stats_time_solving', m.Runtime)
        self.statistics_append('rsa.solver.stats_infeasible', m.status == 3)
        self.statistics_append('rsa.solver.stats_timelimit', m.status == 9)
        self.statistics_append('rsa.solver.stats_itercount', m.Itercount)
        self.statistics_append('rsa.solver.stats_nodecount', m.Nodecount)
        try:
            self.statistics_append('rsa.solver.stats_gap', m.MIPGap)
        except:
            self.statistics_append('rsa.solver.stats_gap', 0) 

        #print("solve model", m.status, m.Runtime, m.Itercount, m.Baritercount, m.Nodecount)
        # parse results from printStats (there seem to be no other way to get this...)
        stats = io.StringIO()
        with redirect_stdout(stats):
            m.printStats()
        data = stats.getvalue().split()
        self.ctx.statistics['rsa.solver.stats_linear_constraint_matrix'] = int(data[9])
        self.ctx.statistics['rsa.solver.stats_constraints'] = int(data[11])
        self.ctx.statistics['rsa.solver.stats_nz_variables'] = int(data[13])
        if m.status == 2: self.ctx.statistics['rsa.solver.cnt_feasable'] += 1
        if m.status == 3: self.ctx.statistics['rsa.solver.cnt_infeasable'] += 1
        if m.status == 9: self.ctx.statistics['rsa.solver.cnt_timelimit'] += 1

    def run_dss(self, switch):


        print("run dts from scenario file")
        # create new ctx for a sub-problem with only one switch
        new_ctx = Context()

        # copy existing context from global problem
        new_ctx.config = self.ctx.config.copy()
        new_ctx.scenario = self.ctx.scenario
        new_ctx.started = self.ctx.started #  for time checks
        # some basic statistics are required in DTS for normalization
        for k, v in self.ctx.statistics.items():
            if k.startswith('scenario'):
                new_ctx.statistics[k] = v   
        
        # prepare data object for dts solver
        new_ctx.scenario_data =  DTSData(self.ctx, switch)
        new_ctx.scenario_data.calculate_raw_utils()

        dts_solver = DTSSolver(new_ctx, link_bandwidth=self.link_bandwidth)
        new_ctx.stored_solution = dts_solver.run(new_ctx)
        if new_ctx.stored_solution == None:
            raise RuntimeError("stop here (no return from solver = abort)")
        # store solution 
        self.ctx.all_commands[switch] = new_ctx.stored_solution
        # take over statistics from new_ctx
        for k,v in new_ctx.statistics.items():
            self.ctx.statistics['dts.%d.%s' % (switch, k)] = v


        if self.ctx.config.get('param_debug_verify_with_simulator') == 1:
            print("")
            print("run simulator (param_debug_verify_with_simulator=1)")
            topo = TopologyFromDTSResult(new_ctx, switch)
            sim = Simulator(new_ctx)
            sim.run()

        return new_ctx, new_ctx.scenario_result

    # ==========================
    # ALGO 1 (Non-periodic RSA algorithm)
    # ==========================

    def algo1_wrapper(self):
        stime = time.time()
        m = Model("solver_rss")
        X = {}
        objective_function = []
        constraint_free_capacity = {}

        demand_constraints = {}
        demand_constraint_mapping = {}

        for vj in self.vjobs:
            j = vj.job_id
            X[j] = {}

            constraint_pick_one_assignment = []
            for assignment in vj.assignments:
                a = assignment.id
                X[j][a] = m.addVar(vtype=GRB.BINARY, name='x_%d_%d' % (j, a))
                constraint_pick_one_assignment.append(X[j][a])

                objective_function.append(X[j][a] * assignment.rating)

                # handle constraints of the remote switch flow table
                for s, t, v in assignment.get_delegations():
                    # v is the amount of rules that is delegated towards switch s at time slot t
                    if not constraint_free_capacity.get(s):
                        constraint_free_capacity[s] = {}
                    if not constraint_free_capacity[s].get(t):
                        constraint_free_capacity[s][t] = []
                    constraint_free_capacity[s][t].append(X[j][a] * v)

                # handle link capacity constraints between delegation and remote switch
                for s, t, v in assignment.get_link_demand_constraints():
                    # it is assumed here that the capacity to the backup extension switch is not limited,
                    # i.e., this part is excluded
                    if s != 9999:
                        s_from = vj.switch
                        #link_key = '%s->%s' % (, s.label)
                        if not demand_constraints.get(t):
                            demand_constraints[t] = {}
                        if not demand_constraints[t].get(s_from):
                            demand_constraints[t][s_from] = {}
                        if not demand_constraints[t][s_from].get(s):
                            demand_constraints[t][s_from][s] = []
                        demand_constraints[t][s_from][s].append(X[j][a] * v)


            # make sure one of the assignments is picked
            m.addConstr(quicksum(constraint_pick_one_assignment) == 1)

        for s, data in constraint_free_capacity.items():
            for t, list_x in data.items():
                m.addConstr(quicksum(list_x) <= self.free_capacity[s][t])

        # add link demand constraints for all jobs in this time slot
        for t, data in demand_constraints.items(): # time slot considered
            for s1, data2 in data.items(): # demand sent from s1
                for s2, coefficients in data2.items(): # towards s2
                    assert(self.ctx.scenario.g.has_edge(s1, s2))
                    if len(coefficients) > 0:
                        try:
                            free = self.free_util[s1][s2][t]   
                        except KeyError as e:
                            print("KeyError")
                            print("  s1", s1)
                            print("  s2", s2)
                            print("  t", t)
                            raise e

                        try:
                            free2 = self.free_util[s2][s1][t]  
                        except KeyError as e:
                            print("KeyError")
                            print("  s1", s1)
                            print("  s2", s2)
                            print("  t", t)
                            raise e

                        
                        logger.debug("add constr %d->%d coefficients=%d free=%f | %f" % (s1, 
                            s2, len(coefficients), self.link_bandwidth  - free, self.link_bandwidth -free2))
                        # this can only happen if there is something wrong with the traffic generator script 
                        # is taken care of with the max() structure below, so printing a warning should be sufficient
                        if self.link_bandwidth  - free < 0:
                            logger.info("RSS-WARNING: negative free link capacity at %d->%d [%f]" % (
                                s1, s2, self.link_bandwidth  - free2))
                        if self.link_bandwidth  - free2 < 0:
                            logger.info("RSS-WARNING: negative free link capacity at %d->%d [%f]" % (
                                s2, s1, self.link_bandwidth  - free2))

                        m.addConstr(quicksum(coefficients) <= max(self.link_bandwidth  - free, 0))
                        m.addConstr(quicksum(coefficients) <= max(self.link_bandwidth  - free2, 0))


        m.setObjective(quicksum(objective_function), GRB.MINIMIZE)
        self.statistics_append('rsa.solver.stats_time_modeling', time.time()-stime)

        stime = time.time()
        m.setParam('TimeLimit', self.ctx.config.get('param_rsa_timelimit'))
        m.optimize()
        self.statistics_append('rsa.solver.stats_time_solving2', time.time()-stime)

        self.statistics_model(m)

        if m.status == 9:
            self.ctx.statistics['rsa.solver.cnt_timelimit'] += 1 
            # get results from the solved optimization problem
            for v in m.getVars():
                if v.varname[0] == 'x':
                    j = int(v.varname.split('_')[1])
                    a = int(v.varname.split('_')[2])
                    if v.x > 0.5:
                        #print(j, s)
                        for vj in self.vjobs:
                            if vj.job_id == j:
                                assignment = vj.get_assignment(a).assignment
                                logger.info("found solution j=%d a=%d  %s" % (j, a, 
                                    str([sw for sw in assignment])))

                                idx = 0
                                for t, job in vj.jobs.items():
                                    job.es_switch = assignment[idx]
                                    job.es = job.es_switch
                                    idx+=1
            return

        if m.status == 3:
            # infeasable or timelimit
            logger.info("! model was infeasable")
            return None
        else:
            # get results from the solved optimization problem
            for v in m.getVars():
                if v.varname[0] == 'x':
                    j = int(v.varname.split('_')[1])
                    a = int(v.varname.split('_')[2])
                    if v.x > 0.5:
                        #print(j, s)
                        for vj in self.vjobs:
                            if vj.job_id == j:
                                assignment = vj.get_assignment(a).assignment
                                logger.info("found solution j=%d a=%d  %s" % (j, a, 
                                    str([sw for sw in assignment])))

                                idx = 0
                                for t, job in vj.jobs.items():
                                    job.es_switch = assignment[idx]
                                    job.es = job.es_switch
                                    idx+=1

    # ==========================
    # ALGO 2 (Periodic RSA algorithm)
    # ==========================

    def algo2_wrapper(self, DTS):
        """Iterative RSS algorithm"""
        
        export_ratings = self.ctx.config.get('param_debug_export_rsa_ratings') == 1
        # run iterative solver
        all_results = []

        history_allocation = {} # <job_id> -> remote switch
        history_relocated = {} # <delegation_switch, remote_switch> -> relocated demand at this link
        history_remote_from = {}
        history_remote_utilization_used = {}  
        history_link_used = {} #  <delegation_switch, remote_switch> -> used total link capacity

        for tick in range(0, self.maxtick - self.look_ahead - 2):
            #logger.info("** tick=%d" % tick)
            # a list with local jobs (only for the part selected by tick and look-ahead)
            jobs = {}
            stime = time.time()
            self.check_for_time_limit()
            """
            for switch, dts_result in DTS.items():
                for t in range(tick, tick+self.look_ahead):
                    for port, dstat in dts_result.delegation_status.get(t).items():
                        if dstat.status == 1:
                            if not jobs.get(dstat.job_id):
                                jobs[dstat.job_id] = {}
                            jobs[dstat.job_id][t] = dstat

            for job_id, joblist in jobs.items():
                print('  job=%d len=%d ' % (job_id, len(joblist)))
            """
            jobs = {}
            for vj in self.vjobs:
                if tick in vj.ticks:
                    jobs[vj.job_id] = {}
                    for usetick in range(tick, tick+self.look_ahead):
                        dstats = vj.jobs.get(usetick)    
                        if dstats:
                            jobs[vj.job_id][usetick] = dstats

            # the first run needs history_util_free; the other updates are done below because
            # history_remote_utilization_used has to be considered (similar for link utilization)
            if len(jobs) > 0 and len(history_remote_utilization_used) == 0:
                history_util_free = {} # <switch> -> free flow table capacity
                for switch in self.ctx.scenario.g.nodes():
                    history_util_free[switch] = self.free_capacity[switch][tick]
            if len(jobs) > 0 and len(history_relocated) == 0:
                history_link_used = {} #  <delegation_switch, remote_switch> -> used total link capacity
                for s1, s2 in self.ctx.scenario.g.edges():
                    history_link_used[(s1, s2)] = self.free_util[s1][s2][tick]
                    history_link_used[(s2, s1)] = self.free_util[s2][s1][tick]


            # create local vjob objects
            all_rating_summaries = []
            buckets = {}
            vjobs = []
            for j, joblist in jobs.items():
                vj = VJob(self, joblist) 
                vj.create_assignments() 
                rating_summary = vj.update_ratings_iterative2(tick, 
                    history_remote_utilization_used, history_remote_from, history_allocation,
                    history_relocated, history_util_free, history_link_used, buckets,
                    self.w_table, self.w_link, self.w_ctrl, self.w_backup, export_ratings=export_ratings)

                if export_ratings:
                    all_rating_summaries.append(rating_summary)

                #logger.info('assignment length = %d' % len(vj.assignments))      
                vjobs.append(vj)

            if export_ratings:
                # add rating summary for this tick
                self.export_ratings(all_rating_summaries, tick)

            # balancing assignments
            balancing_cnt = 5
            balancing_weights = {}
            balancing_step = 0
            if len(vjobs) > 0:
                normalize = self.ctx.statistics['scenario.rules_per_switch_avg']
                for t in range(0, self.look_ahead):
                    #print("balancing_weights t=%d" % (tick+t))
                    free_all = []
                    # collect all free capacities in this time slot
                    for switch in self.ctx.scenario.g.nodes():
                        free = self.free_capacity[switch][tick+t]
                        #print("free", tick+t, switch, free)
                        if free > 0:
                            free_all.append(free)
                    try:
                        maxfree = max(free_all)
                        balancing_step = maxfree/balancing_cnt
                    except:
                        balancing_step = -1
                    if balancing_step > 0:
                        weights = []
                        for x in range(1, balancing_cnt+1):
                            factor = sum([1 for s in free_all if s > x*balancing_step])
                            weight = -1*(self.w_table * balancing_step * (balancing_cnt-x) * factor) / normalize
                            #print(x, balancing_step, factor, maxfree, normalize, weight)
                            weights.append(weight)
                        balancing_weights[tick+t] = weights


            self.statistics_append('rsa.solver.stats_time_modeling', time.time()-stime)

            stime = time.time()
            results = self.algo2_wrapper_runmodel(vjobs, tick, balancing_weights, balancing_step)
            self.statistics_append('rsa.solver.stats_time_solving2', time.time()-stime)

            all_results.append(results)

            if len(results) > 0:

                history_util_free = {} # <remote_switch> -> free flow table capacity
                history_link_used = {} #  <delegation_switch, remote_switch> -> used total link capacity
                history_allocation = {} # <job_id> -> remote switch
                history_relocated = {} # <delegation_switch, remote_switch> -> relocated demand at this link
                history_remote_from = {}
                history_remote_utilization_used = {} 

                #print(tick, "== update history")

                for dstat, t, assigned_switch in results: 
                    if t == tick: # only "first" result is important
                        history_allocation[dstat.job_id] = assigned_switch
                        relocated = dstat.util_raw-dstat.util
                        demand_relocated = dstat.demand
                        if math.isclose(demand_relocated, 0):
                            demand_relocated = 0
                        if demand_relocated > 0:
                            try:
                                history_relocated[(dstat.switch, assigned_switch)] += demand_relocated
                            except KeyError:
                                history_relocated[(dstat.switch, assigned_switch)] = demand_relocated

                        if relocated > 0:
                            try:
                                history_remote_utilization_used[assigned_switch] += relocated
                            except KeyError:
                                history_remote_utilization_used[assigned_switch] = relocated

                            try:
                                history_remote_from[dstat.port_label] += relocated
                            except KeyError:
                                history_remote_from[dstat.port_label] = relocated 

                for switch in self.ctx.scenario.g.nodes():
                    history_util_free[switch] = self.free_capacity[switch][tick] - history_remote_utilization_used.get(switch, 0)

                for s1, s2 in self.ctx.scenario.g.edges():
                    history_link_used[(s1, s2)] = self.free_util[s1][s2][tick] + history_relocated.get((s1,s2), 0)
                    history_link_used[(s2, s1)] = self.free_util[s2][s1][tick] + history_relocated.get((s2,s1), 0)
                """
                print(history_remote_utilization_used)
                print(history_remote_from)
                print("history_allocation", history_allocation)
                print("history_relocated", history_relocated)
                print("history_util_free", history_util_free)
                print("history_link_used", history_link_used)
                """



        # update the global vjobs that are used
        for results in all_results:
            for job, t, assigned_switch in results:
                for vj in self.vjobs:
                    if vj.job_id == job.job_id:
                        for t2, job2 in vj.jobs.items():
                            if t==t2:
                                #logger.info(' ===> result %d : %d -> %s' % (job.job_id, t, assigned_switch.label))
                                job2.es_switch = assigned_switch
                                job2.es = job2.es_switch
        
    def algo2_wrapper_runmodel(self, vjobs, tick, balancing_weights, balancing_step):
        m = Model("algo2_wrapper_runmodel")
        X = {}
        objective_function = []
        constraint_free_capacity = {}

        demand_constraints = {}
        demand_constraint_mapping = {}

        for vj in vjobs:
            j = vj.job_id
            X[j] = {}

            constraint_pick_one_assignment = []
            for assignment in vj.assignments:
                a = assignment.id
                X[j][a] = m.addVar(vtype=GRB.BINARY, name='x_%d_%d' % (j, a))
                constraint_pick_one_assignment.append(X[j][a])

                objective_function.append(X[j][a] * assignment.rating)

                # handle constraints of the remote switch flow table
                for s, t, v in assignment.get_delegations():
                    # v is the amount of rules that is delegated towards switch s at time slot t
                    if not constraint_free_capacity.get(s):
                        constraint_free_capacity[s] = {}
                    if not constraint_free_capacity[s].get(t):
                        constraint_free_capacity[s][t] = []
                    constraint_free_capacity[s][t].append(X[j][a] * int(v))

                # handle link capacity constraints between delegation and remote switch
                for s, t, v in assignment.get_link_demand_constraints():
                    # it is assumed here that the capacity to the backup extension switch is not limited,
                    # i.e., this part is excluded; in other words: the solver can always assign 9999 (the backup
                    # switch) but has to "accept" the high assignment.rating value then
                    if s != 9999:
                        s_from = vj.switch
                        #link_key = '%s->%s' % (, s.label)
                        if not demand_constraints.get(t):
                            demand_constraints[t] = {}
                        if not demand_constraints[t].get(s_from):
                            demand_constraints[t][s_from] = {}
                        if not demand_constraints[t][s_from].get(s):
                            demand_constraints[t][s_from][s] = []
                        demand_constraints[t][s_from][s].append(X[j][a] * int(v))


            # make sure one of the assignments is picked
            m.addConstr(quicksum(constraint_pick_one_assignment) == 1)

        ObjBalanceTable = []
        for s, data in constraint_free_capacity.items():
            for t, list_x in data.items():

                balancing_table = [0]
                if balancing_weights.get(t):
                    for i, weight in enumerate(balancing_weights.get(t)):
                        x_table = m.addVar(vtype=GRB.BINARY, name='balance_%d_%d_%d' % (s, t, i))
                        balancing_table.append(x_table * balancing_step)
                        ObjBalanceTable.append(x_table * weight)
                # self.free_capacity[s][t] is not supposed to be smaller than 0 which
                # can happen in very rare cases; however, using a negative value here can
                # lead to an infeasible model (try param_topo_scenario_generator=2 and
                # param_topo_seed=1461 and param_dts_algo=3 to create such a rare case)
                # use model.write('out.lp') to see what happens; 
                # Problematic rule in this example is  R80: - x_2000043_3 <= -1
                m.addConstr(quicksum(list_x) + quicksum(balancing_table) <= max(0, int(self.free_capacity[s][t])))

        # add link demand constraints for all jobs in this time slot
        
        for t, data in demand_constraints.items(): # time slot considered
            for s1, data2 in data.items(): # demand sent from s1
                for s2, coefficients in data2.items(): # towards s2
                    assert(s1 != 9999)
                    assert(s2 != 9999)
                    if len(coefficients) > 0:
                        free = self.free_util[s1][s2][t]   
                        free2 = self.free_util[s2][s1][t] 
                        #print("add constr %d->%d coefficients=%d free=%f | %f" % (s1, 
                        #    s2, len(coefficients), max(self.link_bandwidth  - free, 0), max(self.link_bandwidth  - free2, 0)))
                        m.addConstr(quicksum(coefficients) <= int(max(self.link_bandwidth  - free, 0)))
                        m.addConstr(quicksum(coefficients) <= int(max(self.link_bandwidth  - free2, 0)))


        m.setObjective(quicksum(objective_function) + quicksum(ObjBalanceTable), GRB.MINIMIZE)


        logging.disable(logging.INFO);
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', self.ctx.config.get('param_rsa_timelimit'))
        status = m.optimize()
        logging.disable(logging.NOTSET);

        self.statistics_model(m)
   
        if m.status == 3:
            # infeasable or timelimit
            logger.info("ERROR: model for RSA was infeasable; this should never happen")
            logger.info("An error file (error-rsa.lp) was written to disk with the to-be-solved model")
            m.write("error-rsa-%f.lp" % (time.time()))
            raise RuntimeError('RSA model got infeasible')

        # get results from the solved optimization problem
        results = []
        for v in m.getVars():
            if v.varname[0] == 'x':
                j = int(v.varname.split('_')[1])
                a = int(v.varname.split('_')[2])
                if v.x > 0.5:
                    #print(j, s)
                    for vj in vjobs:
                        if vj.job_id == j:
                            assignment = vj.get_assignment(a).assignment
                            print("--- t=%d j=%d a=%d  %s" % (tick, j, a, 
                                str([sw for sw in assignment])))

                            idx = 0
                            for t, job in vj.jobs.items():
                                #job.es_switch = assignment[idx]
                                #job.es = job.es_switch.label   
                                #logger.info(' -> set %d@%d to %s' % (job.job_id, t, job.es))    
                                results.append((job, t, assignment[idx]))
                                idx += 1

        return results


    # ==========================
    # ALGO 3 (FirstFit)
    # ==========================

    def algo3_wrapper(self):
        """FirstFit RSS algorithm"""

        # run iterative solver
        all_results = []
        for tick in range(0, self.maxtick - self.look_ahead - 2):
            logger.debug("** tick=%d" % tick)
            stime = time.time()
            # a list with local jobs (only for the part selected by tick and look-ahead)
            jobs = {}
            has_job = False
            for switch in self.switches:
                if not self.dss.get(switch): continue;
                for t in range(tick, tick+self.look_ahead):
                    for port, dstat in self.dss[switch][t].items():
                        if dstat.status == 1:
                            has_job = True
                            if not jobs.get(dstat.job_id):
                                jobs[dstat.job_id] = {}
                            jobs[dstat.job_id][t] = dstat

            if not has_job:
                continue;

            for job_id, joblist in jobs.items():
                logger.debug('  job=%d len=%d ' % (job_id, len(joblist)))

            # create local vjob objects
            vjobs = []
            for j, joblist in jobs.items():
                vj = VJob(self, joblist) 
                vj.create_assignments() 
                if len(all_results) > 0:
                    vj.update_ratings_iterative(all_results[-1])
                #logger.info('assignment length = %d' % len(vj.assignments))      
                vjobs.append(vj)
            self.statistics_append('rsa.solver.stats_time_modeling', time.time()-stime)

            stime = time.time()
            results = self.algo3_wrapper_runmodel(tick, vjobs)
            self.statistics_append('rsa.solver.stats_time_solving2', time.time()-stime)
            all_results.append(results)

        # update the global vjobs that are used
        for results in all_results:
            for job, t, assigned_switch in results:
                for vj in self.vjobs:
                    if vj.job_id == job.job_id:
                        for t2, job2 in vj.jobs.items():
                            if t==t2:
                                #logger.info(' ===> result %d : %d -> %s' % (job.job_id, t, assigned_switch.label))
                                job2.es_switch = assigned_switch
                                job2.es = job2.es_switch.label
    
    def algo3_wrapper_runmodel(self, tick, vjobs):
        sorted_jobs = []
        for vj in vjobs:
            # handle job with the highest weight first; note that the weights are the same for
            # all assignments because they only consist of the amount of rules that have to be delegated and
            # the amount of traffic that has to be redirected (independent from the potential remote switch that
            # defines an assignment);
            # the weight is calculated here as "amount delegated per rule" -> jobs with rules that carry 
            # less traffic will will have a lower weight and will be selected frist from the minheap; 
            util, demand = vj.assignments[0].get_weights()
            weight = (demand/(util + random.random()/1000000 )) # random value will avoid having two tuples with the same weight
            if weight == 0:
                weight = 1000000  + random.random()/1000000 
            heappush(sorted_jobs, (weight, vj))

        cnt = 0
        results = []
        assigned_to = {} # debug counter
        assigned_util = {} # every time a job is assigned to one of the switches, the used capacity is added here
        assigned_demand = {}
        while len(sorted_jobs) > 0:
            util, job = heappop(sorted_jobs)
            logger.debug("  # handle job %d weight=%d" % (job.job_id, util))
            
            # the assignments will be sorted by their rating (negative numbers are 
            # better so the default sort order is ok)
            for i, assignment in enumerate(sorted(job.assignments, key=lambda x: x.rating)):
                cnt += 1
                logger.debug("  -> assignment %d" % (i))
                # handle constraints of the remote switch flow table
                can_be_assigned = True

                # check free table capacity
                for s, t, v in assignment.get_delegations():
                    if not assigned_util.get(s):
                        assigned_util[s] = {}
                    if not assigned_util[s].get(t):
                        assigned_util[s][t] = 0

                    # v is the amount of rules that is delegated towards switch s at time slot t
                    if self.free_capacity[s][t] - assigned_util[s][t] >= v:
                        logger.debug('     check table_capacity OK rating=%f s=%s %d/%d' % (assignment.rating, s.label, 
                            v, self.free_capacity[s][t] - assigned_util[s][t]))
                    else:
                        logger.debug('     check table_capacity FAILED rating=%f s=%s %d/%d' % (assignment.rating, s.label, 
                            v, self.free_capacity[s][t] - assigned_util[s][t]))        
                        can_be_assigned = False
                        break; # leave table capacity loop

                # skip link check if table check failed
                if not can_be_assigned:
                    continue

                # check free link capacity 
                for s, t, v in assignment.get_link_demand_constraints():
                    if not assigned_demand.get(job.switch):
                        assigned_demand[job.switch] = {}
                    if not assigned_demand[job.switch].get(s):
                        assigned_demand[job.switch][s] = {}
                    if not assigned_demand[job.switch][s].get(t):  
                        assigned_demand[job.switch][s][t] = 0
                    if s.label != 'ES':
                        # delegated traffic will be on both directions of the link
                        free1 = self.link_bandwidth  - (self.free_util[job.switch][s][t] + assigned_demand[job.switch][s][t])
                        free2 = self.link_bandwidth  - (self.free_util[s][job.switch][t] + assigned_demand[job.switch][s][t])
                        if free1 >= v and free2 >= v:
                            logger.debug('     check link_capacity %s->%s OK rating=%f s=%s %d/%d (%d)' % (
                                job.switch.label, s.label, assignment.rating, s.label, 
                                v, min(free1, free2), assigned_demand[job.switch][s][t]))
                        else:
                            logger.debug('     check link_capacity %s->%s FAILED rating=%f s=%s %d/%d (%d)' % (
                                job.switch.label, s.label, assignment.rating, s.label, 
                                v, min(free1, free2), assigned_demand[job.switch][s][t]))                        
                            can_be_assigned = False 
                            break; # leave link capacity loop

                if can_be_assigned:
                    logger.debug("  # set job %d to assignment %d" % (job.job_id, i))
                    idx = 0
                    # now do the assignment and save the resources that are no longer available
                    for s, t, v in assignment.get_delegations():
                        assigned_util[s][t] += v
                        es_switch = assignment.assignment[idx]
                        results.append((job, t, es_switch))
                        # count for output/debugging
                        if not assigned_to.get(es_switch.label):
                            assigned_to[es_switch.label] = 0
                        assigned_to[es_switch.label] += 1
                        idx += 1
                    # also save resources for link capacity
                    for s, t, v in assignment.get_link_demand_constraints():
                        assigned_demand[job.switch][s][t] += v    

                    # break for loop (this assignment is done)
                    break  

        # the firstFit heuristic does not consider link capacity constraints towards the backup switch
        # and the table capacity of the backup switch ("ES") is set to a very high value; thus, the heuristic
        # should be able to solve every input (in the worst case, everything is delegated to the 
        # backup switch)                      
        assert(len(vjobs) == len(results))
        # some default output to console
        logger.info(" ** [firstFit] t=%d jobs=%d iterations=%d  >>  %s " % (
            tick, len(vjobs), cnt, ','.join(['%s=%d' % (k,v) for k, v in sorted(assigned_to.items())])))
        return results

    # ==========================
    # ALGO 7 (Quadratic Algorithm)
    # ==========================

    def algo7_wrapper(self):
        stime = time.time()
        m = Model("solver_rss")
        X = {}
        objective_function = []
        constraint_free_capacity = {}

        demand_constraints = {}
        demand_constraint_mapping = {}

        PENALITY_FLOW_MIGRATION = 5

        mapping = {}

        for vj in self.vjobs:
            j = vj.job_id
            logger.info("-----------job=%d" % j) 

            use_jobs = list(vj.jobs.items())



            if len(use_jobs) > 1:
                for j1, j2 in zip(use_jobs, use_jobs[1:]):
                    t1, dstat1 = j1
                    t2, dstat2 = j2
                    assert(t1+1 == t2)

                    delegate1 = dstat1.util_raw - dstat1.util
                    delegate2 = dstat2.util_raw - dstat2.util

                    c = list(itertools.product(dstat1.es_options, dstat2.es_options))
                    for options in c:
                        logger.info("c job=%d t=(%d,%d) len=%d %s" % (j, t1, t2, 
                            len(use_jobs), str([s.label for s in options])))

                        key1 = '%d_%s_%d' % (j, options[0].label, t1)
                        key2 = '%d_%s_%d' % (j, options[1].label, t2)

                        if not X.get(key1):
                            X[key1] = m.addVar(vtype=GRB.BINARY, name=key1)
                            mapping[key1] = (dstat1, options[0])
                            if options[0].label != 'ES':
                                free = self.free_capacity[options[0]][t1]
                                objective_function.append(X[key1] * ((free-delegate1)*-1))
                            else:
                                objective_function.append(X[key1] * 100)   

                        if not X.get(key2):
                            X[key2] = m.addVar(vtype=GRB.BINARY, name=key2)
                            mapping[key2] = (dstat2, options[1])
                            if options[1].label != 'ES':
                                free = self.free_capacity[options[1]][t2]
                                objective_function.append(X[key2] * ((free-delegate2)*-1))
                            else:
                                objective_function.append(X[key2] * 100) 

                        if options[0].label != options[1].label:
                            objective_function.append(X[key1] * X[key2] * (dstat1.util_raw - dstat1.util) * PENALITY_FLOW_MIGRATION)
            else:
                for t, dstat in use_jobs:
                    for s in self.switches:
                        if s in dstat.es_options and not s == dstat.es_switch:
                            key = '%d_%s_%d' % (j, s.label, t)
                            if not X.get(key):
                                X[key] = m.addVar(vtype=GRB.BINARY, name=key)
                                mapping[key] = (dstat, s)
                                if s.label != 'ES':
                                    free = self.free_capacity[s][t]
                                    objective_function.append(X[key] * ((free-(dstat.util_raw - dstat.util))*-1))
                                else:
                                    objective_function.append(X[key] * 100) 


            # table capacity
            for t, dstat in use_jobs:
                choose_one = []
                for s in self.switches:
                    if s in dstat.es_options and not s == dstat.es_switch:
                        key = '%d_%s_%d' % (j, s.label, t)

                        choose_one.append(X[key])

                        if not constraint_free_capacity.get(s):
                            constraint_free_capacity[s] = {}
                        if not constraint_free_capacity[s].get(t):
                            constraint_free_capacity[s][t] = []

                        constraint_free_capacity[s][t].append(X[key] * (dstat.util_raw - dstat.util))

                        if s.label != "ES":
                            if not demand_constraints.get(t):
                                demand_constraints[t] = {}
                            if not demand_constraints[t].get(dstat.switch):
                                demand_constraints[t][dstat.switch] = {}
                            if not demand_constraints[t][dstat.switch].get(s):
                                demand_constraints[t][dstat.switch][s] = []
                            demand_constraints[t][dstat.switch][s].append(X[key] * dstat.demand)

                m.addConstr(quicksum(choose_one) == 1)

        for s, data in constraint_free_capacity.items():
            for t, list_x in data.items():
                m.addConstr(quicksum(list_x) <= self.free_capacity[s][t])

        # add link demand constraints for all jobs in this time slot
        for t, data in demand_constraints.items(): # time slot considered
            for s1, data2 in data.items(): # demand sent from s1
                for s2, coefficients in data2.items(): # towards s2
                    if len(coefficients) > 0:
                        free = self.free_util[s1][s2][t]   
                        free2 = self.free_util[s2][s1][t] 
                        logger.debug("add constr %s->%s coefficients=%d free=%f | %f" % (s1.label, 
                            s2.label, len(coefficients),  - free, self.link_bandwidth -free2))
                        m.addConstr(quicksum(coefficients) <= max(self.link_bandwidth  - free, 0))
                        m.addConstr(quicksum(coefficients) <= max(self.link_bandwidth  - free2, 0))


        m.setObjective(quicksum(objective_function), GRB.MINIMIZE)
        self.statistics_append('rsa.solver.stats_time_modeling', time.time()-stime)

        m.setParam('TimeLimit', 600.0)

        stime = time.time()
        m.optimize()
        self.statistics_append('rsa.solver.stats_time_solving2', time.time()-stime)

        self.statistics_model(m)

        if m.status == 3 or m.status == 9:
            # infeasable or timelimit
            logger.info("! model was infeasable")
            return None
        else:
            self.ctx.statistics['solver.cnt_feasable'] += 1  
            # get results from the solved optimization problem
            for v in m.getVars():
                if v.x > 0.5:
                    dstat, switch = mapping.get(v.varname)
                    logger.info("found solution %s %d->%s" % (v.varname, dstat.job_id, switch.label))
                    dstat.es_switch = switch
                    dstat.es = switch.label



    def split_traffic(self):

        
        print("")
        print("run dts algorithms for all switches in scenario")
        topo = self.ctx.scenario.g

        #for i, f in enumerate(self.ctx.scenario.all_flow_rules):
        #    logger.info(" flow %d %f %f" % (i, f.get('start'), f.get('duration')))
        DTS = {}
        DTS_CTX = {} # required for verification with simulator
        for switch in topo.nodes():
            self.timer.start('dts.%d' % switch)
            print("")
            print("============ run dts for switch %d, capacity=%d ============" % (switch, 
                self.ctx.scenario.threshold))
            DTS_CTX[switch], DTS[switch] = self.run_dss(switch)
            #DTS[switch].plot_utilization()
            self.timer.stop()

        threshold = self.ctx.scenario.threshold
   
        print("")
        print("result summary dts")
        print("  threshold", threshold)

        self.free_capacity[9999] = {} # 9999 is the backup switch
        for t in range(0, self.ctx.maxtick):
            self.free_capacity[9999][t] = 10000000

        for switch, dts_result in DTS.items():
            print("  switch =", switch)
            print("    ports", len(dts_result.ports))

            self.ctx.statistics['dts.%d.solver.considered_ports' % (switch)] = len(dts_result.ports)


            if len(dts_result.ports) == 0:
                # this is a special rare case where a switch is only connected to other switches and
                # there is no raw traffic flowing over this switch; in this case, the switch can be used
                # as a remote switch with its full capacity by all connected switches
                print("    special case; currently no traffic at all")
                self.free_capacity[switch] = {}
                for t in range(0, self.ctx.maxtick):
                    self.free_capacity[switch][t]  = threshold              
                #for src, tgt in self.ctx.scenario.g.edges():
                #    if src == switch:
                #        print("  can be used as remote switch by s%s" % tgt)
                print("    free_capacity_total", sum(self.free_capacity[switch].values()))
                continue # important

            # we need the free capacity of each switch and time slot
            self.free_capacity[switch] = {}
            for t in range(1, dts_result.maxtick-1):
                rawutil = dts_result.utils_raw.get(t, 0)
                util = dts_result.utils.get(t,0)
                checkutil = 0
                checkraw = 0
                delegated_ports = []
                for port, dstatus in dts_result.delegation_status[t].items():
                    checkutil += dstatus.util 
                    checkraw += dstatus.util_raw
                    if dstatus.status == 1:
                        delegated_ports.append(dstatus)
 
                # dts_result is the key input for the rss algorithm so the values
                # should be accurate; what the above test does is it checks whether the
                # expected current utilization of the switch (stored in util[t]) maps the
                # utilization values of the individual ports
                try:
                    assert(checkutil == util)
                except Exception as e:
                    logger.info("ERROR: dts_result seems to be invalid! check carefully!")
                    logger.info("* t=%d" % t)
                    logger.info("* checkutil = %d" % checkutil)
                    logger.info("* util = %d" % util)
                    for port, dstatus in dts_result.delegation_status[t].items():
                        logger.info("  -> port=%d status=%d util=%d" % (port, dstatus.status, dstatus.util))
                    raise(e)

                #if checkraw > 0:
                #    assert(checkraw == rawutil)
                #print(t, "rawutil", rawutil, checkraw, "checkutil", checkutil, "util", util)
                #for delegation in delegated_ports:
                #    print("   ", delegation.port, delegation.util, delegation.util_raw, delegation.util_raw-delegation.util)
                self.free_capacity[switch][t] = threshold - rawutil
                if len(delegated_ports) > 0:
                    self.free_capacity[switch][t] = 0 


            print("    free_capacity_total", sum(self.free_capacity[switch].values()))


        if self.ctx.config.get('param_rsa_skip') == 1:
            print("")
            print("skip rsa... (param_rsa_skip=1)")     
            return


        self.timer.start('prepare_rsa')

        # store dts results in ctx
        self.ctx.all_dts_results = DTS

        print("")
        print("run rsa")
        print("  param_rsa_timelimit", self.ctx.config.get('param_rsa_timelimit'))
        print("  param_rsa_algo", self.ctx.config.get('param_rsa_algo'))
        print("  param_rsa_max_assignments", self.ctx.config.get('param_rsa_max_assignments'))

        job_lengths = []
        option_lengths = []  

        # next, the individual jobs and remote switch options need to be determined
        for switch, dts_result in DTS.items():

            jobs = {}
            job_id = switch * 1000000 # make sure that job ids are unique 
            for p in dts_result.ports:

                status = []
                for t in range(1, dts_result.maxtick-1):
                    dstat = dts_result.delegation_status[t][p]
                    dstat.switch = switch # set switch
                    status.append('%d' % dstat.status)
                    # assign a unique job id
                    dstat.job_id = job_id
                    if dstat.status == 1 and dts_result.delegation_status[t+1][p].status == 0:
                        job_id+=1

                    # status=1 means the port is delegated
                    if dstat.status == 1:
                        # add the backup switch
                        dstat.es_options.append(9999)
                        # add remote switch options
                        for remote_switch in topo.nodes():
                            if switch == remote_switch: continue; # we cannot delegate to ourself
                            # make sure the remote switch is connected to switch
                            if self.ctx.scenario.g.has_edge(switch, remote_switch):
                                # make sure remote switch has free capacity
                                if self.free_capacity[remote_switch][t] >= dstat.util_raw - dstat.util:
                                    dstat.es_options.append(remote_switch)
                        # add this as a new job
                        if not jobs.get(dstat.job_id):
                            jobs[dstat.job_id] = {}
                        jobs[dstat.job_id][t] = dstat

            print("")
            print("  jobs for switch %d:" % switch)

            # create vjobs objects for the rsa solver     
            for j, jobs in jobs.items():
                job_lengths.append(len(jobs))
                for t, dstat in jobs.items():
                    option_lengths.append(len(dstat.es_options))
                    #if not self.active_jobs.get(t):
                    #     self.active_jobs[t] = [0,0]
                    #self.active_jobs[t][0] += 1
                    #self.active_jobs[t][1] += dstat.util_raw - dstat.util
                vj = VJob(self, jobs)
                vj.create_assignments()
                self.vjobs.append(vj)
                print('  job=%d length=%d start=%d end=%d' % (j, len(jobs), vj.start, vj.end))


        # add some statistics about the delegation jobs
        self.ctx.statistics['rsa.solver.jobs'] = len(self.vjobs)
        self.ctx.statistics['rsa.solver.jobs_lengths'] = job_lengths
        if len(self.vjobs) >= 1:
            self.ctx.statistics['rsa.solver.jobs_maxlen'] = max(job_lengths)
            self.ctx.statistics['rsa.solver.jobs_minlen'] = min(job_lengths)
            self.ctx.statistics['rsa.solver.jobs_sumlen'] = sum(job_lengths)
            self.ctx.statistics['rsa.solver.jobs_avglen'] = sum(job_lengths)/len(job_lengths)
            self.ctx.statistics['rsa.solver.jobs_options_avg'] = sum(option_lengths)/len(option_lengths)

        # get free link utilization
        for switch, dts_result in DTS.items():
            self.free_util[switch] = {}
            # gather "raw" demand for all ports, i.e., the demand that the rules process in a scenario
            # without delegation
            # p is the port id (0,1,..)
            # port is the label of the connected device such as s1h4 or dummy_switch_5 -> 5 in this case
            for p, port in dts_result.ports.items():
                if port.startswith('dummy_switch_'):
                    # we are only interested here in demand that goes to another switch; these 
                    # are all encoded as dummy switches; the switch id can be extracted by just
                    # removing the prefix "dummy_switch_"
                    port = int(port.replace('dummy_switch_', '')) # (not really a port, its the other switch)
                    for remote_switch in self.ctx.scenario.g.nodes():
                        if remote_switch == port:
                            self.free_util[switch][remote_switch] = {}
                            for t in range(1, dts_result.maxtick-1):
                                dstat = dts_result.delegation_status[t][p]
                                self.free_util[switch][remote_switch][t] = dstat.demand_raw

        # it can happen that a switch has a connection but the port count is 0; this happens if a small
        # number of hosts is distributed over a large number of switches so that a switch exists that is only
        # connected to another switch but to no host; this switch is thus fully idle and can provide its full
        # capacity as a remote switch (it would also be viable to exclude such situations completely)
        for switch, dts_result in DTS.items():
            if len(dts_result.ports) == 0:
                for remote_switch in self.ctx.scenario.g.nodes():  
                    if switch == remote_switch: continue;
                    if self.ctx.scenario.g.has_edge(switch, remote_switch):
                        # all capacity is available -> used demand is 0
                        # (has to be added for both directions)
                        if not self.free_util[switch].get(remote_switch):
                            self.free_util[switch][remote_switch] = {}   
                        for t in range(0, self.ctx.maxtick):
                            self.free_util[switch][remote_switch][t] = 0   
                        if not self.free_util[remote_switch].get(switch):
                            self.free_util[remote_switch][switch] = {}   
                        for t in range(0, self.ctx.maxtick):
                            self.free_util[remote_switch][switch][t] = 0 

        # finally, check whether any links are missing; this happens if there is 
        # no traffic at all on these link; todo: check whether this replaces the code above
        for s1, s2 in self.ctx.scenario.g.edges():
            try:
                self.free_util[s1][s2]
            except KeyError:
                logger.info('.. added missing free_util[%d][%d]' % (s1, s2)) # rare event, make sure this is noticable
                if not self.free_util.get(s1): self.free_util[s1] = {}
                if not self.free_util.get(s1).get(s1): self.free_util[s1][s2] = {}
                for t in range(0, self.ctx.maxtick):      
                    self.free_util[s1][s2][t] = 0     
            try:
                self.free_util[s2][s1]
            except KeyError:
                logger.info('.. added missing free_util[%d][%d]' % (s2, s1)) # rare event, make sure this is noticable
                if not self.free_util.get(s2): self.free_util[s2] = {}
                if not self.free_util.get(s2).get(s1): self.free_util[s2][s1] = {}
                for t in range(0, self.ctx.maxtick):
                    self.free_util[s2][s1][t] = 0   

        algo = self.ctx.config.get('param_rsa_algo')

        self.timer.stop()
        self.timer.start('run_rsa')
        # select RSS algorithm and execute it; the result will be stored inside the dts_result objects, i.e., 
        # inside DTS (the dstat.es_switch variable contains the decision that was made for a specific port 
        # and time slot)
        if algo == 1:
            # uses the global self.vjobs object (no need to pass DTS)
            self.algo1_wrapper()

        if algo == 2:
            self.algo2_wrapper(DTS)

        if algo == 3:
            self.algo3_wrapper()

        if algo == 7: 
            self.algo7_wrapper()

        self.timer.stop()

        backup_switch_utilization = {}
        for t in range(0,449):
            backup_switch_utilization[t] = 0
        
        # update shared utilization (rules reloacted to remote switch)
        for switch, dts_result in DTS.items():
            for p, port in dts_result.ports.items():
                for t in range(1, dts_result.maxtick-1):
                    dstat = dts_result.delegation_status[t][p]
                    if dstat.status == 1:
                        try:
                            assert(dstat.es_switch is not None) # there must be a decision
                        except AssertionError as e:
                            # this should never happen (but happened several times during
                            # development; the debug code was not removed here just in case)
                            print("Debugging info for AssertionError:")
                            print("switch", switch)
                            print("p", p)
                            print("t", t)
                            print("dstat", dstat)
                            print("dstat.status", dstat.status)
                            print("dstat.es", dstat.es)
                            print("dstat.es_switch", dstat.es_switch)
                            for job in self.vjobs:
                                if job.switch == switch:
                                    print("")
                                    print(" job ", job.job_id)
                                    for k, dstat2 in job.jobs.items():
                                        if dstat2.switch == switch and dstat2.tick == t:
                                            print("**** dstat", k, dstat2.es)
                                        else:
                                            print("     dstat", k, dstat2.es)  
                            raise e

                        # delegated means some rules are moved
                        # in the case of select-copyFirst, the delegated value can get
                        # negative (-1) because a port with 0 utilization has to be delegated at t1
                        # due to a bottleneck at t1+x (with x smaller as the look ahead factor)
                        delegated = dstat.util_raw - dstat.util
                        #if delegated < 0:
                        #    print("!!!!", t, port, dstat.util_raw, dstat.util)

                        assert(delegated >= -1)     
                        if dstat.es_switch == 9999:
                            # backup switch was chosen 
                            backup_switch_utilization[t] += delegated
                        else:
                            # regular switch was chosen
                            DTS.get(dstat.es_switch).utils_shared[t] += delegated


        plt.close()
        switch_cnt = len(DTS)+1
        ncol = 3 # number of colums
        nrow = int(switch_cnt/ncol)
        if ncol*nrow < switch_cnt:
            nrow+=1
        fig, ax = plt.subplots(nrow, ncol, figsize=(14, 8))
        fig.tight_layout(pad=2.7)
        row_x = 0
        row_y = 0
        max_flowcnt = 0

        util_total = 0
        shared_total = 0

        min_x_arr = []
        max_x_arr = []


        stats_remote_options = [] # used to calculate the average amount of remote switch options
        stats_remote_used = [] # used to calculate the average amount of actually used remote switches
        stats_rsa_table_fairness_mse = []
        stats_rsa_table_fairness_avg = []
        stats_rsa_ctrl_overhead = 0

        for switch, dts_result in DTS.items():
            delegated_once = False
            datax = []
            datay = []
            datay_raw = []
            datay_shared = []

            min_x = 0
            max_x = 0
            
            stats_ctrl_overhead = 0
            stats_table_fairness_jain = [] # calculate table fairness
            stats_table_fairness_mse = []
            stats_table_fairness_avg = []
            last_dstat = {}
            for t in range(1, dts_result.maxtick-1):
                util = 0
                util_raw = 0
                util_shared = dts_result.utils_shared[t]

                relocated_to = [] # list of switches used as remote switch in this time slot
                for p, port in dts_result.ports.items():
                    dstat = dts_result.delegation_status[t][p]
                    if dstat.status == 1: 
                        delegated_once = True
                        relocated_to.append(dstat.es)
                        # check whether the remote switch was changed; in this case, additional
                        # control overhead is counted because rules have to relocated to the new
                        # remote switch
                        last = last_dstat.get(p)
                        if last and last.status == 1 and last.es != dstat.es:
                            stats_ctrl_overhead += 2*dstat.util
                    util += dstat.util # this includes the overhead already!
                    util_raw += dstat.util_raw # the raw utilization
                    last_dstat[p] = dstat
                datax.append(t)
                datay.append(util + util_shared)
                datay_raw.append(util_raw)
                datay_shared.append(util)

                # to calculate the table fairness score, it is required to iterate over all remote
                # switches (in case there is a bottleneck in this switch which is the case if relocated_to
                # is non-empty) and calculate the difference between the total utilization at the remote switch
                # compared to the table capacity; the closes these three values are to each other, the more
                # fairness is given
                if len(relocated_to) > 0:
                    neighbors = list(self.ctx.scenario.g.neighbors(switch))
                    stats_remote_options.append(len(neighbors))
                    stats_remote_used.append(len(relocated_to))
                    r_utils = []
                    for r in neighbors:
                        # total difference between capacity and current table utilization at that switch
                        r_utils.append(threshold - (DTS.get(r).utils.get(t, 0) + DTS.get(r).utils_shared.get(t, 0)))
                    if len(r_utils) > 0:
                        # mean error squared
                        m = max(r_utils)
                        mse = sum([abs(x-m)*abs(x-m) for x in r_utils]) / (len(r_utils) * threshold)
                        # average error
                        avg = sum([abs(x-m) for x in r_utils]) / len(r_utils)
                        # jains fairness index
                        try:
                            jains = (sum(r_utils)*sum(r_utils)) / (len(r_utils) * sum([x*x for x in r_utils]))
                            stats_table_fairness_jain.append(jains)
                        except:
                            pass
                        stats_table_fairness_mse.append(mse)
                        stats_table_fairness_avg.append(avg)


                testmax = max(util_raw,  util + util_shared)
                if testmax > max_flowcnt:
                    max_flowcnt = testmax
                if testmax > 0 and min_x == 0:
                    min_x = t
                if min_x > 0 and testmax == 0 and max_x == 0:
                    max_x = t

                util_total += util_raw
                shared_total += util_shared

            # add data to statistics.json
            self.ctx.statistics['dts.%d.table.datax' % switch] = datax
            self.ctx.statistics['dts.%d.table.datay' % switch] = datay
            self.ctx.statistics['dts.%d.table.datay_raw' % switch] = datay_raw
            self.ctx.statistics['dts.%d.table.datay_shared' % switch] = datay_shared
            self.ctx.statistics['dts.%d.table.fairness_jain' % switch] = stats_table_fairness_jain
            self.ctx.statistics['dts.%d.table.fairness_mse' % switch] = stats_table_fairness_mse
            self.ctx.statistics['dts.%d.table.fairness_avg' % switch] = stats_table_fairness_avg
            self.ctx.statistics['dts.%d.ctrl.overhead_from_rsa' % switch] = stats_ctrl_overhead

            min_x_arr.append(min_x)
            max_x_arr.append(max_x)

            if len(stats_table_fairness_avg) > 0:
                stats_rsa_table_fairness_avg.append(statistics.mean(stats_table_fairness_avg))
            if len(stats_table_fairness_mse) > 0:
                stats_rsa_table_fairness_mse.append(statistics.mean(stats_table_fairness_mse))

            stats_rsa_ctrl_overhead += stats_ctrl_overhead

            #fig.axes[switch].set_xlim(1, dts_result.maxtick-1)
            if delegated_once:
                fig.axes[switch].plot(datax, datay_raw, label="without flow delegation", 
                    color="red", linestyle=":", linewidth=0.8)

            fig.axes[switch].plot(datax, datay_shared, label="shared", color="green")

            fig.axes[switch].plot(datax, datay, label="with flow delegation", color="black")

            fig.axes[switch].fill_between(datax, datay, datay_shared, 
                interpolate=True, color='green', alpha=0.4)


        if len(stats_rsa_table_fairness_avg) > 0:
            self.ctx.statistics['rsa.table.fairness_avg'] = statistics.mean(stats_rsa_table_fairness_avg)
        if len(stats_rsa_table_fairness_mse) > 0:
            self.ctx.statistics['rsa.table.fairness_mse'] = statistics.mean(stats_rsa_table_fairness_mse)
           
        self.ctx.statistics['rsa.ctrl.overhead_from_rsa'] = stats_rsa_ctrl_overhead

        data_failed_x = list(backup_switch_utilization.keys())
        data_failed_y = list(backup_switch_utilization.values())

        # some simple statistics
        percent_relocated = (sum(data_failed_y) / util_total) * 100
        percent_shared = (shared_total / util_total) * 100
        percent_over_capacity = (shared_total+sum(data_failed_y) / util_total) * 100

        # add backup switch to last axis
        fig.axes[switch+1].fill_between(data_failed_x, [0]*len(data_failed_x), data_failed_y, 
            interpolate=True, color='red', alpha=0.3)
        fig.axes[switch+1].plot(data_failed_x, data_failed_y, color="red", linewidth=2)
        
        description = 'Capacity reduction: %d%%\nRelocated: %.2f%%\nNot handled: %.2f%%' % (
            100-self.ctx.scenario.factor, percent_shared, percent_relocated)
        fig.axes[switch+1].text(0.95, 0.95, '%s' % description, transform=fig.axes[switch+1].transAxes,
            fontsize=10, fontweight='normal', color="black", va='top', ha="right", bbox=dict(boxstyle="square",
            ec='white', fc='white',
        ))

        # plot threshold
        for i, ax in enumerate(fig.axes):
            # adjust ranges
            if i == switch+1:
                # backup switch
                ax.set_ylim(0, max(max(data_failed_y), max_flowcnt))
            else:
                # all other switches
                ax.set_ylim(0, max_flowcnt)
            ax.set_xlim(0, 400) 
            t1 = ax.hlines(threshold, 0, 449, color='black', 
                label="Flow table capacity", linestyle='--', linewidth=1.5)

        filename = os.path.join(os.path.dirname(self.ctx.configfile), 'result.pdf')
        fig.savefig(filename, dpi=300)

        if self.ctx.config.get('param_debug_show_result') == 1:
            plt.show()
        plt.close()

        # save data in statistics
        self.ctx.statistics['rsa.table.percent_relocated'] = percent_relocated
        self.ctx.statistics['rsa.table.percent_shared'] = percent_shared
        self.ctx.statistics['rsa.table.percent_over_capacity'] = percent_over_capacity
        self.ctx.statistics['rsa.table.backup_switch_util_over_time'] = list(backup_switch_utilization.values())

        #plt.show()
        DTS_DATA = self.update_statistics_dts(DTS, DTS_CTX)

        rsa_data = self.update_statistics_rsa(DTS_DATA)

        if self.ctx.config.get('param_debug_show_result_demands') == 1:
            self.plot_demand_with_fd()


    def update_statistics_rsa(self, DTS_DATA):

        self.timer.start('update_statistics_rsa')
        rsa_data = RSAData(self.ctx, DTS_DATA)
        rsa_data.calculate_with_fd()
        rsa_data.update_statistics_with_fd()
        self.timer.stop()
        return rsa_data


    def update_statistics_dts(self, DTS_RESULT, DTS_CTX):
        self.timer.start('update_statistics_dts')
        DTS_DATA = {}
        delegated_total = 0
        for switch in range(0, self.ctx.config.get('param_topo_num_switches')):

            # initialize analytics
            analytics = DTSData(self.ctx, switch)
            DTS_DATA[switch] = analytics
            new_ctx = DTS_CTX[switch]

            # calculate results with and without flow delegation
            analytics_utils_raw, analytics_raw_per_port, demand_in, demand_out, flows_of_timeslot_per_port = analytics.get_raw_util()
            analytics_util_with_delegation, _, delegated_demand_in, delegated_demand_out, backdelegations = analytics.get_util(DTS_RESULT[switch])

            if self.ctx.config.get('param_debug_verify_analytically') == 1:
                # verify dstat.utils_raw 
                checked = 0
                status = 0 # number of mismatched util values (should be zero)
                for k, v in DTS_RESULT[switch].utils_raw.items():
                    if v:
                        if v > 0:
                            checked += 1
                            if not analytics_utils_raw.get(k) == v:
                                status += 1
                                print("error check analytics_utils_raw", k, v, analytics_utils_raw.get(k))
                logger.info("[switch=%d] check analytics_utils_raw                 %s (%d checked)" % (switch, str(status), checked ))
                self.ctx.statistics['dts.%d.verify.utils_raw' % (switch)] = status

                # verify dstat.demand_raw
                checked = 0
                status = 0 
                for t, data_by_port in DTS_RESULT[switch].delegation_status.items():
                    for p, dstat in data_by_port.items():
                        checked += 1
                        if not math.isclose(dstat.demand_raw, 
                            analytics.demand_in_per_port.get(dstat.port_label).get(t, 0)):
                            status += 1
                logger.info("[switch=%d] check analytics_demand_raw                %s (%d checked)" % (switch, str(status),checked ))
                self.ctx.statistics['dts.%d.verify.demand_raw' % (switch)] = status

                # check utilization with flow delegation
                checked = 0
                status = 0 # number of mismatched util values (should be zero)
                for k, v in DTS_RESULT[switch].utils.items():
                    checked += 1
                    if not analytics_util_with_delegation.get(k, 0) == v:
                        status += 1
                        print("error check analytics_util_with_delegation", k, v, analytics_util_with_delegation.get(k))
                logger.info("[switch=%d] check analytics_util_with_delegation      %s (%d checked)" % (switch, str(status), checked))  
                self.ctx.statistics['dts.%d.verify.utils' % (switch)] = status

                # verify dstat.demand
                checked = 0
                status = 0 
                for t, data_by_port in DTS_RESULT[switch].delegation_status.items():
                    for p, dstat in data_by_port.items(): 
                        checked += 1
                        if not math.isclose(dstat.demand, 
                            analytics.with_fd_delegated_demand_per_port.get(dstat.port_label).get(t, 0)):
                            status += 1
                logger.info("[switch=%d] check analytics_demand_delegated          %s (%d checked)" % (switch, str(status),checked ))
                self.ctx.statistics['dts.%d.verify.demand' % (switch)] = status


            # prepare statistics delegated_mbit
            checked = 0
            status = 0 
            draw = {}
            for t, data_by_port in DTS_RESULT[switch].delegation_status.items():
                for p, dstat in data_by_port.items():
                    if dstat.status == 1:
                        try:
                            draw[(switch, dstat.es)].append((t, dstat.demand))
                        except KeyError:
                            draw[(switch, dstat.es)] = [(t, dstat.demand)]
            if len(draw) > 0:
                for key, data in draw.items():
                    s1, s2 = key
                    datax = []
                    datay = []   
                    for x in range(0, self.ctx.maxtick):
                        datax.append(x)
                        summed = 0
                        for t, v in data:
                            if t == x:
                                delegated_total += v
                                summed += v/1000000
                        datay.append(summed)
                    if not self.ctx.config.get('param_debug_small_statistics') == 1:
                        self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datax' % (s1, s2)] = datax
                        self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datay' % (s1, s2)] = datay

            
            self.ctx.statistics['dts.%d.analytics.ctrl_overhead' % (switch)] = backdelegations
            # calculate demand
            total_demand_in = 0
            for t, d in demand_in.items():
                total_demand_in += d
            self.ctx.statistics['dts.%d.analytics.total_demand_in' % (switch)] = total_demand_in

            total_demand_out = 0
            for t, d in demand_out.items():
                total_demand_out += d
            self.ctx.statistics['dts.%d.analytics.total_demand_out' % (switch)] = total_demand_out

            total_delegated_demand_in = 0
            for t, d in delegated_demand_in.items():
                total_delegated_demand_in += d
            self.ctx.statistics['dts.%d.analytics.total_delegated_demand_in' % (switch)] = total_delegated_demand_in

            
            cTable = self.ctx.scenario.threshold
            cPortCount = len(DTS_RESULT[switch].ports)
            cRuleCount = len(analytics.flows_by_id)
            underutil = 0
            underutil_max = 0
            underutil_percent = 0
            overutil = 0
            overutil_max = 0
            overutil_percent = 0
            table_overhead = 0
            table_overhead_max = 0
            table_overhead_percent = 0
            ctrl_overhead = backdelegations
            ctrl_overhead_max = cRuleCount
            ctrl_overhead_percent = 0
            link_overhead_percent = 0

            # calculate metrics: link overhead
            if total_demand_in > 0:
                link_overhead_percent = (total_delegated_demand_in / total_demand_in)*100

            # calculate metrics: underutil and overutil
            for t, v in analytics_util_with_delegation.items():
                raw = analytics_utils_raw.get(t, 0)
                if raw > 0:
                    table_overhead_max += cPortCount
                if raw > cTable:
                    overutil_max += raw-cTable
                    underutil_max += cTable
                if raw > cTable and v < cTable:
                    underutil += cTable-v
                if v > cTable:
                    overutil += v - cTable

            if underutil_max > 0:
                underutil_percent = (underutil/underutil_max)*100
            if overutil_max > 0:
                overutil_percent = (overutil/overutil_max)*100
            if ctrl_overhead_max > 0:
                ctrl_overhead_percent = (ctrl_overhead/ctrl_overhead_max)*100

            # calculate metrics: rule overhead
            for t, dstats in DTS_RESULT[switch].delegation_status.items():
                for p, dstat in dstats.items():
                    table_overhead += dstat.status
            if table_overhead_max > 0:
                table_overhead_percent = (table_overhead/table_overhead_max)*100

            self.ctx.statistics['dts.%d.underutil' % switch] = underutil
            self.ctx.statistics['dts.%d.underutil_max' % switch] = underutil_max
            self.ctx.statistics['dts.%d.underutil_percent' % switch] = underutil_percent

            self.ctx.statistics['dts.%d.overutil' % switch] = overutil
            self.ctx.statistics['dts.%d.overutil_max' % switch] = overutil_max
            self.ctx.statistics['dts.%d.overutil_percent' % switch] = overutil_percent

            self.ctx.statistics['dts.%d.table_overhead' % switch] = table_overhead
            self.ctx.statistics['dts.%d.table_overhead_max' % switch] = table_overhead_max
            self.ctx.statistics['dts.%d.table_overhead_percent' % switch] = table_overhead_percent

            self.ctx.statistics['dts.%d.ctrl_overhead' % switch] = ctrl_overhead
            self.ctx.statistics['dts.%d.ctrl_overhead_max' % switch] = ctrl_overhead_max
            self.ctx.statistics['dts.%d.ctrl_overhead_percent' % switch] = ctrl_overhead_percent

            self.ctx.statistics['dts.%d.link_overhead' % switch] = total_delegated_demand_in
            self.ctx.statistics['dts.%d.link_overhead_max' % switch] = total_demand_in
            self.ctx.statistics['dts.%d.link_overhead_percent' % switch] = link_overhead_percent



            # copy simulator results for comparison
            if self.ctx.config.get('param_debug_verify_with_simulator') == 1:

                mapStats = [
                    ('metrics.ds.underutil_percent', 'verify.underutil_percent'),
                    ('metrics.ds.overhead_percent', 'verify.table_overhead_percent'),
                    ('metrics.ds.backdelegations_percent', 'verify.ctrl_overhead_percent'),
                    ('metrics.demand_delegated_percent', 'verify.link_overhead_percent')
                ]

                for old_key, new_key in mapStats:
                    self.ctx.statistics['dts.%d.%s' % (switch, new_key)] =  new_ctx.statistics.get(old_key) 
                                

                self.ctx.statistics['dts.%d.verify.ctrl_overhead' % (switch)] = new_ctx.statistics.get('metrics.ds.backdelegations')
                self.ctx.statistics['dts.%d.verify.total_demand_in' % (switch)] = new_ctx.statistics.get('metrics.demand_total')
                self.ctx.statistics['dts.%d.verify.total_demand_out' % (switch)] = new_ctx.statistics.get('metrics.demand_total')
                self.ctx.statistics['dts.%d.verify.total_delegated_demand_in' % (switch)] = new_ctx.statistics.get('metrics.demand_delegated')

                diff = new_ctx.statistics.get('metrics.ds.backdelegations') - backdelegations
                self.ctx.statistics['dts.%d.verify.diff.ctrl_overhead' % (switch)] = diff
                logger.info("check diff.ctrl_overhead                  %.2f" % abs(diff))

                diff = 0
                if not numpy.isclose(new_ctx.statistics.get('metrics.demand_total') - total_demand_in, 0):
                    diff = new_ctx.statistics.get('metrics.demand_total') - total_demand_in
                self.ctx.statistics['dts.%d.verify.diff.total_demand_in' % (switch)] = diff
                logger.info("check diff.total_demand_in                %.2f" % abs(diff))

                diff = 0
                if not numpy.isclose(new_ctx.statistics.get('metrics.demand_total') - total_demand_out,0):
                    diff = new_ctx.statistics.get('metrics.demand_total') - total_demand_out
                self.ctx.statistics['dts.%d.verify.diff.total_demand_out' % (switch)] = diff
                logger.info("check diff.total_demand_out               %.2f" % abs(diff))

                diff = 0
                if not numpy.isclose(new_ctx.statistics.get('metrics.demand_delegated') - total_delegated_demand_in,0):
                    diff = new_ctx.statistics.get('metrics.demand_delegated') - total_delegated_demand_in
                self.ctx.statistics['dts.%d.verify.diff.total_delegated_demand_in' % (switch)] = diff                
                logger.info("check diff.total_delegated_demand_in      %.2f" % abs(diff))

        self.ctx.statistics['rsa.link.util_delegated_bits_total2'] = delegated_total
        self.timer.stop()
        return DTS_DATA

    def plot_demand_with_fd(self):

        fig, ax = plt.subplots(4, 4, figsize=(14, 8))
        fig.tight_layout(pad=2.7)
        
        cnt = 0
        for _s1, _s2 in self.ctx.scenario.g.edges():
            for s1, s2 in [(_s1, _s2), (_s2, _s1)]:
                rawx = self.ctx.statistics['rsa.link.%d.%d.raw_mbit.datax' % (s1, s2)] 
                rawy = self.ctx.statistics['rsa.link.%d.%d.raw_mbit.datay' % (s1, s2)] 
                try:
                    delegatedy = self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datay' % (s1, s2)]
                    delegatedx = self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datax' % (s1, s2)]
                except KeyError:
                    delegatedx = self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datax' % (s2, s1)]
                    delegatedy = self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datay' % (s2, s1)]

                cnt_raw = 0
                cnt_delegated = 0
                datax = []
                datay_raw = []
                datay_delegated = []
                for x in range(0, self.ctx.maxtick):
                    datax.append(x)

                    if x >= rawx[0]:
                        try:
                            datay_raw.append(min(rawy[cnt_raw], 1000))
                        except:
                            datay_raw.append(0)
                        cnt_raw += 1
                    else:
                        datay_raw.append(0)

                    if x >= delegatedx[0]:
                        try:
                            datay_delegated.append(delegatedy[cnt_delegated])
                        except:
                            datay_delegated.append(0)
                        cnt_delegated += 1
                    else:
                        datay_delegated.append(0)

                ax = fig.axes[cnt]
                ax.stackplot(datax, datay_raw, datay_delegated, colors=['lightblue', 'green'])
                ax.set_title("%d %d" % (s1, s2))
                cnt+=1
        plt.show()
        plt.close()

    def check_for_time_limit(self):
        if self.timelimit > 0:
            if self.ctx.started > 0:
                if time.time() - self.ctx.started > self.timelimit:
                    raise TimeoutError()


    def export_ratings(self, all_rating_summaries, tick):

        print("")
        print("   summary", tick)
        data_ratings = []
        data_ratings_min = []
        data_ratings_max = []
        data_ratings_avg = []

        data_table = []
        data_table_min = []
        data_table_max = []
        data_table_avg = []

        data_link = []
        data_link_min = []
        data_link_max = []
        data_link_avg = []

        data_ctrl = []
        data_ctrl_min = []
        data_ctrl_max = []
        data_ctrl_avg = []

        data_static = []
        data_static_min = []
        data_static_max = []
        data_static_avg = []

        for summary_ratings, summary_table, summary_link, summary_ctrl, summary_static in all_rating_summaries:
            data_ratings_min.append(summary_ratings[0])
            data_ratings_max.append(summary_ratings[1])
            data_ratings_avg.append(summary_ratings[2])

            data_table_min.append(summary_table[0])
            data_table_max.append(summary_table[1])
            data_table_avg.append(summary_table[2])

            data_link_min.append(summary_link[0])
            data_link_max.append(summary_link[1])
            data_link_avg.append(summary_link[2])

            data_ctrl_min.append(summary_ctrl[0])
            data_ctrl_max.append(summary_ctrl[1])
            data_ctrl_avg.append(summary_ctrl[2])

            data_static_min.append(summary_static[0])
            data_static_max.append(summary_static[1])
            data_static_avg.append(summary_static[2])    

        try:
            data_ratings_min = (round(min(data_ratings_min),2), 
                round(statistics.mean(data_ratings_min),2), round(max(data_ratings_min),2))
            print("     data_ratings_min:", data_ratings_min)
        except:
            data_ratings_min = (0,0,0) 
        try:
            data_ratings_avg = (round(min(data_ratings_avg),2), 
                round(statistics.mean(data_ratings_avg),2), round(max(data_ratings_avg),2))
            print("     data_ratings_avg:", data_ratings_avg)
        except:
            data_ratings_avg = (0,0,0) 
        try:
            data_ratings_max = (round(min(data_ratings_max),2), 
                round(statistics.mean(data_ratings_max),2), round(max(data_ratings_max),2))
            print("     data_ratings_max:", data_ratings_max)
        except:
            data_ratings_max = (0,0,0) 
        # ------------------------------------------
        try:
            data_table_min = (round(min(data_table_min),2), 
                round(statistics.mean(data_table_min),2), round(max(data_table_min),2))
            print("     data_table_min:", data_table_min)
        except:
            data_table_min = (0,0,0) 
        try:
            data_table_avg = (round(min(data_table_avg),2), 
                round(statistics.mean(data_table_avg),2), round(max(data_table_avg),2))
            print("     data_table_avg:", data_table_avg)
        except:
            data_table_avg = (0,0,0) 
        try:
            data_table_max = (round(min(data_table_max),2), 
                round(statistics.mean(data_table_max),2), round(max(data_table_max),2))
            print("     data_table_max:", data_table_max)
        except:
            data_table_max = (0,0,0) 
        # ------------------------------------------
        try:
            data_link_min = (round(min(data_link_min),2), 
                round(statistics.mean(data_link_min),2), round(max(data_link_min),2))
            print("     data_link_min:", data_link_min)
        except:
            data_link_min = (0,0,0) 
        try:
            data_link_avg = (round(min(data_link_avg),2), 
                round(statistics.mean(data_link_avg),2), round(max(data_link_avg),2))
            print("     data_link_avg:", data_link_avg)
        except:
            data_link_avg = (0,0,0) 
        try:
            data_link_max = (round(min(data_link_max),2), 
                round(statistics.mean(data_link_max),2), round(max(data_link_max),2))
            print("     data_link_max:", data_link_max)
        except:
            data_link_max = (0,0,0) 
        # ------------------------------------------
        try:
            data_ctrl_min = (round(min(data_ctrl_min),2), 
                round(statistics.mean(data_ctrl_min),2), round(max(data_ctrl_min),2))
            print("     data_ctrl_min:", data_ctrl_min)
        except:
            data_ctrl_min = (0,0,0) 
        try:
            data_ctrl_avg = (round(min(data_ctrl_avg),2), 
                round(statistics.mean(data_ctrl_avg),2), round(max(data_ctrl_avg),2))
            print("     data_ctrl_avg:", data_ctrl_avg)
        except:
            data_ctrl_avg = (0,0,0) 
        try:
            data_ctrl_max = (round(min(data_ctrl_max),2), 
                round(statistics.mean(data_ctrl_max),2), round(max(data_ctrl_max),2))
            print("     data_ctrl_max:", data_ctrl_max)
        except:
            data_ctrl_max = (0,0,0) 
        # ------------------------------------------
        try:
            data_static_min = (round(min(data_static_min),2), 
                round(statistics.mean(data_static_min),2), round(max(data_static_min),2))
            print("     data_static_min:", data_static_min)
        except:
            data_static_min = (0,0,0) 
        try:
            data_static_avg = (round(min(data_static_avg),2), 
                round(statistics.mean(data_static_avg),2), round(max(data_static_avg),2))
            print("     data_static_avg:", data_static_avg)
        except:
            data_static_avg = (0,0,0) 
        try:
            data_static_max = (round(min(data_static_max),2), 
                round(statistics.mean(data_static_max),2), round(max(data_static_max),2))
            print("     data_static_max:", data_static_max)
        except:
            data_static_max = (0,0,0)  

        try:
            test = self.ctx.statistics['rsa.solver.weights.rating_total.min']
        except:
            self.ctx.statistics['rsa.solver.weights.rating_total.min'] = [] 
            self.ctx.statistics['rsa.solver.weights.rating_total.avg'] = []
            self.ctx.statistics['rsa.solver.weights.rating_total.max'] = []
            self.ctx.statistics['rsa.solver.weights.table.min'] = []
            self.ctx.statistics['rsa.solver.weights.table.avg'] = []
            self.ctx.statistics['rsa.solver.weights.table.max'] = []
            self.ctx.statistics['rsa.solver.weights.link.min'] = []
            self.ctx.statistics['rsa.solver.weights.link.avg'] = []
            self.ctx.statistics['rsa.solver.weights.link.max'] = []
            self.ctx.statistics['rsa.solver.weights.ctrl.min'] = []
            self.ctx.statistics['rsa.solver.weights.ctrl.avg'] = []
            self.ctx.statistics['rsa.solver.weights.ctrl.max'] = []
            self.ctx.statistics['rsa.solver.weights.static.min'] = []
            self.ctx.statistics['rsa.solver.weights.static.avg'] = []
            self.ctx.statistics['rsa.solver.weights.static.max'] = []

        try:
            self.ctx.statistics['rsa.solver.weights.rating_total.min'].append(min(data_ratings_min))
            self.ctx.statistics['rsa.solver.weights.rating_total.avg'].append(statistics.mean(data_ratings_min))
            self.ctx.statistics['rsa.solver.weights.rating_total.max'].append(max(data_ratings_min))
            
            self.ctx.statistics['rsa.solver.weights.table.min'].append(min(data_table_min))
            self.ctx.statistics['rsa.solver.weights.table.avg'].append(statistics.mean(data_table_min))
            self.ctx.statistics['rsa.solver.weights.table.max'].append(max(data_table_min))
            
            self.ctx.statistics['rsa.solver.weights.link.min'].append(min(data_link_min))
            self.ctx.statistics['rsa.solver.weights.link.avg'].append(statistics.mean(data_link_min))
            self.ctx.statistics['rsa.solver.weights.link.max'].append(max(data_link_min))

            self.ctx.statistics['rsa.solver.weights.ctrl.min'].append(min(data_ctrl_min))
            self.ctx.statistics['rsa.solver.weights.ctrl.avg'].append(statistics.mean(data_ctrl_min))
            self.ctx.statistics['rsa.solver.weights.ctrl.max'].append(max(data_ctrl_min))

            self.ctx.statistics['rsa.solver.weights.static.min'].append(min(data_static_min))
            self.ctx.statistics['rsa.solver.weights.static.avg'].append(statistics.mean(data_static_min))
            self.ctx.statistics['rsa.solver.weights.static.max'].append(max(data_static_min))
        except Exception as e:
            logger.info("Error creating rating statistics for t=%d (%s)" % (tick, str(e)))

