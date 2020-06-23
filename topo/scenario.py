"""
Scenario evaluation of the overall flow delegation problem (DTS+RSA) 
based on the barabasi-albert model
"""
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import random, math
import logging
import os
import operator
import statistics 
import numpy as np
import scipy

from pprint import pprint
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import skewnorm
from engine.solve_dts_data import DTSData
from engine.solve_rsa_data import RSAData
from topo.mix import rvs, avg
from core.statistics import Timer
from topo.static import LAYOUTS, MIX

logger = logging.getLogger(__name__)


def get_random_demand(g):
    number_of_switches = len(g.nodes())
    if number_of_switches == 1:
        return 0, 0, [0]
    source = random.randint(0, number_of_switches-1)
    target = source
    while target == source:
        target = random.randint(0, number_of_switches-1)

    return source, target, nx.shortest_path(g, source=source, target=target)

def get_host_pair(set1, set2):
    choose1 = random.choice(set1)
    choose2 = choose1
    tries = 0
    while choose2 == choose1 and tries < 100:
        choose2 = random.choice(set2)
        tries+=1
    if choose2 == choose1:
        raise IndexError("could not find a valid host pair; sets: %s | %s" % (str(set1), str(set2)))
    return choose1, choose2    


class DelegationScenario(object):
    """
    This is a selection of scenario data structures that together describes a flow delegation scenario. Such 
    a scenario consists of the following data structures:
        g: a topology graph with switches (hosts are NOT included here)
        all_flow_rules: list of flow rules to be used with the gen_copy.py class
        hosts_of_switch: a dict that maps which hosts are attached to which switch
        flows_of_switch: a dict that maps flow rules to switch

    The scenario can be parameterized in various ways (the parameters are listed below in the init function).
    """
    def __init__(self, ctx, **kwargs):
        super(DelegationScenario, self).__init__()
        self.ctx = ctx
        self.timer = Timer(self.ctx, 'scenario_generator')
        self.verbose = kwargs.get('verbose', False)
        self.preview_scenario = kwargs.get('preview_scenario', False) 

        # these variables are set during scenario generation
        self.dts_data = {} # stores DTSData object per switch (if required)
        self.rsa_data = None # stores RSAData object
        self.factor = -1 # factor of maximum utilization (in percent)
        self.demand_scale_factor = {} # <flow_id> -> scale factor for demand
        self.threshold = -1 # the effective threshold (based on maximum utilization)
        self.max_utilization = 0 # maximum utilization of all switches
        self.concentrated_switches = [] # contains a list with switches where demand is concentrated (if configured)
        self.bottlenecks = None
        self.figure = None

        self.param_topo_scenario_generator = self.ctx.config.get('param_topo_scenario_generator')
        self.param_topo_num_hosts =self.ctx.config.get('param_topo_num_hosts') # number of hosts in total (NOT per switch)
        self.param_topo_num_flows =self.ctx.config.get('param_topo_num_flows') # number of flow rules in total
        self.param_topo_num_switches =self.ctx.config.get('param_topo_num_switches') # number of switches
        self.param_topo_seed =self.ctx.config.get('param_topo_seed') # seed used for all random number generators
        self.param_topo_scenario_ba_modelparam =self.ctx.config.get('param_topo_scenario_ba_modelparam') # parameter m of BA model
        self.param_topo_scenario_fd_beta =self.ctx.config.get('param_topo_scenario_fd_beta') # parameter beta for flow delegation duration 
        self.param_topo_bottleneck_cnt =self.ctx.config.get('param_topo_bottleneck_cnt') # number of artifical bottlenecks added
        self.param_topo_bottleneck_duration =self.ctx.config.get('param_topo_bottleneck_duration') # duration of one bottleneck
        self.param_topo_bottleneck_intensity =self.ctx.config.get('param_topo_bottleneck_intensity') # iat reduction factor (bottleneck intensity)
        self.param_topo_concentrate_demand =self.ctx.config.get('param_topo_concentrate_demand') # means that bottlenecks occur only on x switches, not all switches
        self.param_topo_concentrate_demand_retries =self.ctx.config.get('param_topo_concentrate_demand_retries') # retries for demand allocation to create localized bottlenecks
        self.param_topo_iat_scale =self.ctx.config.get('param_topo_iat_scale') # scale down iat to allow larger flow rule counts
        self.param_topo_traffic_interswitch = self.ctx.config.get('param_topo_traffic_interswitch')
        self.param_topo_traffic_scale =self.ctx.config.get('param_topo_traffic_scale')
        self.param_topo_idle_timeout = self.ctx.config.get('param_topo_idle_timeout') 

        self.static_iat_shape = 0.4754
        self.static_iat_scale = 13.7300
        self.static_max_util = 1000000000

        self.log("run scenario generator")
        self.log("  param_topo_scenario_generator", self.param_topo_scenario_generator)
        self.log("  param_topo_num_switches", self.param_topo_num_switches)
        self.log("  param_topo_num_hosts", self.param_topo_num_hosts)
        self.log("  param_topo_num_flows", self.param_topo_num_flows)
        self.log("  param_topo_seed", self.param_topo_seed)
        self.log("  param_topo_scenario_ba_modelparam", self.param_topo_scenario_ba_modelparam)
        self.log("  param_topo_bottleneck_cnt", self.param_topo_bottleneck_cnt)
        self.log("  param_topo_concentrate_demand", self.param_topo_concentrate_demand)
        self.log("  param_topo_bottleneck_duration", self.param_topo_bottleneck_duration)
        self.log("  param_topo_concentrate_demand_retries", self.param_topo_concentrate_demand_retries)
        self.log("  param_topo_scenario_fd_beta", self.param_topo_scenario_fd_beta)
        self.log("  param_topo_iat_scale", self.param_topo_iat_scale)

        self.g = None # nx graph of the topology
        self.all_flow_rules = [] # list with all flow rules
        self.hosts_of_switch = {}
        self.flows_of_switch = {}

        assert(self.param_topo_traffic_interswitch >= 0 and self.param_topo_traffic_interswitch <= 100)
        self.param_topo_traffic_interswitch = float(self.param_topo_traffic_interswitch)/100.0

        if self.param_topo_seed >= 0:
            logger.info("use seed: %d" % self.param_topo_seed)
            random.seed(self.param_topo_seed)
            np.random.seed(self.param_topo_seed)

        if not self.verbose:
            logger.info("verbose=False, no output is generated; use DelegationScenario(verbose=True) to get output.")
            self.preview_scenario = False # requires verbose true

    def get_data_slim(self):
        return dict(
            switch_cnt=len(self.g.nodes()),
            edges=[(n1,n2) for n1,n2 in self.g.edges()],
            hosts_of_switch=self.hosts_of_switch,
            concentrated_switches=self.concentrated_switches,
            bottlenecks=self.bottlenecks,
            figure=self.figure
        )

    def get_data(self):
        return dict(
            g=json_graph.node_link_data(self.g),
            all_flow_rules=self.all_flow_rules,
            hosts_of_switch=self.hosts_of_switch,
            flows_of_switch=self.flows_of_switch,
            concentrated_switches=self.concentrated_switches,
            bottlenecks=self.bottlenecks
        )

    def add_data_to_ctx(self):
        """Add scenario parameters etc to ctx.statistics"""
        for k, v in self.get_data_slim().items():
            self.ctx.statistics['scenario.%s' % k] = v

        rules = []
        for k, flows in sorted(self.flows_of_switch.items()):
            rules.append(len(flows))

        hostcnt = []  
        for h, hosts in self.hosts_of_switch.items():
            try:
                hostcnt.append(len(hosts))
            except:
                hostcnt.append(0)

        self.ctx.statistics['scenario.rules_per_switch'] = rules
        try:
            self.ctx.statistics['scenario.hosts_per_switch_max'] = max(hostcnt)
        except:
            self.ctx.statistics['scenario.hosts_per_switch_max'] = 0 
        try:
            self.ctx.statistics['scenario.hosts_per_switch_min'] = min(hostcnt)
        except:
            self.ctx.statistics['scenario.hosts_per_switch_min'] = 0            
        try:
            self.ctx.statistics['scenario.hosts_per_switch_avg'] = statistics.mean(hostcnt)
        except:
            self.ctx.statistics['scenario.hosts_per_switch_avg'] = 0        
        try:
            self.ctx.statistics['scenario.rules_per_switch_avg'] = statistics.mean(rules)
        except:
            self.ctx.statistics['scenario.rules_per_switch_avg'] = 0
        try:
            self.ctx.statistics['scenario.rules_total'] = sum(rules)
        except:
            self.ctx.statistics['scenario.rules_total'] = 0         
        try:
            self.ctx.statistics['scenario.rules_per_switch_max'] = max(rules)
        except:
            self.ctx.statistics['scenario.rules_per_switch_max'] = 0  
    def log(self, *msg):
        if self.verbose:
            print(*msg)


    def execute_generator(self):
        """
        Run the scenario generator with the provided parameters
        """
        logger.info("scenario generator will create %d flow rules for %d switches and %d hosts" % (
            self.param_topo_num_flows, self.param_topo_num_switches, self.param_topo_num_hosts))
        self.create_flow_rules() # sets all_flow_rules and flows_of_switch


        # calculate analytics
        self.timer.start('precalculate_data_dts')
        self.ctx.scenario = self # to be sure
        max_flowcnt = 0
        for switch in range(0, self.param_topo_num_switches):
            dts = DTSData(self.ctx, switch)
            dts.calculate_raw_utils()
            try:
                maxutil = max(dts.utils_raw.values())
            except ValueError:
                maxutil = 0
            if maxutil > max_flowcnt:
                max_flowcnt = maxutil
            self.dts_data[switch] = dts
        self.max_utilization = max_flowcnt
        self.timer.stop()

        self.timer.start('precalculate_data_rts')
        self.rsa_data = RSAData(self.ctx, self.dts_data)
        self.rsa_data.calculate_raw()
        self.rsa_data.update_statistics()
        self.timer.stop()
        
        # calculate the thresholds to be used in the following simulations based on
        # max_flowcnt; the new thresholds will be a percentage of the maximum threshold, e.g.,
        # 70% of max_flowcnt
        threshold = int(max_flowcnt * self.ctx.config['param_topo_switch_capacity']/100)
        self.threshold = threshold # store threshold in scenario
        assert(threshold > 0)
        self.factor = self.ctx.config['param_topo_switch_capacity']
        self.ctx.statistics['scenario.table_capacity'] = self.threshold
        self.ctx.statistics['scenario.table_capacity_reduction'] = self.factor

    def create_topo(self):
        self.log("")
        self.log("create topology")

        # special case: a single node
        if self.param_topo_num_switches == 1:
            self.log("  use single node topology")
            self.g = nx.trivial_graph()
            self.hosts_of_switch = {}
            self.flows_of_switch = {}
            self.flows_of_switch[0] = []
            self.hosts_of_switch[0] = []
            # all hosts are assigned to the one switch
            for host in range(0, self.param_topo_num_hosts):
                self.hosts_of_switch[0].append(host) 
            return

        # use the Barabasi–Albert model to create scale-free topologies
        if self.param_topo_scenario_generator == 1:
            self.log("  use Barabasi–Albert model with m=%d" % self.param_topo_scenario_ba_modelparam)
            # first step is to create the topology of the switches
            seed = None
            if self.param_topo_seed >= 0:
                seed = self.param_topo_seed
            
            # n = Number of nodes
            # m = Number of edges to attach from a new node to existing nodes
            # seed = Seed for random number generator (default=None)
            self.g = nx.barabasi_albert_graph(self.param_topo_num_switches, 
                self.param_topo_scenario_ba_modelparam, seed=seed)

            # next, the hosts are attached (stored in hosts_of_switch, i.e., separate from g)
            self.hosts_of_switch = {}
            self.flows_of_switch = {}
            for switch in self.g.nodes():
                self.flows_of_switch[switch] = []
                self.hosts_of_switch[switch] = []

            # assign each host to one switch (randomly)
            for host in range(0, self.param_topo_num_hosts):
                use_switch = random.randint(0, self.param_topo_num_switches-1)
                self.hosts_of_switch[use_switch].append(host)  
            
            self.log("  nodes", len(self.g.nodes()))
            self.log("  edges", len(self.g.edges()))
            self.log("  node degrees", self.g.degree())
            self.log("  average degree", sum(d for n, d in self.g.degree()) / float(len(self.g.nodes())))

            # done
            return

        # invalid parameter
        raise Exception("param_topo_scenario_generator = %d is not supported" % self.param_topo_scenario_generator)
 
    # timeout 2-3 / lower than 5s --> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7946177
    def create_flow_rules(self):

        self.timer.start('create_topology')
        self.create_topo() # sets g and hosts_of_switch
        self.timer.stop()

        warm_up = 50
        iat_warmup = warm_up * 1000 # in ms
        self.timer.start('create_flow_arrival_times')
        dist_flow_iat = self.create_flow_arrival_times(plot=False)
        self.timer.stop() 

        sum_dist_iat = sum(dist_flow_iat)/1000.0

        #dist_flow_duration = self.create_flow_durations(beta=self.param_topo_scenario_fd_beta)
        
        self.timer.start('create_flow_demands')
        dist_flow_duration, dist_demands = self.create_flow_demands()
        max_dist_flow_duration = max(dist_flow_duration)
        self.timer.stop()

        # scale iat is only necessary if rule cnt is >50.000; in this case, the total time
        # can excess 450 seconds which is currently the upper limit for the simulator (hardcoded,
        # could be changed). scale will just reduce all values in the iat array so that the total sum
        # is small enough (param_topo_iat_scale has to be in the range of 300-400 for this to work);
        # the parameter can be ignored for smaller scenarios
        if self.param_topo_iat_scale > 1:
            f = (sum_dist_iat + max_dist_flow_duration) / float(self.param_topo_iat_scale)
            dist_flow_iat = [x / f for x in dist_flow_iat]
            sum_dist_iat = sum(dist_flow_iat)/1000.0

        # scale traffic
        if self.param_topo_traffic_scale > 0:
            factor = self.param_topo_traffic_scale/100.0
            assert(factor > 0)
            if not math.isclose(factor, 1):
                dist_demands = [x*factor for x in dist_demands]
                print("  scale traffic with factor %f" % factor)
 
        # prepare a list for each switch that will contain the flows passing through it
        self.flows_of_switch = {}
        for switch in self.g.nodes():
            self.flows_of_switch[switch] = []
        self.all_flow_rules = [] # list of all flow rules 

        # calculate the starting offset; recall that each experiment runs 400 seconds (static)
        # note that iat is given in milliseconds
        self.log("")
        self.log("create flow rules")
        self.log("  iat_scale", self.param_topo_iat_scale)
        self.log("  sum(dist_flow_iat)", sum_dist_iat)
        self.log("  max(dist_flow_duration)", max_dist_flow_duration)
        current_time_offset = 200-((sum_dist_iat + max_dist_flow_duration)/2)
        if current_time_offset < 0:
            current_time_offset = 10
        self.log("  start_time_offset (in seconds)", current_time_offset)

        self.log("")
        self.log("bottlenecks and concentrated demands")
        # calculate the bottlenecks; bottlenecks are simulated by artifically reducing the inter
        # arrival time of new flow rules, i.e., more rules are created in the same time span compared
        # to a scenario without a bottleneck; note that this doesn't change the distributions; larger 
        # bottlenecks (>=5s) follow a simple normal distribution, i.e., the center of the bottleneck has
        # a higher intensity (lower iat);
        bottlenecks = []
        # select time ranges for the bottleneck based on parameters (used for both, global and local bottlenecks)
        if self.param_topo_bottleneck_cnt > 0:
            for i in range(self.param_topo_bottleneck_cnt):
                start = random.randint(int(current_time_offset), int(current_time_offset + sum_dist_iat))
                end = start + self.param_topo_bottleneck_duration
                this_bottleneck = {}
                base_intensity = self.param_topo_bottleneck_intensity/100.0
                if self.param_topo_bottleneck_duration > 4:
                    n = np.random.normal(loc=1, scale=1/8, size=self.param_topo_bottleneck_duration) 
                    dist_bottleneck = [round(max(0, x * base_intensity),2) for x in n]
                    for t in range(start, end):
                        this_bottleneck[t] = dist_bottleneck.pop()
                else:
                    for t in range(start, end):
                        this_bottleneck[t] = base_intensity 
                bottlenecks.append((start, end, this_bottleneck))
        self.log("  number of bottlenecks", self.param_topo_bottleneck_cnt)
        self.log("  bottleneck times", bottlenecks)
        self.bottlenecks = bottlenecks # store bottlenecks

        # handle concentrated demands
        # concentrated_switches > 0 increase rules for one (or multiple switches)
        # concentrated_switches == 0 -> demand is distributed uniformly over all hosts
        concentrated_switches = None
        if self.param_topo_concentrate_demand > 0:
            if self.param_topo_num_switches == 1:
                self.log("  warning: param_topo_concentrate_demand ignored because there is only one switch")
            else:
                if self.param_topo_concentrate_demand > len(self.g.nodes()):
                    self.log("  concentrated demand parameter is too large (>switch cnt) --> reduced")
                    self.param_topo_concentrate_demand = len(self.g.nodes())
                # select switches
                candidates = []+list(self.g.nodes())
                random.shuffle(candidates)
                concentrated_switches = candidates[0:self.param_topo_concentrate_demand]
                self.concentrated_switches = concentrated_switches # store in scenario
                self.log("  number of (localized) bottlenecks", self.param_topo_bottleneck_cnt)
                self.log("  switches with bottlenecks", concentrated_switches)
                self.log("  bottleneck times", bottlenecks)

        self.timer.start('create_flows')
        #if self.param_topo_seed >= 0:
        #    random.seed(self.param_topo_seed)
        last_start_time = -1
        for demand_idx in range(self.param_topo_num_flows):

            # first decision is whether the two communication parties are both
            # attached to the same switch (rule is only required in this switch)
            # or whether a remote target is involved (rules have to be installed in
            # multiple switches)
            if random.random() > (1-self.param_topo_traffic_interswitch):
                # multiple switches
                host_assigned = False
                tries = 0
                tries_for_bottleneck = self.param_topo_concentrate_demand_retries
                while not host_assigned and tries < 100:
                    try:
                        source, target, shortest_path = get_random_demand(self.g)
                        # localized bottleneck: retry if demand was not for a bottlenecked switch
                        if concentrated_switches and tries_for_bottleneck > 0:
                            if source not in concentrated_switches:
                                tries_for_bottleneck -= 1
                                raise IndexError() # abort and try again

                        source_hostname, target_hostname = get_host_pair(self.hosts_of_switch[source], 
                            self.hosts_of_switch[target])
                        host_assigned = True
                    except IndexError:
                        # this happens if one of the two parameters is empty, i.e., there is no
                        # host attached to one of the switches 
                        tries +=1
                        if tries >= 100:
                            # if the algorithm is not capable of finding an interswitch connection with 100
                            # tries, it could be that all hosts are attached to a single node (i.e.,
                            # it is not possible to find one); this occurs very rarely in completely random
                            # scenarios with a very small amount of switches and hosts, e.g., 2 switches and 
                            # 10 hosts; first check whether this special situation caused the error
                            switches_with_more_than_one_host = []
                            for check in self.g.nodes():
                                if len(self.hosts_of_switch.get(check)) > 0:
                                    switches_with_more_than_one_host.append(check)
                            if len(switches_with_more_than_one_host) == 1:
                                # the error was caused by the special situation; to solve this, the current
                                # host pair is defined manually and after that, the inter switch factor
                                # is modified so that all followup connections are treated as single switch
                                # connections
                                switch = switches_with_more_than_one_host[0]
                                shortest_path = [switch]
                                source_hostname, target_hostname = get_host_pair(self.hosts_of_switch[switch], 
                                    self.hosts_of_switch[switch])
                                host_assigned = True
                                source = target = switch 
                                # now change parameter; note that we do NOT update the ctx.config dict
                                # to make sure experiments can be assigned properly later on!
                                self.param_topo_traffic_interswitch = 0
                                logger.info('WARNING: changed param_topo_traffic_interswitch to 0 because all hosts are attached to one switch')
                            else:
                                # other error (should never happen)
                                print("hosts_of_switch", self.hosts_of_switch)
                                raise Exception("cannot assign host after 100 tries; there might be too few hosts...")

            else:
                # single switch
                host_assigned = False
                tries = 0
                tries_for_bottleneck = self.param_topo_concentrate_demand_retries
                while not host_assigned and tries < 100:
                    try:
                        switch = random.randint(0, self.param_topo_num_switches-1)
                        # localized bottleneck: retry if demand was not for a bottlenecked switch
                        if concentrated_switches and tries_for_bottleneck > 0:
                            if switch not in concentrated_switches:
                                tries_for_bottleneck -= 1
                                raise IndexError() # abort and try again

                        shortest_path = [switch]
                        source_hostname, target_hostname = get_host_pair(self.hosts_of_switch[switch], 
                            self.hosts_of_switch[switch])
                        host_assigned = True
                        source = target = switch
                    except IndexError:
                        # this happens if one of the two parameters is empty, i.e., there is no
                        # host attached to one of the switches 
                        tries +=1
                        if tries >= 100:
                            print("hosts_of_switch", self.hosts_of_switch)
                            raise Exception("cannot assign host after 100 tries; there might be too few hosts...")

            # labeling of hosts follows a strict format (has to be this way!)
            source_label = "s%dh%d" % (source, source_hostname)
            target_label = "s%dh%d" % (target, target_hostname)

            iat = dist_flow_iat[demand_idx]/1000.0 # iat is given in milliseconds

            # reduce iat to start more flows to simulate a bottleneck
            for start, end, this_bottleneck in bottlenecks:
                if current_time_offset > start and current_time_offset < end:
                    try:
                        iat = iat / this_bottleneck.get(int(current_time_offset))
                    except:
                        # should never happen
                        iat = iat/max(1, self.param_topo_bottleneck_intensity/100.0)

            # in rare cases, it can happen that iat is none and two flows would
            # have the exact same start time which causes troubles because they cannot be 
            # compared against each other (important for min heap structure). To avoid
            # this, the case is manually avoided here by adding a small value to the iat
            # (very rare)
            if math.isclose(current_time_offset + iat, last_start_time):
                iat += 0.000001    

            new_flow = dict(
                flow_id=demand_idx,
                label='%d->%d' % (source, target),
                start=current_time_offset + iat, 
                demand_per_tick=dist_demands[demand_idx],
                duration=dist_flow_duration[demand_idx]+1,
                source_label=source_label,
                target_label=target_label,
                path=shortest_path
            )

            assert(not math.isclose(new_flow.get('start'), last_start_time))
            last_start_time = current_time_offset + iat

            # store the newly created flow rule
            self.all_flow_rules.append(new_flow)

            current_time_offset += iat

            for switch in shortest_path:
                self.flows_of_switch[switch].append(new_flow)

        self.timer.stop()

    def create_flow_durations(self, beta=1, offset=0, **kwargs):

        #s = np.random.lognormal(mu, sigma, 25000)
        s = np.random.exponential(beta, self.param_topo_num_flows)

        datax = []
        datay = []
        vsum = 0
        printed = 0
        for i, v in enumerate(sorted(s)):
            v = v + offset
            vsum += v
            datax.append(v)
            datay.append(i/self.param_topo_num_flows)

        #count, bins, ignored = plt.hist(s, 50, normed=True)
        if kwargs.get("plot"):
            plt.close()
            plt.plot(datax, datay, label="beta=%.2f offset=%.2f" % (float(beta), float(offset)))
            plt.xscale('log')
            plt.xlim(0.001, 1000)
            plt.legend()
            plt.show()

        return s

    def create_flow_demands(self):

        samples = rvs(MIX['flows'], 'length', self.param_topo_num_flows, random_state=self.param_topo_seed)

        # retreive average packet sizes for these flow lengths
        packet_size = avg(MIX, samples, 'length', 'packet_size') # returns octets!
        packet_size[packet_size < 64] = 64 # minimum packet size in octets
        packet_size[packet_size > 1522] = 1522 # maximum packet size in octets
        octets = samples * packet_size # final total bandwith demand of the flow
        print("  total_octets: %d" % sum(octets))

        mapper_ocs = {} # <flow length in packets> -> octets
        mapper_flow_cnt = {} # <flow length in packets -> number of flows with this length
        for i, flowlen in enumerate(samples):
            try:
                mapper_ocs[flowlen] += octets[i]
            except KeyError:
                mapper_ocs[flowlen] = octets[i]

            try:
                mapper_flow_cnt[flowlen] += 1
            except KeyError:
                mapper_flow_cnt[flowlen] = 1
        #print(mapper_ocs)

        ocs_total = sum(octets)
        flow_cnt_total = len(samples)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        datax = []
        datay = []
        datay2 = []
        ocs_sum = 0
        flow_cnt_sum = 0
        vals_ocs = {}
        vals_flow_cnt = {}
        for flowlen, ocs in sorted(mapper_ocs.items()):
            ocs_sum += ocs
            flow_cnt_sum += mapper_flow_cnt[flowlen]
            for x in [1,2,4,8,10,100,1000, 10000, 100000, 1000000]:
                if flowlen > x and vals_ocs.get(x) == None:
                    vals_ocs[x] = ocs_sum/ocs_total
                    vals_flow_cnt[x] = flow_cnt_sum/flow_cnt_total
            datax.append(flowlen)
            datay.append(ocs_sum/ocs_total)   
            datay2.append(flow_cnt_sum/flow_cnt_total)   
        ax = fig.axes[0]
        ax.set_xscale('log')
        ax.plot(datax, datay, color="blue")
        ax.plot(datax, datay2, color="red")
          
        print("octets")
        pprint(vals_ocs)

        print("flowcnt")
        pprint(vals_flow_cnt)

        bits = octets * 8
        dist_demand_per_tick = []
        dist_flow_duration_new = []
        dist_factor = {}

        default_bitrate = 10000.0
        for v in bits:
            factor = 1
            cnt = 1
            bitrate = default_bitrate*factor
            duration = v/bitrate


            while duration > factor:
                cnt += 1
                factor += math.sqrt(factor) 
                bitrate =  default_bitrate*factor*factor 
                duration = v/bitrate
            use_duration = min(duration, 35)

            effective_bitrate = v/use_duration
            if factor > 35:
                effective_bitrate *= (use_duration/factor)*(use_duration/factor)
            dist_flow_duration_new.append(use_duration + self.param_topo_idle_timeout)
            dist_demand_per_tick.append(int(effective_bitrate)) # exclude idle timeout here!
            try:
                dist_factor[factor][0] += 1
                dist_factor[factor][1] += use_duration+self.param_topo_idle_timeout
                dist_factor[factor][2] += int(effective_bitrate)
            except KeyError:
                dist_factor[factor] = [1, use_duration+self.param_topo_idle_timeout, effective_bitrate]

        for k, arr in dist_factor.items():
            arr[1] = arr[1]/arr[0]
            arr[2] = arr[2]/arr[0]
        print("bitrate / flow distribution")
        pprint(sorted(dist_factor.items()))
        #raise Exception()

        ax = fig.axes[1]
        datax = []
        datay = []
        vsum = 0
        for i, v in enumerate(sorted(dist_flow_duration_new)):
            v = v
            vsum += v
            datax.append(v)
            datay.append(i/self.param_topo_num_flows)
        ax.plot(datax, datay, label="new approach")
        #ax.set_xscale('log')
        #ax.set_xlim(0.001, 1000)
        #ax.legend()
        if self.ctx.config.get('param_debug_plot_distributions') == 1:
            plt.show()
        #raise Exception()
        plt.close()

        return dist_flow_duration_new, dist_demand_per_tick


    def create_flow_arrival_times(self, **kwargs):

        s = np.random.gamma(self.static_iat_shape, 
            self.static_iat_scale, self.param_topo_num_flows)

        sums = sum(s)
        datax = []
        datay = []
        vsum = 0
        printed = 0
        for i, v in enumerate(sorted(s)):
            vsum += v
            datax.append(v)
            datay.append(i/self.param_topo_num_flows)

        #count, bins, ignored = plt.hist(s, 50, normed=True)
        if kwargs.get("plot"):
            plt.xscale('log')
            plt.xlim(0.001, 100)
            plt.plot(datax, datay)
            plt.show()

        return s

    def add_scenario_data_to_statistics(self):
        # only used if only the scenario generator is executed
        data = {}
        for switch in self.g.nodes():
            dts_data = self.dts_data[switch]
            assert(dts_data)
            datax = []
            datay = []
            for t, v in sorted(dts_data.utils_raw.items()):
                datax.append(t)
                datay.append(v)
            self.ctx.statistics['scenario.table.datax.%d' % switch] = datax
            self.ctx.statistics['scenario.table.datay.%d' % switch] = datay

    def plot_scenario(self, show=False):

        self.timer.start('create_scenario_plot')
        plt.close()

        # +1 for topology plot in the top left
        x=9999 # used with LAYOUTS; topology is placed here
        switch_cnt = len(self.g.nodes())
        axes = []
        layout = LAYOUTS.get(switch_cnt)
        cols = len(layout[0])
        rows = len(layout)
        fig = plt.figure(constrained_layout=True, figsize=(14, 8))
        gs = GridSpec(rows, cols, figure=fig)

        # first the topology 
        coords = None
        for y in range(rows):
            for x in range(cols):
                if layout[y][x] == 9999:
                    if coords:
                        break;
                    coords = [y,x]
                    colspan = sum([1 if v == 9999 else 0 for v in layout[y]])
                    rowspan = sum([1 if 9999 in v else 0 for v in layout])  
                    break;
        axes.append(plt.subplot(gs.new_subplotspec((coords[0], coords[1]), rowspan=rowspan, colspan=colspan)))

        # and then all the other axes
        oldval = 0
        for y in range(rows):
            for x in range(cols):
                val = layout[y][x]
                if val == 9999:
                    continue;
                if val > oldval:
                    colspan = sum([1 if v == val else 0 for v in layout[y]])
                    rowspan = sum([1 if val in v else 0 for v in layout])
                    axes.append(plt.subplot(gs.new_subplotspec((y, x), rowspan=rowspan, colspan=colspan)))
                    oldval = val

        # plot topology in the top left
        self._plot_topo(axes[0])

        data = {}
        for switch in self.g.nodes():

            dts_data = self.dts_data[switch]
            assert(dts_data)

            datax = []
            datay = []
            for t, v in sorted(dts_data.utils_raw.items()):
                datax.append(t)
                datay.append(v)

            axes[switch+1].set_xlim(0,400)
            axes[switch+1].set_ylim(0, self.max_utilization)
            axes[switch+1].plot(datax, datay, color="black", linewidth=0.8)

            # plot threshold
            t1 = axes[switch+1].hlines(self.threshold, 0, 449, color='black', 
                label="Flow table capacity", linestyle='--', linewidth=1.5)

            # color overutil red
            if len(datay) > 0:
                total_utilization = sum(datay)
                total_overutilization = sum([x-self.threshold if x > self.threshold else 0 for x in datay])
                fill_overutil = [True if x > self.threshold else False for x in datay]
                axes[switch+1].fill_between(datax, datay, [self.threshold]*len(datay),
                    where=fill_overutil, interpolate=True, color='red', alpha=0.5, label='Utilization with flow delegation')

            # draw a small circle in the top left (the switch id to map to the topology figure)
            circlex = 0.05
            circley = 0.95
            circle = plt.Circle((circlex, circley), 0.032, color='lightblue', transform=axes[switch+1].transAxes)
            axes[switch+1].add_patch(circle)
            axes[switch+1].text(0.05, 0.95, '%d' % switch, transform=axes[switch+1].transAxes,
            fontsize=10, fontweight='normal', va='center', ha="center")

        # create a title with the most important parameters
        fig.suptitle("switches=%d hosts=%d m=%d hotspots=%d bottlenecks=%d capacity=%d%% seed=%d\n" % (
            self.param_topo_num_switches, self.param_topo_num_hosts, self.param_topo_scenario_ba_modelparam,
            self.param_topo_concentrate_demand, self.param_topo_bottleneck_cnt, self.factor,
            self.ctx.config.get('param_topo_seed')))

        #gs.update(top=0.8)
        #fig.subplots_adjust(top=0.95) # padding top

        filename = os.path.join(os.path.dirname(self.ctx.configfile), 'scenario.pdf')
        self.figure = filename
        fig.savefig(filename, dpi=300)
        self.timer.stop()

        if show:
            plt.show()
        plt.close()

    def plot_topo(self, show=False):
        """ store figure of only the topo """
        plt.close()   
        fig, ax = plt.subplots(1, figsize=(8, 8))
        self._plot_topo(ax)
        ax.set_title("switches=%d hosts=%d m=%d flows=%d hotspots=%d bottlenecks=%d|%d|%d seed=%d\n" % (
            self.param_topo_num_switches, self.param_topo_num_hosts, self.param_topo_scenario_ba_modelparam,
            self.param_topo_num_flows, self.param_topo_concentrate_demand, self.param_topo_bottleneck_cnt,
            self.param_topo_bottleneck_duration, self.param_topo_bottleneck_intensity,
            self.ctx.config.get('param_topo_seed')))   
        filename = os.path.join(os.path.dirname(self.ctx.configfile), 'topo.pdf')
        self.figure = filename
        fig.savefig(filename, dpi=300)  
        if show:
            plt.show()
        plt.close()   

    def _plot_topo(self, ax=None):
        """
        Plots the scenario topology
        see https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
        """
        gnew = self.g.__class__() # create a copy for the visualization
        gnew.add_nodes_from(self.g)
        gnew.add_edges_from(self.g.edges)
        switch_cnt = len(self.g.nodes())

        # now add the hosts (not present in g)
        for switch, hosts in self.hosts_of_switch.items():
            for h in hosts:
                gnew.add_node('h%d' % h)
                gnew.add_edge(switch, 'h%d' % h)

        node_size = []
        color_map = []
        edge_color_map = []
        style = []
        width = []
        for node in gnew:
            try:
                if node < switch_cnt:
                    if len(self.concentrated_switches) > 0:
                        if node in self.concentrated_switches:
                            # localized bottlenecks get a different color
                            color_map.append('red')
                        else:
                            color_map.append('lightblue')
                    else:
                        color_map.append('lightblue')
                    node_size.append(300)
            except:
                color_map.append('white')
                node_size.append(100) 


        for n1, n2 in gnew.edges():
            try:
                if n1 < switch_cnt and n2 < switch_cnt:
                    edge_color_map.append('lightblue')
                    style.append('solid')
                    width.append(3.0)
                else:
                    edge_color_map.append('black')
                    style.append('dashed')
            except TypeError:
                edge_color_map.append('black')   
                style.append('dashed')
                width.append(1.0)


        draw_params = dict(node_color=color_map, with_labels=True, font_color="black", 
                font_size=9,edge_color=edge_color_map,node_size=node_size,style=style,width=width)
        if ax:
            draw_params['ax'] = ax
            nx.draw(gnew, **draw_params)          
        else:
            plt.close()     
            nx.draw(gnew, **draw_params)
            plt.show()
            
    def plot_demands_of_switch(self):
        plt.close()
        fig, ax = plt.subplots(figsize=(16, 8))
        self._plot_demands_of_switch(ax)
        plt.show()

    def _plot_demands_of_switch(self, ax):
        ally = []
        for source, data in self.rsa_data.link_utilization.items():
            for target, timedata in data.items():
                xvalues = []
                yvalues = []     
                first = True     
                for t, v in sorted(timedata.items()):
                    if first:
                        first = False
                        xvalues.append(int(t)-1) # for plotter
                        yvalues.append(0)
                    xvalues.append(int(t))
                    yvalues.append(v/1000000.0)
                xvalues.append(int(t)+1) # for plotter
                yvalues.append(0)
                ax.plot(xvalues, yvalues, label="%d->%d" % (source, target), alpha=0.3, linewidth=1) 
                ally.append(max(yvalues))


        xvalues = []
        yAvg = []
        yMax = []
        yOverflow = []
        for t, data in sorted(self.rsa_data.link_utilization_by_tick.items()):
            xvalues.append(t)
            yAvg.append(statistics.mean(data))
            yMax.append(max(data))
            over = [x-1000 for x in data if x >= 1000]
            if len(over) > 0:
                yOverflow.append(sum(over))
            else:
                yOverflow.append(0)
        total_avg = statistics.mean(yAvg)
        total_max = statistics.mean(yMax)
        total_over = statistics.mean(yOverflow)

        ax.plot(xvalues, yAvg, label="avg (%.2f Mbit/s)" % total_avg, color="black", linewidth=2) 
        ax.plot(xvalues, yMax, label="max (%.2f Mbit/s)" % total_max, color="black", linestyle=':', linewidth=2) 
        ax.plot(xvalues, yOverflow, label="over capacity sum (%.2f Mbit/s)" % total_over, color="blue", linestyle='--', linewidth=1) 

        if max(ally) > 900:
            # plot link capacity
            ax.hlines(1000, 0, 449, color='red', 
                label="Link capacity (1000 Mbit/s)", linestyle='--', linewidth=1.2)


        ax.set_xlabel('time (s)')
        ax.set_ylabel('link utilization (Mbit/s)')
        #ax.fill_between(xvalues, yvalues, 0,  color='orange', alpha=0.3)
        ax.set_title('Demand', fontsize=10, fontweight='bold')
        ax.set_xlim(0,450)
        #ax.set_ylim(0,350)
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax.xaxis.grid(True, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(loc=1)
        return ax


