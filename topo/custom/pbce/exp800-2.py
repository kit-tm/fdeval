# Scenario evaluation of the overall flow delegation problem (DTS+RSA) based on the Barabasi-Albert model
# This is the main file for defining all the experiments in the thesis!

import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import random, math
import logging
import os

from random import normalvariate
from topo.scenario import DelegationScenario
from topo.custom.topo import Topo
from core.context import Context
from engine.solve_dts import DTSSolver
from engine.solve_rsa import RSSSolver
from engine.engine_pbce_solver import PBCEEngineSolver

logger = logging.getLogger(__name__)

# taken from https://stackoverflow.com/questions/35472461/select-one-element-from-a-list-using-python-following-the-normal-distribution
def normal_choice(lst, mean=None, stddev=None):
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # if stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / 6

    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]

class MyTopo( Topo ):
    "exp800-2 (using global rsa scenarios)"

    def __init__( self, ctx ):

        Topo.__init__( self )
        self.ctx = ctx

        # IMPORTANT: all parameters that may be set to 0 HAVE to
        # have the default value 0! Parameters where 0 is not a valid
        # value can have any default value

        # scenario parameters
        self.paramInt('param_topo_scenario_generator', 1) # 1 is BA model
        self.paramInt('param_topo_seed', -1) # -1 --> raise exception if seed is not defined (all experiments should be reproducible)
        self.paramInt('param_topo_switch_capacity', 90) # 20-99
        self.paramInt('param_topo_switch_capacity_overwrite', 0) # DEFAULT HAS TO BE 0!!
        self.paramInt('param_topo_traffic_scale_overwrite', 0) # DEFAULT HAS TO BE 0!!
        self.paramInt('param_topo_num_hosts', 20)
        self.paramInt('param_topo_num_flows', 25000)
        self.paramInt('param_topo_num_switches', 3)
        self.paramInt('param_topo_bottleneck_cnt', 0)
        self.paramInt('param_topo_bottleneck_duration', 0)
        self.paramInt('param_topo_bottleneck_intensity', 110) # 110=iat is reduced by 10% during bottleneck
        self.paramInt('param_topo_concentrate_demand', 0)
        self.paramInt('param_topo_concentrate_demand_retries', 0)
        self.paramInt('param_topo_scenario_ba_modelparam', 1)
        self.paramInt('param_topo_traffic_scale', 100) # 100=no scaling; > 100: more traffic; <100: less traffic 
        self.paramInt('param_topo_traffic_interswitch', 0) # DEFAULT HAS TO BE 0!!
        self.paramInt('param_topo_iat_scale', 250)
        self.paramInt('param_topo_idle_timeout', 1) # DEFAULT HAS TO BE 1

        # flow delegation parameters
        self.paramInt('param_dts_algo', 2)
        self.paramInt('param_dts_look_ahead', 3)
        self.paramInt('param_dts_skip_ahead', 1)
        self.paramInt('param_dts_timelimit', 200)
        self.paramInt('param_dts_objective', 3)
        self.paramInt('param_dts_weight_table', 0) # DEFAULT HAS TO BE 0!!
        self.paramInt('param_dts_weight_link', 0) # DEFAULT HAS TO BE 0!!
        self.paramInt('param_dts_weight_ctrl', 0) # DEFAULT HAS TO BE 0!! 
        self.paramInt('param_rsa_algo', 2)
        self.paramInt('param_rsa_look_ahead', 3)
        self.paramInt('param_rsa_max_assignments', 50)
        self.paramInt('param_rsa_timelimit', 200)
        self.paramInt('param_rsa_weight_table', 0) # DEFAULT HAS TO BE 0!!
        self.paramInt('param_rsa_weight_link', 0) # DEFAULT HAS TO BE 0!!
        self.paramInt('param_rsa_weight_ctrl', 0) # DEFAULT HAS TO BE 0!!  
        self.paramInt('param_rsa_weight_backup', 10000)  
        self.paramInt('param_rsa_skip', 0) # DEFAULT HAS TO BE 0

        # other parameters for verification, debugging etc
        self.paramInt('param_debug_total_time_limit', 600)
        self.paramInt('param_debug_small_statistics', 0) # DEFAULT HAS TO BE 0
        self.paramInt('param_debug_verify_with_simulator', 0) # DEFAULT HAS TO BE 0
        self.paramInt('param_debug_verify_analytically', 0)  # DEFAULT HAS TO BE 0
        self.paramInt('param_debug_export_rsa_ratings', 0)  # DEFAULT HAS TO BE 0
        self.paramInt('param_debug_only_run_scenario_generator', 0) # DEFAULT HAS TO BE 0
        self.paramInt('param_debug_plot_demands', 0) # DEFAULT HAS TO BE 0!! 
        self.paramInt('param_debug_plot_distributions', 0) # DEFAULT HAS TO BE 0!! 
        self.paramInt('param_debug_plot_scenario', 0) # DEFAULT HAS TO BE 0!! 
        self.paramInt('param_debug_plot_result', 0) # DEFAULT HAS TO BE 0!! 
        self.paramInt('param_debug_show_result', 0) # DEFAULT HAS TO BE 0!! 
        self.paramInt('param_debug_show_result_demands', 0) # DEFAULT HAS TO BE 0!! 

        # param_topo_scenario_generator = 2 will randomize all parameters using the given parameter as an 
        # upper or lower bound (the other bound parameter is determined automatically)
        cfg = self.ctx.config
        if cfg.get('param_topo_scenario_generator') == 2:

            # we want the same "random" scenario of the seed is identical (for reproducibility)
            random.seed(self.ctx.config.get('param_topo_seed'))

            # these are hold static
            self.ctx.config['param_topo_scenario_generator'] = 1
            self.ctx.config['param_dts_objective'] = 5

            # set default objective for DTS if not provided
            if self.ctx.config['param_dts_weight_table'] == 0 and self.ctx.config['param_dts_weight_link'] == 0 and self.ctx.config['param_dts_weight_ctrl'] == 0:
                self.ctx.config['param_dts_weight_table'] = 0
                self.ctx.config['param_dts_weight_link'] = 1
                self.ctx.config['param_dts_weight_ctrl'] = 0

            # set default objective for RSA if not provided
            if self.ctx.config['param_rsa_weight_table'] == 0 and self.ctx.config['param_rsa_weight_link'] == 0 and self.ctx.config['param_rsa_weight_ctrl'] == 0:
                self.ctx.config['param_rsa_weight_table'] = 1
                self.ctx.config['param_rsa_weight_link'] = 0
                self.ctx.config['param_rsa_weight_ctrl'] = 5

            # all with length 2
            bounds = dict(
                param_topo_switch_capacity = [20, 99], # capacity reduction factor
                param_topo_num_switches = [2, 15],
                param_topo_num_flows = [25000, 250000],
                param_topo_seed = [1, 1000000],
                param_topo_traffic_interswitch = [20,30,40,50,60,70,80],
                param_topo_bottleneck_cnt = [20,15,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,10,15,20],
                param_topo_bottleneck_duration = [1,50],
                param_topo_bottleneck_intensity = [250,200,180,160,140,120,110,125,155,170,190,210,280],
                param_topo_concentrate_demand = [4,3,2,2,1,1,0,0,1,1,2,2,3,4],
                param_topo_concentrate_demand_retries =  [10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,10],
                param_topo_scenario_ba_modelparam = [5,4,3,2,2,1,1,1,2,2,3,4,5],
                param_topo_iat_scale = [280,300,350,300,280],
                param_topo_idle_timeout = [1,5]
            )

            for k, v in self.ctx.config.items():
                for k2, boundaries in bounds.items():
                    if k == k2:
                        if len(boundaries) == 2:
                            randval = random.randint(boundaries[0], boundaries[1])   
                            self.ctx.config[k] = randval
                        if len(boundaries) > 2:
                            randval = normal_choice(boundaries) 
                            self.ctx.config[k] = randval                   

            switch_cnt = self.ctx.config.get('param_topo_num_switches')

            # only randomize param_topo_switch_capacity if the parameter is not specified
            if self.ctx.config.get('param_topo_switch_capacity_overwrite') > 0:
                self.ctx.config['param_topo_switch_capacity'] = ctx.config.get('param_topo_switch_capacity_overwrite')

            if self.ctx.config.get('param_topo_bottleneck_cnt') == 0:
                self.ctx.config['param_topo_bottleneck_duration'] = 0   
                self.ctx.config['param_topo_bottleneck_intensity'] = 0  

            if self.ctx.config.get('param_topo_concentrate_demand') == 0:
                self.ctx.config['param_topo_concentrate_demand_retries'] = 0 

            if self.ctx.config.get('param_topo_scenario_ba_modelparam') >= switch_cnt:
                self.ctx.config['param_topo_scenario_ba_modelparam'] = random.randint(1,switch_cnt-1)  

            if self.ctx.config.get('param_topo_concentrate_demand') >= switch_cnt:
                self.ctx.config['param_topo_concentrate_demand'] = random.randint(0,switch_cnt-1)  

            self.ctx.config['param_topo_num_hosts'] = random.randint(5*switch_cnt, 20*switch_cnt) 

            self.ctx.config['param_topo_traffic_scale'] = random.randint(1, 20)*25 
            if random.random() > 0.9:
                # use very high traffic scale for 10% of the scenarios
                self.ctx.config['param_topo_traffic_scale'] = random.randint(1, 50)*250 
        
            # only randomize param_topo_switch_capacity if the parameter is not specified
            if self.ctx.config.get('param_topo_traffic_scale_overwrite') > 0:
                self.ctx.config['param_topo_traffic_scale'] = ctx.config.get('param_topo_traffic_scale_overwrite')

            # write randomized parameters to disk in case an error occurs (so that
            # it is possible to reproduce the error)
            params = []
            for k, v in self.ctx.config.items():
                if k.startswith('param'):
                    print(k, v)
                    params.append('%s %d\n' % (k, self.ctx.config[k]))
                # we need access to the generated parameters for scenario selection
                if k.startswith('param_topo'):
                    self.ctx.statistics['scenario.gen.%s' % k] = v
            store_params = os.path.join(os.path.dirname(ctx.configfile), 'randomized_params.txt')
            with open(store_params, 'w') as file:
                file.write('Randomized params are:\n\n' + ''.join(params)) 

        # create the scenario
        scenario = DelegationScenario(ctx, **dict(
            verbose=False,
            preview_scenario=True, # plot and show scenario, requires verbose=True
        ))
        scenario.execute_generator()
        #scenario.plot_topo()

        # write statistics
        scenario.add_data_to_ctx()

        # debug: plot scenario to file
        if self.ctx.config.get('param_debug_plot_scenario') == 1:
            scenario.plot_scenario(show=False)           

        # debug: plot demands
        if self.ctx.config.get('param_debug_plot_demands') == 1:
            scenario.plot_demands_of_switch()

        # indicates that the run generator function from the gui was used, i.e., 
        # no simulation is run (on_simulation_setup_complete etc are not executed)
        if self.ctx.run_scenario_generator: 
            scenario.plot_scenario(show=True)
            return

        # flag was set to only run the scenario generator (similar to above but
        # can be used in batch mode)
        if self.ctx.config.get('param_debug_only_run_scenario_generator') == 1:
            scenario.add_scenario_data_to_statistics()
            return

        # add scenario to ctx so it is available for the solvers etc
        ctx.scenario = scenario

        if not scenario.threshold > 0:
            raise Exception("Invalid param_topo_switch_capacity parameter")
        self.ctx.config['param_topo_switch_capacity'] = scenario.threshold

        # now pass the scenario over to the RSA algorithm which will then 
        # further call the DTS algorithms
        self.rss = RSSSolver(ctx)
        self.rss.run()
        self.ctx.skip_main_simulation = True # we do not need the usual simulator

        return 

    def on_simulation_setup_complete(self, ctx):
        pass

    def on_done(self, ctx):
        pass

def get_topo(ctx):
    return MyTopo(ctx)

topos = { 'exp800-2': ( lambda: MyTopo() ) }