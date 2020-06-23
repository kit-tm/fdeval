import random, math

from topo.custom.topo import Topo
from engine.engine_pbce_solver_v2 import PBCEEngineSolverV2
from engine.solve_dts import DTSSolver

import logging
logger = logging.getLogger(__name__)

class MyTopo( Topo ):
    "global_single_v3: used for verification with simulator, see solve_rsa.py"

    def __init__( self, ctx, switch, **kwargs ):

        Topo.__init__( self )
        self.ctx = ctx
        self.solver = None

        logger.info("running global_single_v3 for switch=%d" % switch)
        scenario = self.ctx.scenario
        if not scenario:
            raise Exception("ctx.scenario is not set, cannot continue")
        
        all_labels = sorted(self.ctx.scenario_data.ports)
        all_flow_rules = sorted(self.ctx.scenario_data.flows_by_id.values(), key=lambda x: x.get('start'))

        print("")
        print("required dummy switches")
        for label in all_labels:
            if label.startswith('dummy'):
                print("  ", label)

        print("")
        print("all ports of the simulated switch=%d (%d in total)" % (switch, len(all_labels)))
        for label in all_labels:
            print("  ", label)

        ds = self.addSwitch( 'DS', x=2, y=1, engine=PBCEEngineSolverV2(ctx))
        es = self.addSwitch( 'ES', x=0, y=0, engine=PBCEEngineSolverV2(ctx))

        self.addLink( ds, es, propagation_delay=0 )

        for label in all_labels:
            host = self.addHost(label, x=1, y=1)
            self.addLink( host, ds, propagation_delay=0 )

        # take traffic from constructor
        self.addTraffic(dict(fg_class='gen_copy', fg_params=dict(
            flow_params=all_flow_rules, all_labels=all_labels)))

        # call on_done if simulation is finished
        ctx.on_simulation_finished = self.on_done
        ctx.on_simulation_setup_complete = self.on_simulation_setup_complete

    def on_simulation_setup_complete(self, ctx):
        assert(self.ctx.stored_solution != None)

    def on_done(self, ctx):
        pass


def get_topo(ctx, switch, **kwargs):
    return MyTopo(ctx, switch, **kwargs)

topos = { 'global_single_v3': ( lambda: MyTopo() ) }