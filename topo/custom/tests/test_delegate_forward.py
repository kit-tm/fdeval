from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class TestEngine(Engine):

    def on_EVSwitchStats(self, switch, ev):
        #        
        es = self.ctx.topo.get_switch_by_label('ES')      
        if switch.label == 'DS':
            if math.isclose(ev.tick, 3):
                print("@%.0f add" % ev.tick)
                for id, flow in self.active_flows.items():
                    self.add_delegation(ev.tick, flow, switch, es)



        super().on_EVSwitchStats(switch, ev)

    def on_EVSwitchNewFlow(self, switch, ev):
        # forward flow on next switch in path
        super().on_EVSwitchNewFlow(switch, ev)

class MyTopo( Topo ):
    "delegate to a switch that is used again on the path afterwards, i.e., ..->ds->es->ds->es->s2->... "

    def __init__( self, ctx ):

        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0.5))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # Initialize
        Topo.__init__( self )
        ds = self.addSwitch( 'DS', x=2, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        ds2 = self.addSwitch( 'DS2', x=2, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        es = self.addSwitch( 'ES', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        h1 = self.addHost( 'h1', x=4, y=1)
        h2 = self.addHost( 'h2',x=4, y=3)

        self.addLink( ds, es, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( ds2, es, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h1, ds, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h2, ds2, capacity=1000, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f0", fg_start=0,  fg_demand=100, fg_duration=10, 
            fg_fixed_path=['h1', 'DS', 'ES', 'DS2', 'h2']))


        # call on_done if simulation is finished
        ctx.on_test_finished = self.on_done

    def on_done(self, ctx):
        testutil.print_summary(ctx)

        print(testutil.get_flow_timings(ctx))
        errors = []
        errors += testutil.verify_flow_timings(ctx, FLOW_TIMINGS)
        return errors
        #return []

def get_topo(ctx):
    return MyTopo(ctx)

topos = { 'MyTopo': ( lambda: MyTopo() ) }

FLOW_TIMINGS = """{"DS->ES": {"f0": [0.5, 12.0]}, "DS->h1": {}, "DS2->ES": {},
"DS2->h2": {"f0": [1.5, 13.0]}, "ES->DS": {"f0": [3, 11.5]}, 
"ES->DS2": {"f0": [1.0, 12.5]}, "h1->DS": {"f0": [0, 10.5]}, "h2->DS2": {}}"""