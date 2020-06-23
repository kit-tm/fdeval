from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class TestEngine(Engine):

    def on_EVSwitchStats(self, switch, ev):
        # this test will stresstest the delegation mechanism;
        # the same delegation relationship is added and removed
        # so quickly that the relationshop is removed BEFORE
        # the first packet of the delegated flow returns to the
        # delegation switch; this is not realistic, it's only
        # here to test for extreme error conditions;
        # note that the propagation delay is set to 0.75 in this test!         
        es = self.ctx.topo.get_switch_by_label('ES')      
        if switch.label == 'DS':
            if math.isclose(ev.tick, 3):
                print("@%.0f add" % ev.tick)
                for id, flow in self.active_flows.items():
                    self.add_delegation(ev.tick, flow, switch, es)
            if math.isclose(ev.tick, 4):
                print("@%.0f add" % ev.tick)
                for id, flow in self.active_flows.items():
                    self.remove_delegation(ev.tick, flow, switch, es)
            if math.isclose(ev.tick, 5):
                print("@%.0f add" % ev.tick)
                for id, flow in self.active_flows.items():
                    self.add_delegation(ev.tick, flow, switch, es)
            if math.isclose(ev.tick, 6):
                print("@%.0f add" % ev.tick)
                for id, flow in self.active_flows.items():
                    self.remove_delegation(ev.tick, flow, switch, es)
            if math.isclose(ev.tick, 8):
                print("@%.0f add" % ev.tick)
                for id, flow in self.active_flows.items():
                    self.add_delegation(ev.tick, flow, switch, es)
            if math.isclose(ev.tick, 9):
                print("@%.0f add" % ev.tick)
                for id, flow in self.active_flows.items():
                    self.remove_delegation(ev.tick, flow, switch, es)


        super().on_EVSwitchStats(switch, ev)

    def on_EVSwitchNewFlow(self, switch, ev):
        # forward flow on next switch in path
        super().on_EVSwitchNewFlow(switch, ev)

class MyTopo( Topo ):
    "quickly add and remove delegation relationships"

    def __init__( self, ctx ):

        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0.25))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # Initialize
        Topo.__init__( self )
        ds = self.addSwitch( 'DS', x=2, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        es = self.addSwitch( 'ES', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        h1 = self.addHost( 'h1', x=4, y=1)
        h2 = self.addHost( 'h2',x=4, y=3)

        self.addLink( ds, es, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h1, ds, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h2, ds, capacity=1000, propagation_delay=propagation_delay )

        # add traffic
        i=0
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f%d" % i, fg_start=i,  fg_demand=100, fg_duration=10, 
            fg_fixed_path=['h1', 'DS', 'h2']))


        # call on_done if simulation is finished
        ctx.on_test_finished = self.on_done

    def on_done(self, ctx):
        testutil.print_summary(ctx)

        print(testutil.get_flow_timings(ctx))
        errors = []
        errors += testutil.verify_flow_timings(ctx, FLOW_TIMINGS)
        return errors
   

def get_topo(ctx):
    return MyTopo(ctx)

topos = { 'MyTopo': ( lambda: MyTopo() ) }

# with delay set to 0.75 there are errors because the prop. delay is so high that
# stop events are not triggered correctly; below are the respected results for
# delay=0.75 (not 100% sure that this is correct, though); please keep in mind that
# adding/removing delegations in timescales smaller than the prop. delay of a link
# is far from realistic, it is just mentioned here for completeness (and because
# it might cause some sideeffects if the use case changes)
FLOW_TIMINGS = """{"DS->ES": {"f0": [3, 9]}, "DS->h1": {}, "DS->h2": {"f0": [0.75, 11.5]}, 
"ES->DS": {"f0": [3.0, 9.75]}, "h1->DS": {"f0": [0, 10.75]}, "h2->DS": {}}"""

# with delay = 0.25
FLOW_TIMINGS = """{"DS->ES": {"f0": [3, 9]}, "DS->h1": {}, "DS->h2": {"f0": [0.25, 10.5]}, 
"ES->DS": {"f0": [3.0, 9.25]}, "h1->DS": {"f0": [0, 10.25]}, "h2->DS": {}}"""