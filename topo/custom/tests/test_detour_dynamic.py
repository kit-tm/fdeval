from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class TestEngine(Engine):

    def on_EVSwitchStats(self, switch, ev):
        # this engine will statically delegate all currently active flows 
        # from DS-->ES at tick 5
        if switch.label == 'DS':
            if math.isclose(ev.tick, 5):
                for id, flow in self.active_flows.items():
                    print("@", ev.tick, "change route for flow=%s" % flow.id)
                    if flow.label == 'f0':
                        new_path = self.ctx.topo.path_to_id(['h1', 'DS', 'ES', 'DS', 's1', 'h2'])
                        flow.change_path(ev.tick, new_path)

        # create report
        super().on_EVSwitchStats(switch, ev)

class MyTopo( Topo ):
    "most simple dynamic detour"

    def __init__( self, ctx ):

        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0.5))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # Initialize
        Topo.__init__( self )
        ds = self.addSwitch( 'DS', x=2, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        es = self.addSwitch( 'ES', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        s1 = self.addSwitch( 's1', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay))
        h1 = self.addHost( 'h1', x=4, y=1)
        h2 = self.addHost( 'h2',x=4, y=3)

        self.addLink( ds, es, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( ds, s1, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h1, ds, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h2, s1, capacity=1000, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f0", fg_start=0,  fg_demand=100, fg_duration=10, 
                fg_fixed_path=['h1', 'DS', 's1', 'h2']),
        )

        # call on_done if simulation is finished
        ctx.on_test_finished = self.on_done

    def on_done(self, ctx):
        testutil.print_summary(ctx)

        #print(testutil.get_flow_timings(ctx))
        errors = []
        errors += testutil.verify_flow_timings(ctx, FLOW_TIMINGS)
        return errors

def get_topo(ctx):
    return MyTopo(ctx)

topos = { 'myTopo': ( lambda: MyTopo() ) }

FLOW_TIMINGS = """{
   "DS->ES":{
      "f0":[
         5,
         11.0
      ]
   },
   "DS->s1":{
      "f0":[
         0.5,
         12.0
      ]
   },
   "DS->h1":{

   },
   "ES->DS":{
      "f0":[
         5.0,
         11.5
      ]
   },
   "s1->DS":{

   },
   "s1->h2":{
      "f0":[
         1.0,
         12.5
      ]
   },
   "h1->DS":{
      "f0":[
         0,
         10.5
      ]
   },
   "h2->s1":{

   }
}"""