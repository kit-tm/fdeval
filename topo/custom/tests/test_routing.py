from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class TestEngine(Engine):

    def on_EVSwitchStats(self, switch, ev):
        # f0 = ['h1', 's1', 'DS', 's3', 's4', 'h2']
        if switch.label == 'DS':
            if math.isclose(ev.tick, 5):
                print("@", ev.tick, "change route for f0")
                for id, flow in self.active_flows.items():
                    if flow.label == 'f0':
                        new_path = self.ctx.topo.path_to_id(['h1', 's1', 'DS', 's2', 's4', 'h2'])
                        flow.change_path(ev.tick, new_path)

            if math.isclose(ev.tick, 9):
                print("@", ev.tick, "change route for f0")
                for id, flow in self.active_flows.items():
                    if flow.label == 'f0':
                        new_path = self.ctx.topo.path_to_id(['h1', 's1', 'DS', 's3', 's4', 'h2'])
                        flow.change_path(ev.tick, new_path)

        # create report
        super().on_EVSwitchStats(switch, ev)

class MyTopo( Topo ):
    "change a path proactively after some timeout and then change it back"

    def __init__( self, ctx ):

        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0.5))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # Initialize
        Topo.__init__( self )
        ds = self.addSwitch( 'DS', x=2, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        es = self.addSwitch( 'ES', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        s1 = self.addSwitch( 's1', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        s2 = self.addSwitch( 's2', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        s3 = self.addSwitch( 's3', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        s4 = self.addSwitch( 's4', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        h1 = self.addHost( 'h1', x=4, y=1 )
        h2 = self.addHost( 'h2',x=4, y=3 )

        self.addLink( ds, es, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( ds, s1, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( ds, s2, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( ds, s3, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( s3, s2, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( s3, s4, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( s4, s2, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h1, s1, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h2, s4, capacity=1000, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f0", fg_start=0,  fg_demand=100, fg_duration=10, 
                fg_fixed_path=['h1', 's1', 'DS', 's3', 's4', 'h2']),
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

topos = { 'MyTopo': ( lambda: MyTopo() ) }

FLOW_TIMINGS = """{
   "DS->ES":{

   },
   "DS->s1":{

   },
   "DS->s2":{
      "f0":[
         5,
         9
      ]
   },
   "DS->s3":{
      "f0":[
         1.0,
         11.5
      ]
   },
   "ES->DS":{

   },
   "s1->DS":{
      "f0":[
         0.5,
         11.0
      ]
   },
   "s1->h1":{

   },
   "s2->DS":{

   },
   "s2->s3":{

   },
   "s2->s4":{
      "f0":[
         5.0,
         9.5
      ]
   },
   "s3->DS":{

   },
   "s3->s2":{

   },
   "s3->s4":{
      "f0":[
         1.5,
         12.0
      ]
   },
   "s4->s2":{

   },
   "s4->s3":{

   },
   "s4->h2":{
      "f0":[
         2.0,
         12.5
      ]
   },
   "h1->s1":{
      "f0":[
         0,
         10.5
      ]
   },
   "h2->s4":{

   }
}"""