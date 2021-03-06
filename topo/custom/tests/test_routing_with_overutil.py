from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class TestEngine(Engine):

    def on_EVSwitchStats(self, switch, ev):
        # between t=5 and t=9, the overload from link s3-s4 is removed
        # by rerouting flow f0 via the alternate path s2-s4
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
    "change a path proactively after some timeout and trigger overutilization"

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
        h3 = self.addHost( 'h3', x=4, y=1 )

        self.addLink( ds, es, capacity=100, propagation_delay=propagation_delay )
        self.addLink( ds, s1, capacity=100, propagation_delay=propagation_delay )
        self.addLink( ds, s2, capacity=100, propagation_delay=propagation_delay )
        self.addLink( ds, s3, capacity=100, propagation_delay=propagation_delay )
        self.addLink( s3, s2, capacity=100, propagation_delay=propagation_delay )
        self.addLink( s3, s4, capacity=10, propagation_delay=propagation_delay )
        self.addLink( s4, s2, capacity=10, propagation_delay=propagation_delay )
        self.addLink( h1, s1, capacity=100, propagation_delay=propagation_delay )
        self.addLink( h2, s4, capacity=100, propagation_delay=propagation_delay )
        self.addLink( h3, s1, capacity=100, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(         
            # green flow   
            dict(fg_class='Single', fg_label="f0", fg_color="g", fg_start=0,  fg_demand=100, 
                fg_duration=10, fg_fixed_path=['h1', 's1', 'DS', 's3', 's4', 'h2']),
            # red flow
            dict(fg_class='Single', fg_label="f1", fg_color="r", fg_start=0,  fg_demand=100, 
                fg_duration=10, fg_fixed_path=['h1', 's1', 'DS', 's3', 's4', 'h2']),   
        )
        # 

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

def get_topo(ctx):
    return MyTopo(ctx)

topos = { 'myTopo': ( lambda: MyTopo() ) }


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
         17.5
      ],
      "f1":[
         1.0,
         17.0
      ]
   },
   "ES->DS":{

   },
   "s1->DS":{
      "f0":[
         0.5,
         16.5
      ],
      "f1":[
         0.5,
         16.0
      ]
   },
   "s1->h1":{

   },
   "s1->h3":{

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
         18.0
      ],
      "f1":[
         1.5,
         18.0
      ]
   },
   "s4->s2":{

   },
   "s4->s3":{

   },
   "s4->h2":{
      "f0":[
         2.0,
         19.0
      ],
      "f1":[
         2.0,
         18.5
      ]
   },
   "h1->s1":{
      "f0":[
         0,
         15.5
      ],
      "f1":[
         0,
         15.0
      ]
   },
   "h2->s4":{

   },
   "h3->s1":{

   }
}"""