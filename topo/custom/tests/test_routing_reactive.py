from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class TestEngine(Engine):

    def on_EVSwitchStats(self, switch, ev):
        super().on_EVSwitchStats(switch, ev)

    def on_EVSwitchNewFlow(self, switch, ev):
        if switch.label == 'DS':
          if ev.flow.delegation.is_redirected:
            print("flow %s was already redirected, skip" % ev.flow.label)
          else:
            print("reroute flow", ev.flow.label)
            extension_switch = self.ctx.topo.get_switch_by_label('ES')
            ev.flow.delegation.is_redirected = True          
            self.add_delegation(ev.tick, ev.flow, switch, extension_switch)
        # forward flow on next switch in path
        super().on_EVSwitchNewFlow(switch, ev)

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
        for i in range(3):
          self.addTraffic(            
              dict(fg_class='Single', fg_label="f%d" % i, fg_start=i,  fg_demand=100, fg_duration=10, 
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
      "f0":[
         1.0,
         11.5
      ],
      "f1":[
         2.0,
         12.5
      ],
      "f2":[
         3.0,
         13.5
      ]
   },
   "DS->s1":{

   },
   "DS->s2":{

   },
   "DS->s3":{
      "f0":[
         2.0,
         12.5
      ],
      "f1":[
         3.0,
         13.5
      ],
      "f2":[
         4.0,
         14.5
      ]
   },
   "ES->DS":{
      "f0":[
         1.5,
         12.0
      ],
      "f1":[
         2.5,
         13.0
      ],
      "f2":[
         3.5,
         14.0
      ]
   },
   "s1->DS":{
      "f0":[
         0.5,
         11.0
      ],
      "f1":[
         1.5,
         12.0
      ],
      "f2":[
         2.5,
         13.0
      ]
   },
   "s1->h1":{

   },
   "s2->DS":{

   },
   "s2->s3":{

   },
   "s2->s4":{

   },
   "s3->DS":{

   },
   "s3->s2":{

   },
   "s3->s4":{
      "f0":[
         2.5,
         13.0
      ],
      "f1":[
         3.5,
         14.0
      ],
      "f2":[
         4.5,
         15.0
      ]
   },
   "s4->s2":{

   },
   "s4->s3":{

   },
   "s4->h2":{
      "f0":[
         3.0,
         13.5
      ],
      "f1":[
         4.0,
         14.5
      ],
      "f2":[
         5.0,
         15.5
      ]
   },
   "h1->s1":{
      "f0":[
         0,
         10.5
      ],
      "f1":[
         1,
         11.5
      ],
      "f2":[
         2,
         12.5
      ]
   },
   "h2->s4":{

   }
}"""