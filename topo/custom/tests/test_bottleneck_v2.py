from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class MyTopo( Topo ):
    "simple bottleneck test 2"

    def __init__( self, ctx ):

        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0.5))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # Initialize
        Topo.__init__( self )
        s1 = self.addSwitch( 's1', x=3, y=2, processing_delay=processing_delay )
        s2 = self.addSwitch( 's2', x=3, y=2, processing_delay=processing_delay )
        s3 = self.addSwitch( 's3', x=3, y=2, processing_delay=processing_delay )

        h1 = self.addHost( 'h1', x=4, y=1)
        h2 = self.addHost( 'h2',x=4, y=3)
        h3 = self.addHost( 'h3',x=4, y=3)
        self.addLink( h1, s1, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h2, s2, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h3, s3, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( s1, s2, capacity=10, propagation_delay=propagation_delay )
        self.addLink( s2, s3, capacity=10, propagation_delay=propagation_delay )
        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f0", fg_start=0, fg_demand=100, fg_duration=10, 
                fg_fixed_path=['h1', 's1', 's2', 'h2'], fg_color='g'), 
            dict(fg_class='Single', fg_label="f1", fg_start=5, fg_demand=100, fg_duration=10, 
                fg_fixed_path=['h1', 's1', 's2', 's3', 'h3'], fg_color='r'), 
            dict(fg_class='Single', fg_label="f2", fg_start=10, fg_demand=100, fg_duration=10, 
                fg_fixed_path=['h1', 's1', 's2', 's3', 'h3'], fg_color='purple'),  
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
   "s1->h1":{

   },
   "s1->s2":{
      "f0":[
         0.5,
         18.5
      ],
      "f1":[
         5.5,
         28.5
      ],
      "f2":[
         10.5,
         31.25
      ]
   },
   "s2->s1":{

   },
   "s2->h2":{
      "f0":[
         1.0,
         20.0
      ]
   },
   "s2->s3":{
      "f1":[
         6.0,
         29.0
      ],
      "f2":[
         11.0,
         31.416666666666668
      ]
   },
   "s3->s2":{

   },
   "s3->h3":{
      "f1":[
         6.5,
         29.5
      ],
      "f2":[
         11.5,
         31.583333333333336
      ]
   },
   "h1->s1":{
      "f0":[
         0,
         17.0
      ],
      "f1":[
         5,
         27.5
      ],
      "f2":[
         10,
         30.75
      ]
   },
   "h2->s2":{

   },
   "h3->s3":{

   }
}"""