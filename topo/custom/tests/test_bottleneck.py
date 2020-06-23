from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class MyTopo( Topo ):
    "simple bottleneck test"

    def __init__( self, ctx ):
        """
        Two flows compete for the bottleneck;
        green flow has a datarate of 5/sec
        red flow has a datarate of 20/sec
        --> fairshare datarate is 8/sec for red and 1/sec for green; 
        (h1->s1) Flow F1 data:
            red: 
                5.5 to 18 = 12.5 * 8 = 100
            green: 
                0.5 to 5.5 with 5 = 25
                5.5 to 18 with 1 = 12.5
                18 to 30.5 with 5 = 12.5*5 = 62.5
                sum is 100
            spare capacity?
        """
        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0.5))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # Initialize
        Topo.__init__( self )
        s1 = self.addSwitch( 's1', x=3, y=2, processing_delay=processing_delay )
        s2 = self.addSwitch( 's2', x=3, y=2, processing_delay=processing_delay )
        s3 = self.addSwitch( 's3', x=3, y=2, processing_delay=processing_delay )

        host1 = self.addHost( 'h1', x=4, y=1)
        host2 = self.addHost( 'h2',x=4, y=3)
        self.addLink( host1, s1, capacity=20, propagation_delay=propagation_delay )
        self.addLink( s1, s2, capacity=20, propagation_delay=propagation_delay )
        self.addLink( s2, s3, capacity=10, propagation_delay=propagation_delay )
        self.addLink( host2, s3, capacity=20, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f0", fg_start=0, fg_demand=100, fg_duration=20, 
                fg_fixed_path=['h1', 's1', 's2', 's3', 'h2'], fg_color='g'), 
            dict(fg_class='Single', fg_label="f1", fg_start=5, fg_demand=100, fg_duration=5, 
                fg_fixed_path=['h1', 's1', 's2', 's3', 'h2'], fg_color='r'),   
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
         32.2
      ],
      "f1":[
         5.5,
         18.5
      ]
   },
   "s2->s1":{

   },
   "s2->s3":{
      "f0":[
         1.0,
         32.7
      ],
      "f1":[
         6.0,
         19.0
      ]
   },
   "s3->s2":{

   },
   "s3->h2":{
      "f0":[
         1.5,
         33.2
      ],
      "f1":[
         6.5,
         19.5
      ]
   },
   "h1->s1":{
      "f0":[
         0,
         31.7
      ],
      "f1":[
         5,
         18.0
      ]
   },
   "h2->s3":{

   }
}"""