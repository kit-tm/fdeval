import os

from topo.custom.topo import Topo

from . import testutil as testutil

class MyTopo( Topo ):

    def __init__( self, ctx ):
        "very simple overutilization scenario without any delays"

        capacity = float(ctx.config.get("topo.capacity", 10))
        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # initialize topology
        Topo.__init__( self )
        s1 = self.addSwitch( 's1', x=3, y=2, processing_delay=processing_delay )
        host1 = self.addHost( 'h1', x=4, y=1)
        host2 = self.addHost( 'h2',x=4, y=3)
        self.addLink( host1, s1, capacity=capacity, propagation_delay=propagation_delay )
        self.addLink( host2, s1, capacity=capacity, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f0", fg_start=0,  fg_demand=100, fg_duration=10, 
                fg_fixed_path=['h1', 's1', 'h2']),
            dict(fg_class='Single', fg_label="f1", fg_start=5,  fg_demand=100, fg_duration=10, 
                fg_fixed_path=['h1', 's1', 'h2']),
            dict(fg_class='Single', fg_label="f2", fg_start=8,  fg_demand=20, fg_duration=2, 
                fg_fixed_path=['h1', 's1', 'h2']),  
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
   "s1->h1":{

   },
   "s1->h2":{
      "f0":[
         0.0,
         17.0
      ],
      "f1":[
         5.0,
         22.0
      ],
      "f2":[
         8.0,
         14.0
      ]
   },
   "h1->s1":{
      "f0":[
         0,
         17.0
      ],
      "f1":[
         5,
         22.0
      ],
      "f2":[
         8,
         14.0
      ]
   },
   "h2->s1":{

   }
}"""