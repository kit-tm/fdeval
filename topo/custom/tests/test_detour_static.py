import os

from topo.custom.topo import Topo

from . import testutil as testutil

class MyTopo( Topo ):

    def __init__( self, ctx ):
        "scenario with static path that has a 'loop' between ds and es"

        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0.5))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # initialize topology
        Topo.__init__( self )
        ds = self.addSwitch( 'ds', x=3, y=2, processing_delay=processing_delay )
        es = self.addSwitch( 'es', x=2, y=2, processing_delay=processing_delay )
        es2 = self.addSwitch( 'es2', x=1, y=2, processing_delay=processing_delay )
        h1 = self.addHost( 'h1', x=4, y=1)
        h2 = self.addHost( 'h2',x=4, y=3)
        self.addLink( h1, ds, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h2, ds, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( ds, es, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( es, es2, capacity=1000, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="Delegated", 
                fg_start=6,  fg_demand=50, fg_duration=5, 
                fg_fixed_path=['h1', 'ds', 'es', 'ds', 'h2']),
            dict(fg_class='Single', fg_label="Normal", 
                fg_start=0,  fg_demand=50, fg_duration=5, 
                fg_fixed_path=['h1', 'ds', 'h2']),
            dict(fg_class='Single', fg_label="Multi-Hop Delegated", 
                fg_start=15,  fg_demand=50, fg_duration=5, 
                fg_fixed_path=['h1', 'ds', 'es', 'es2', 'es', 'ds', 'h2']),    
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
   "ds->h1":{

   },
   "ds->h2":{
      "Delegated":[
         7.5,
         13.0
      ],
      "Normal":[
         0.5,
         6.0
      ],
      "Multi-Hop Delegated":[
         17.5,
         23.0
      ]
   },
   "ds->es":{
      "Delegated":[
         6.5,
         12.0
      ],
      "Multi-Hop Delegated":[
         15.5,
         21.0
      ]
   },
   "es->ds":{
      "Delegated":[
         7.0,
         12.5
      ],
      "Multi-Hop Delegated":[
         17.0,
         22.5
      ]
   },
   "es->es2":{
      "Multi-Hop Delegated":[
         16.0,
         21.5
      ]
   },
   "es2->es":{
      "Multi-Hop Delegated":[
         16.5,
         22.0
      ]
   },
   "h1->ds":{
      "Delegated":[
         6,
         11.5
      ],
      "Normal":[
         0,
         5.5
      ],
      "Multi-Hop Delegated":[
         15,
         20.5
      ]
   },
   "h2->ds":{

   }
}"""