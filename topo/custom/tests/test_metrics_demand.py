from topo.custom.topo import Topo
from . import testutil as testutil
import math
from core.engine import Engine

class TestEngine(Engine):

    def on_stats(self, switch, ev):
        # this engine will statically delegate all currently active flows 
        # from DS-->ES at tick 5 and undo the delegation at tick 9
        if switch.label == 'DS':
            if math.isclose(ev.tick, 5):
                print("@", ev.tick, "trigger redirect")
                for id, flow in self.active_flows.items():
                    #if flow.label == 'f0':
                    print("@", ev.tick, "redirect flow", flow.id )
                    extension_switch = self.ctx.topo.get_switch_by_label('ES')
                    self.add_delegation(ev.tick, flow, switch, extension_switch) 


            if math.isclose(ev.tick, 7):
                print("@", ev.tick, "trigger undo redirect")
                for id, flow in self.active_flows.items():
                    if flow.label == 'f0' or flow.label == 'f1':
                      print("@", ev.tick, "undo redirection flow", flow.id )
                      extension_switch = self.ctx.topo.get_switch_by_label('ES')
                      self.remove_delegation(ev.tick, flow, switch, extension_switch) 

            if math.isclose(ev.tick, 10):
                print("@", ev.tick, "trigger redirect (second time)")
                for id, flow in self.active_flows.items():
                    if flow.label == 'f0' or flow.label == 'f1':
                      print("@", ev.tick, "redirect flow", flow.id )
                      extension_switch = self.ctx.topo.get_switch_by_label('ES')
                      self.add_delegation(ev.tick, flow, switch, extension_switch) 

            if math.isclose(ev.tick, 13):
                print("@", ev.tick, "trigger undo redirect (second time)")
                for id, flow in self.active_flows.items():
                    if flow.label == 'f0':
                      print("@", ev.tick, "undo redirection flow", flow.id )
                      extension_switch = self.ctx.topo.get_switch_by_label('ES')
                      self.remove_delegation(ev.tick, flow, switch, extension_switch) 

            if math.isclose(ev.tick, 16):
                print("@", ev.tick, "trigger redirect (third time)")
                for id, flow in self.active_flows.items():
                    if flow.label == 'f0':
                      print("@", ev.tick, "redirect flow", flow.id )
                      extension_switch = self.ctx.topo.get_switch_by_label('ES')
                      self.add_delegation(ev.tick, flow, switch, extension_switch) 

            if math.isclose(ev.tick, 20):
                print("@", ev.tick, "trigger undo redirect (third time)")
                for id, flow in self.active_flows.items():
                    if flow.label == 'f0':
                      print("@", ev.tick, "undo redirection flow", flow.id )
                      extension_switch = self.ctx.topo.get_switch_by_label('ES')
                      self.remove_delegation(ev.tick, flow, switch, extension_switch) 


class MyTopo( Topo ):
    "test correct calculation of metrics.demand_delegated"

    def __init__( self, ctx ):

        propagation_delay = float(ctx.config.get("topo.propagation_delay", 0))
        processing_delay = float(ctx.config.get("topo.processing_delay", 0))

        # Initialize
        Topo.__init__( self )
        ds = self.addSwitch( 'DS', x=2, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        es = self.addSwitch( 'ES', x=1, y=1, engine=TestEngine(ctx, processing_delay=processing_delay) )
        h1 = self.addHost( 'h1', x=4, y=1 )
        h2 = self.addHost( 'h2',x=4, y=3 )

        self.addLink( ds, es, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h1, ds, capacity=1000, propagation_delay=propagation_delay )
        self.addLink( h2, ds, capacity=1000, propagation_delay=propagation_delay )

        # add traffic
        self.addTraffic(            
            dict(fg_class='Single', fg_label="f0", fg_start=0,  fg_demand=100, fg_duration=25, 
                fg_fixed_path=['h1', 'DS', 'h2']),
            dict(fg_class='Single', fg_label="f1", fg_start=0,  fg_demand=100, fg_duration=25, 
                fg_fixed_path=['h1', 'DS', 'h2']), 
            dict(fg_class='Single', fg_label="f2", fg_start=0,  fg_demand=100, fg_duration=25, 
                fg_fixed_path=['h1', 'DS', 'h2']) 

        )

        # call on_done if simulation is finished
        ctx.on_test_finished = self.on_done

    def on_done(self, ctx):
        testutil.print_summary(ctx)
        errors = []
        expected = int(ctx.statistics['metrics.demand_delegated'])
        if not expected == 184:
            errors.append('expected metrics.demand_delegated=184, got %d' % expected)
        return errors

def get_topo(ctx):
    return MyTopo(ctx)

topos = { 'myTopo': ( lambda: MyTopo() ) }

