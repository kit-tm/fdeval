# (not used for evaluation in the thesis, only an example)
# There are different ways to define experiments; This is one very simple way;
# run with "python3 main.py -f custom/pbce/exp100.py"

import random, math

from topo.custom.topo import Topo

class MyTopo( Topo ):
    "exp100"

    def __init__( self, ctx ):
        # Initialize
        Topo.__init__( self )
        self.ctx = ctx;
        cnt_hosts = self.paramInt('param_topo_num_hosts', 20)
        cnt_cores = self.paramInt('param_topo_num_cores', 3)
        cnt_edges = self.paramInt('param_topo_num_edges', 3)
        cnt_flows = self.paramInt('param_topo_num_flows', 5000)
        fullmesh_core = self.paramInt('param_topo_fullmesh_core', 1)

        cores = []
        hostnames = []
        for c in range(cnt_cores):
            core = self.addSwitch( 'core%d' % c, x=1, y=1)
            cores.append(core)
            for e in range(cnt_edges):
                edge = self.addSwitch( 'edge%d.%d' % (c, e), x=1, y=2 )
                self.addLink( core, edge )
                for i in range(cnt_hosts):
                    name = 'h%d.%d.%d' % (c, e, i)
                    host = self.addHost(name, x=1, y=3)
                    hostnames.append(name)
                    self.addLink( host, edge )

        # core switches are connected in a full mesh
        links = []
        if fullmesh_core == 1:
            for i in cores:
                for j in cores:
                    if not (i,j) in links and not (j,i) in links:
                        links.append((i,j))
                        self.addLink(i, j)
        else:
            # core switches are only connected to their direct neighbors
            for i, j in zip(cores, cores[1:]):
                if not (i,j) in links and not (j,i) in links:
                    links.append((i,j))
                    print('add', i, j)
                    self.addLink(i, j)
            self.addLink(cores[-1], cores[0])  
                      

        # add traffic for this host
        self.addTraffic(dict(
            fg_class='Random',
            #fg_seed=100, # use fixed seed (not implemented atm)
            fg_numflows = cnt_flows, # number of flows to generate in total
            fg_time_range = 300, # spread epochs over 300 seconds
            fg_shuffle_epoch = True, # randomize shuffle array 
            fg_epoch = [1,1,1,1.2,1.3,1.5,1.9,2.8,1.7,1.1,1,1,0.7,0.5,0.3],
            fg_source_set = [],
            fg_random_destination=hostnames))

        # call on_done if simulation is finished
        ctx.on_simulation_finished = self.on_done

    def on_done(self, ctx):
        
        flowtimes = []
        delegated_counter = []
        undelegated_counter = []
        delegated_miss_counter = []
        diffs = []

        for flow in ctx.flows:
            flowtimes.append(flow.finished_at - flow.start)
            time_actual = flow.finished_at - flow.start
            expected = flow.total_demand/float(flow.demand_per_tick) + flow.path_delay_summed
            diff = expected-time_actual
            if diff < 0.00000001:
                diffs.append(0)
            else:
                diffs.append(abs(expected-time_actual))

        print("on_done...")
        print("flowtimes_min:", min(flowtimes))
        print("flowtimes_max:", max(flowtimes))
        print("flowtimes_diff:", min(diffs), max(diffs), sum(diffs))

def get_topo(ctx):
    return MyTopo(ctx)

topos = { 'exp100': ( lambda: MyTopo() ) }