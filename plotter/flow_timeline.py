import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import math


def plot(ctx):
    """Creates a timeline plot showing links and flows of a scenario.
    Expects a Scenario object as input (to access link and flow definitions);
    """
    levels = np.array([-5, 5, -3, 3, -1, 1])
    fig, ax = plt.subplots(figsize=(16, 8))

    plt.xticks(np.arange(0, 30, 1.0))
    ax.xaxis.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    # get some important statistics to be displayed in the top left
    capacities = []
    prop_delays = []
    proc_delays = []
    for switch_id, opts in ctx.topo.graph.nodes(data=True):
        if opts.get('_switch'):
            d = opts.get('_switch').engine.processing_delay
            if not d in proc_delays: proc_delays.append(d)
    for link_id, n,  opts in ctx.topo.graph.edges(data=True):
        if opts.get('_link'):
            d = opts.get('_link').propagation_delay
            c = opts.get('_link').capacity
            if not d in prop_delays: prop_delays.append(d)
            if not c in capacities: capacities.append(c)

    capacities = ','.join(['%d' % x for x in capacities if x != None])
    prop_delays = ','.join(['%.2f' % x for x in prop_delays if x != None])
    proc_delays = ','.join(['%.2f' % x for x in proc_delays if x != None])

    print(capacities, prop_delays, proc_delays)

    ax.text(0.45, 1,'propagation delay: %s\nprocessing delay: %s\nlink capacity: %s\n' % 
        (prop_delays, proc_delays, capacities),
        horizontalalignment='left', transform = ax.transAxes)

    flow_count  = 0

    for link_id in ctx.topo.graph.edges():
        for flow in ctx.flows:

            if not hasattr(flow, 'color'):
                setattr(flow, 'color', np.random.rand(3,))

            data = filter(lambda e: e.get("link") == link_id, flow.flow_history)  
            data = sorted(data, key=lambda t: t.get('tick'))
            if len(data) == 0: continue;

            pcc = -1 # iterate all pccs of this flow (pcc = path_change_counter, see core/flow.py)
            cnt = 0 # safety counter
            demand_sum = 0
            while len(data) > 0 and cnt < 50:
                current_pcc = list(filter(lambda e: e.get("pcc") == pcc, data))
                remaining = list(filter(lambda e: e.get("pcc") != pcc, data))
                data = remaining
                pcc += 1
                cnt += 1

                # print active time for the flows
                if len(current_pcc) >= 2:

                    start = current_pcc[0].get('tick')
                    end = current_pcc[-1].get('tick')
                    demand = current_pcc[-1].get('demand')
                    
                    annotation = '%.0f' % demand
                    FLOW_HEIGHT = 0.2
                    #if len(current_pcc) == 2:
                    #    ax.text(start+0.1, flow_count+0.05, annotation )
                    # top line

                    ax.text(start+0.1, flow_count-0.07, str(pcc-1), color='r')

                    ax.plot((start, end), (flow_count+FLOW_HEIGHT, flow_count+FLOW_HEIGHT), 'k', alpha=.5)
                    # bottom line
                    ax.plot((start, end), (flow_count, flow_count), 'k', alpha=.5)
                    # line at the end
                    ax.plot((end, end), (flow_count, flow_count+FLOW_HEIGHT), 'r', alpha=.5)
                    
                    color = flow.color
                    if flow.flow_gen.get('fg_color'):
                        color = flow.flow_gen.get('fg_color')


                    rect = patches.Rectangle((start, flow_count+FLOW_HEIGHT),end-start, -FLOW_HEIGHT,
                        linewidth=1,edgecolor='none',facecolor=color, alpha=0.3)
                    ax.add_patch(rect)

                    # color propagation delay as grey
                    if current_pcc[0].get('event') == "EVLinkFlowAdd":
                        #assert(current_pcc[1].get('event') == "EVLinkFlowAdded")
                        start2 = current_pcc[0].get('tick')
                        end2 = current_pcc[1].get('tick')
                        rect = patches.Rectangle((start2, flow_count+FLOW_HEIGHT),end2-start2, -FLOW_HEIGHT,
                            linewidth=1,edgecolor='none',facecolor='grey', alpha=0.3)
                        ax.add_patch(rect)                     

                # print intermediate events; only if added/removed events are present
                if len(current_pcc) > 2:
                    last = current_pcc[0]
                    for ev in current_pcc[1:]:

                        d = ev.get("demand")
                        if d > 0:
                            demand_sum += d
                            ax.plot((ev.get('tick'), ev.get('tick')), (flow_count, flow_count+FLOW_HEIGHT), 'r', alpha=.5)   
                            ax.text(ev.get('tick')-0.1, flow_count+0.05, '%.1f' % d,
                                horizontalalignment='right')
                            last = ev

            link_label = ctx.topo.print_link(link_id)

            ax.text(end+0.4, flow_count+0.05, '(%s), Flow=%s' % (link_label, flow.label))

            ax.text(end+0.4, flow_count-0.2, 'demand=%.2f in %.2fs' % (demand_sum, flow.duration_bck), color='grey')

            flow_count += 1

    plt.show()

    return


    flow_count = 0 # flows are shown as lines with different y values 

    for link_id in ctx.topo.graph.edges():
        for flow in ctx.flows:
            data = filter(lambda e: e.get("link") == link_id, flow.flow_history)  

            if not hasattr(flow, 'color'):
                setattr(flow, 'color', np.random.rand(3,))

            setattr(flow, 'demand_sum', 0)

            data = sorted(data, key=lambda t: t.get('tick'))
            if len(data) == 0: continue;


            # calculate effectice util 
            """
            what is this? --> there can be multiple events at the same tick that change the utilization; 
            example below shows the events for a flow where two flows trigger an event; the first flow 
            is added and increases the utilization to 1.5; the second flow is removed and reduces the 
            utilization back to 1.0; in this case, it is important to actually use the 1.0 value and NOT the
            1.5 value to calculate the demands; this is why the "effective_util" dict is calculated here.

            example1

            === data for link=1 and flow=2 (4 entries)
            {'link': 1, 'tick': 5, 'util': 1.0, 'event': <core.events.EVLinkFlowAdded object at 0x7f60e052e6d8>}
            {'link': 1, 'tick': 10, 'util': 1.5, 'event': <core.events.EVLinkUpdateOnAdded object at 0x7f60e0541a20>}
            {'link': 1, 'tick': 10, 'util': 1.0, 'event': <core.events.EVLinkUpdateOnRemoved object at 0x7f60e0541b00>}
            {'link': 1, 'tick': 15.0, 'util': 0.5, 'event': <core.events.EVLinkUpdateOnFinished object at 0x7f60e0541a58>}

            example2 (with calculated effective values in the last column); at tick 10, there are 4 events but only the
            last event is relevant, i.e., effective_util[10] == 2.0; this can be read as "from tick 10 on, there is a 
            utilization on the link equal to 2.0 * linkCapacity"

            === data for link=1 and flow=3 (7 entries)
            {'link': 1, 'tick': 10, 'util': 1.5, 'event': <core.events.EVLinkFlowAdded object at 0x7fac37737e48>} 2.0
            {'link': 1, 'tick': 10, 'util': 1.0, 'event': <core.events.EVLinkUpdateOnRemoved object at 0x7fac37747dd8>} 2.0
            {'link': 1, 'tick': 10, 'util': 1.5, 'event': <core.events.EVLinkUpdateOnAdded object at 0x7fac37747e48>} 2.0
            {'link': 1, 'tick': 10, 'util': 2.0, 'event': <core.events.EVLinkUpdateOnAdded object at 0x7fac37747f28>} 2.0
            {'link': 1, 'tick': 20.0, 'util': 1.5, 'event': <core.events.EVLinkUpdateOnRemoved object at 0x7fac37758400>} 1.5
            {'link': 1, 'tick': 20.0, 'util': 1.5, 'event': <core.events.EVLinkUpdateOnRemoveDelayed object at 0x7fac37747f98>} 1.5
            {'link': 1, 'tick': 27.5, 'util': 0.5, 'event': <core.events.EVLinkUpdateOnFinished object at 0x7fac37747c18>} 0.5

            """ 
            effective_util = {}
            for el in data:
                check = list(filter(lambda e: e.get("tick") == el.get("tick"), data))
                if len(check) > 1:
                    effective_util[el.get("tick")] = check[-1].get("util")
                else:
                    effective_util[el.get("tick")] = check[0].get("util")

            
            last = data[0]
            print("=== data for link=%s and flow=%d (%d entries)" % (str(link_id), flow.id, len(data)))
            #for e in data:
            #    print(e, effective_util.get(e.get("tick")))

            #print(" %d : %f " % (last.get("tick"), last.get("util")))


            index = 1
            for current in data[1:]:
                util = current.get("util")
                start = last.get('tick')
                end = current.get('tick') 
                util_effective = util
                cnt = index
                while cnt < len(data)-1 and data[cnt+1] and data[cnt+1].get("tick") == end:
                    util_effective = data[cnt+1].get("util")
                    cnt += 1

                diff = current.get("tick") - last.get("tick") # time passed
                effective_demand_per_tick = flow.demand_per_tick
                if effective_util.get(last.get("tick")) > 1:
                    effective_demand_per_tick = flow.demand_per_tick*(1/effective_util.get(last.get("tick"))) # demand handled until now
  
                demand = effective_demand_per_tick * diff
                
                print(" [%.2f-%.2f] : %f %f demand=%f" % (start, end, last.get("util"), effective_util.get(last.get("tick")), demand))

                # it can happen that start and end tick in a traced event are the same, e.g., if two
                # update events occur in the same tick (like two flows are added at the exact same time);
                # these events are not plotted here
                index += 1
                if start==end: continue;

                FLOW_HEIGHT = 0.2


                annotation = '%.2f' % demand

                if last.get("event") == 'EVLinkFlowAdded' or last.get("event") == 'EVLinkFlowResume':
                    delay = ctx.topo.graph.edges[link_id]['_link'].propagation_delay
                    demand -= delay*effective_demand_per_tick
                    annotation = '%.2f (c)' % demand

                if last.get("event") == "EVLinkFlowStopped":
                    annotation = ""
                    demand=0
                else:
                     # filled rectangle for the flow
                    rect = patches.Rectangle((start, flow_count+FLOW_HEIGHT),end-start, -FLOW_HEIGHT,
                        linewidth=1,edgecolor='none',facecolor=flow.color, alpha=0.3)
                    ax.add_patch(rect)

            
                ax.text(start+0.1, flow_count+0.05, annotation )
                # top line
                ax.plot((start, end), (flow_count+FLOW_HEIGHT, flow_count+FLOW_HEIGHT), 'k', alpha=.5)
                # bottom line
                ax.plot((start, end), (flow_count, flow_count), 'k', alpha=.5)
                # line at the end
                ax.plot((end, end), (flow_count, flow_count+FLOW_HEIGHT), 'r', alpha=.5)



                flow.demand_sum += demand
                last = current
            label = flow.label
            if label == None: label = str(flow.id)

            
            end1 = ctx.topo.graph.nodes[link_id[0]]['label']
            end2 = ctx.topo.graph.nodes[link_id[1]]['label']

            ax.text(end+0.4, flow_count+0.05, 'Link=(%s->%s), Flow=%s' % (str(end1),str(end2), label))

            ax.text(end+0.4, flow_count-0.2, 'demand=%.2f' % (flow.demand_sum), color='grey')


       
            flow_count += 1

    plt.show()
    #pp = PdfPages("work_in_progress.pdf")
    #pp.savefig()
    #pp.close()
    #plt.close()