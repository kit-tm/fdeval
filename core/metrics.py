import math
import logging
import networkx as nx

logger = logging.getLogger(__name__)

class Metric():
    def __init__(self, arr):
        if len(arr) == 0:
            self.sum = 0
            self.min = 0
            self.max = 0
            return
        self.sum = sum(arr)
        self.min = min(arr)
        self.max = max(arr)

class Metrics():
    def __init__(self):
        self.overhead = None
        self.underutil = None
        self.overutil = None
        self.backdelegations = None

class LinkUtil():
    """Helper construct to store link utilizatation values"""

    def __init__(self, link):
        self.link = link
        self.demand_total = 0
        self.demand_delegated = 0
        self.demand_over_time = []

        for i in range(0, 450, 1):
            self.demand_over_time.append(0)


    def update(self, t1, t2, demand, delegated=False, reverseDelegated=False):
        # store total demand processed by this link
        self.demand_total += (t2-t1)*demand
        if delegated:
            self.demand_delegated += (t2-t1)*demand
        # store link utilization over time
        lower = int(t1)
        upper = int(t2)+1
        for t in range(lower, upper):
            d = 0
            if t1 >= t and t1 < t+1:
                d = ((t+1)-t1)*demand
            if t2 > t and t2 <= t+1:
                if t1 > t:
                    d = (t2-t1)*demand
                else:
                    d = (t2-t)*demand
            if t >= t1 and t+1 <= t2:
                d = demand
            self.demand_over_time[t] += d

def calculate_standard_metrics(ctx):
    """
    This function is executed once after every simulation and calculates
    several standard metrics
    """
    metrics = {}
    if ctx.config.get("param_topo_switch_capacity"):
        thresh = int(ctx.config.get("param_topo_switch_capacity"))
    else: 
        thresh = 1000000000000

    # get link utilization data
    links = [opts['_link'] for _, _, opts in ctx.topo.graph.edges(data=True) if opts.get('_link')]

    linkMap = dict()
    for link in links:
        #logger.info("link " + str(link))
        #logger.info("  " + link.label + "," + str(link.id))
        linkMap[link.id] = LinkUtil(link)

    logger.debug("prepare link utilization data")
    maxflowtmp = 0
    maxflowtmp_flow = None
    dtotal_all = 0

    cnt = 0
    #for flow in sorted(ctx.flows, key=lambda flow: len(flow.delegation.reports), reverse=True):
    stats_len_usepath = []
    for flow in ctx.flows:
        # first recalculate the shortest path from source to target for this flow; this
        # is required because the currently stored path in the flow object might have been changed
        # during the simulation; please also note that it is assumed here that flows are static and defined 
        # by their shortest path between source and target (the metric fails / gets invalid if this
        # assumption is violated!)
        path_tuples = []
        path = []
        if flow.target:
            path = nx.shortest_path(ctx.topo.graph, source=flow.source.id, target=flow.target.id)
            path_tuples = list(zip(path[0:], path[1:]))
        else:
            raise RuntimeError("link metrics could not be calculated, flow.target is undefined")

        if len(flow.delegation.reports) > 1: 
            logger.debug("==============> flow=%s start=%f end=%f" % (flow.label, flow.start, flow.start+flow.duration))
            
            for r1 in flow.delegation.reports:
                src = ctx.topo.get_label_by_id(r1.source)
                tgt = ctx.topo.get_label_by_id(r1.target)
                logger.debug("  t=%f src=%s target=%s action=%d current_path=%s  " % (r1.tick, 
                    src, tgt, r1.action, str(r1.current_path)))
            logger.debug("")

            flow_history = {}
            flow_history[flow.start] = (path, [], []) # we start with the original path
            delegation_links = []
            delegation_links_reverse = []
            es_list = []
            es_list_reverse = []
            for report in flow.delegation.reports:
                #if report.action in [1,3,0]:
                # not that later entries with the same tick will overwrite
                # existing entries, which is intended!
                if report.action == 1:
                    es_list.append((report.source, report.target))
                    es_list_reverse.append((report.target, report.source))
                if report.action == 2:
                    es_list.remove((report.source, report.target))
                    es_list_reverse.remove((report.target, report.source))
                if report.action == 3:
                    es_list.append((report.source, report.target))
                    es_list_reverse.append((report.target, report.source))
                if report.action == 0:
                    es_list.remove((report.source, report.target))
                    es_list_reverse.remove((report.target, report.source))

                if report.action == 1 or report.action == 3 or report.action == 0:
                    flow_history[report.tick] = (report.current_path, es_list[:], es_list_reverse[:])
            flow_history[flow.start+flow.duration] = (path, [], [])
            for tick, path in flow_history.items():
                logger.debug("  %f - %s" % (tick, str(path)))  

            logger.debug("")
            dtotal = 0
            for t1, t2 in zip(list(flow_history.keys()), list(flow_history.keys())[1:]):
                use_time = t2-t1
                use_path, es_list, es_list_reverse = flow_history.get(t1)

                stats_len_usepath.append(len(use_path))
                logger.debug("  %f - %f = %f seconds with %s es_list=%s" % (t1, t2, t2-t1, 
                    str(flow_history.get(t1)[0]), str(es_list)))  

                rcnt = 0
                dcnt = 0
                demand = 0
                for n1, n2 in zip(use_path[0:], use_path[1:]):

                    if (n1, n2) in es_list:
                        demand += (t2-t1)*flow.demand_per_tick
                        dcnt += 1
                        logger.debug("      d %f -> (%d,%d)" % (t2-t1, n1, n2))
                        linkMap[(n1,n2)].update(t1, t2, flow.demand_per_tick, delegated=True)
                        es_list.remove((n1, n2))
                    elif (n1, n2) in es_list_reverse:
                        rcnt += 1
                        logger.debug("      r %f -> (%d,%d)" % (t2-t1, n1, n2))
                        linkMap[(n1,n2)].update(t1, t2, flow.demand_per_tick, reverseDelegated=True) 
                        es_list_reverse.remove((n1, n2))
                    else:
                        logger.debug("      n %f -> (%d,%d)" % (t2-t1, n1, n2))
                        linkMap[(n1,n2)].update(t1, t2, flow.demand_per_tick)

                assert(rcnt == dcnt)
                dtotal += demand
            logger.debug(" => %f" % demand)
            dtotal_all += dtotal

        else:
            # default case for no delegation
            liste = []
            for linkid in path_tuples:
                link = ctx.topo.graph.edges[linkid]['_link']
                liste.append(str(link.id))
                linkMap[link.id].update(flow.start, flow.start+flow.duration, flow.demand_per_tick)

    logger.info(" ==> dtotal_all=%f" % dtotal_all)
    if len(stats_len_usepath) > 0:
        logger.info("      len_usepath_max=%d" % max(stats_len_usepath))
        logger.info("      len_usepath_avg=%f" % (sum(stats_len_usepath)/len(stats_len_usepath)))

    utils = {}
    link_utils=[]
    for link in links:
        d = linkMap[link.id].demand_total
        check_d = sum(linkMap[link.id].demand_over_time)
        m = max(linkMap[link.id].demand_over_time)
        # there might be a lot if links; we will store detailed information only for those
        # links where the observe_utilization flag is set (done inside the topology/scenario
        # specification)
        if link.observe_utilization:
            if d > 0:
                utils[link.label] = d
                #logger.info("link=%s %s total=%f %f max=%d" % (link.label, str(link.id), d, check_d, m))
            link_util = dict(
                id=list(link.id),
                label=link.label,
                demand_over_time=linkMap[link.id].demand_over_time,
                demand_total=d,
                demand_peak=m
            )
            link_utils.append(link_util)

    # link_utils is pretty large, is not included in larger simulations (because of disk space)
    if ctx.config.get("param_include_link_utils_in_statistics") == 1:
        ctx.statistics['metrics.link_utils'] = link_utils
        
    ctx.statistics['metrics.link_utils_aggregated'] = utils

    # calculate delegated demand statistics (based on link data)
    demand_delegated_by_switch = {}
    demand_delegated_es_by_switch = {}
    demand_total = 0 
    demand_delegated = 0 # amount that was delegated in total (including ES part)
    demand_delegated_es = 0 # amount that was delegated to backup switch (ES)
    for link in links:
        total = linkMap[link.id].demand_total
        delegated = linkMap[link.id].demand_delegated
        logger.debug('link_util %s : total=%f | delegated=%f' % (link.label, total, delegated))
        demand_delegated += delegated
        # if the link label has a h in it, it is an edge link such as s1->s1h2
        if 'h' in link.label:
            demand_total += total   
        if delegated > 0:
            label = ctx.topo.get_label_by_id(link.id[0])
            try:
                demand_delegated_by_switch[label] += delegated 
            except KeyError:
                demand_delegated_by_switch[label] = delegated

            # switch name ES has special meaning (backup extension switch)
            if 'ES' in link.label:
                demand_delegated_es += delegated
                try:
                    demand_delegated_es_by_switch[label] += delegated 
                except KeyError:
                    demand_delegated_es_by_switch[label] = delegated

    logger.debug("==> total=%f | delegated=%f | delegated_to_es=%f" % (demand_total, 
        demand_delegated, demand_delegated_es))
    logger.debug(str(demand_delegated_by_switch))

    if demand_total > 0:
        demand_total = demand_total / 2 # required because the link counts all traffic to the edge twice (sending/receiving side)
        for switch, value in demand_delegated_by_switch.items():
            ctx.statistics['metrics.%s.delegated_demand_percent' % switch.lower()] = (value/demand_total)*100
    if demand_delegated > 0:
        for switch, value in demand_delegated_es_by_switch.items():
            ctx.statistics['metrics.%s.delegated_to_es_percent' % switch.lower()] = (value/demand_delegated)*100
   
    # calculate delegated demand for all flows
    # (the old version to calculate this based on flow data and not link data)
    minflow = [] # first flow rule active
    maxflow = [] # last flow rule active
    mindelegate = []
    maxdelegate = []
    demand_delegated = 0
    demand_total = 0
    per_tick = {}
    for flow in ctx.flows:
        minflow.append(flow.start)
        maxflow.append(flow.start+flow.duration)
        demand_total += flow.duration * flow.demand_per_tick
        if len(flow.delegation.reports) > 1:
            for r1, r2 in zip(flow.delegation.reports, flow.delegation.reports[1:]):
                # start and end time of delegation are recorded
                if r1.action == 1 and r2.action == 0:
                    mindelegate.append(r1.tick)
                    maxdelegate.append(r2.tick)
                    demand =  (r2.tick-r1.tick)*flow.demand_per_tick
                    demand_delegated += demand
                    #print("del", r1.tick, r2.tick,demand)
            rlast =  flow.delegation.reports[-1]
            if rlast.action == 1:
                mindelegate.append(rlast.tick)
                maxdelegate.append(flow.finished_at)
                demand =  (flow.finished_at-rlast.tick)*flow.demand_per_tick
                demand_delegated += demand
                #print("del", rlast.tick, flow.finished_at, demand)   
        if len(flow.delegation.reports) == 1:
            r1 = flow.delegation.reports[0]
            mindelegate.append(r1.tick)
            maxdelegate.append(flow.finished_at)
            demand =  (flow.finished_at-r1.tick)*flow.demand_per_tick
            demand_delegated += demand

    if len(minflow) > 0:
        ctx.statistics['metrics.flow_first_active'] = min(minflow)
    if len(maxflow) > 0:
        ctx.statistics['metrics.flow_last_active'] = max(maxflow)
    if len(mindelegate) > 0:
        ctx.statistics['metrics.flow_first_delegate'] = min(mindelegate)
    if len(maxdelegate) > 0:
        ctx.statistics['metrics.flow_last_delegate'] = max(maxdelegate)

    ctx.statistics['metrics.demand_delegated'] = demand_delegated
    ctx.statistics['metrics.demand_total'] = demand_total
    if demand_total > 0:
        ctx.statistics['metrics.demand_delegated_percent'] = (demand_delegated/demand_total)*100

    switches = [opts['_switch'] for _, opts in ctx.topo.graph.nodes(data=True) if opts.get('_switch')]
    switch_names = []
    for switch in switches:
        cnt_active_flows = []
        cnt_active_flows_total = []
        cnt_active_flows_evicted = []
        cnt_ports_delegated = []
        cnt_backdelegations = []
        cnt_total_backdelegations = 0
        cnt_total_adddelegations = 0
        for report in switch.reports:
            #print(report.tick, report.cnt_active_flows)
            cnt_active_flows.append(report.cnt_active_flows)
            cnt_active_flows_total.append(report.cnt_active_flows_total)
            cnt_active_flows_evicted.append(report.cnt_active_flows_evicted)
            cnt_ports_delegated.append(report.cnt_ports_delegated)  
            cnt_backdelegations.append(report.cnt_backdelegations)
            cnt_total_backdelegations += report.cnt_backdelegations
            cnt_total_adddelegations += report.cnt_adddelegations
        m = Metrics()
        m.overhead = Metric(cnt_ports_delegated)
        m.overutil = Metric([x - thresh  for x in cnt_active_flows if x > thresh])
        m.underutil = Metric([thresh - x  for x, e in zip(cnt_active_flows, cnt_active_flows_evicted) if x < thresh and x+e > thresh])
        m.backdelegations = Metric(cnt_backdelegations)

        switch_name = switch.label.lower()
        switch_names.append(switch_name)

        if switch.label == 'ES':
            ctx.statistics['metrics.rss_sum_util_to_es'] = sum(cnt_active_flows)

        ctx.statistics['metrics.%s.flowtable_cnt_total_ctrl_operations' % switch_name] = cnt_total_backdelegations + cnt_total_adddelegations
        ctx.statistics['metrics.%s.flowtable_cnt_active_flows' % switch_name] = cnt_active_flows
        ctx.statistics['metrics.%s.flowtable_cnt_active_flows_evicted' % switch_name] = cnt_active_flows_evicted
        ctx.statistics['metrics.%s.flowtable_cnt_ports_delegated' % switch_name] = cnt_ports_delegated
        ctx.statistics['metrics.%s.flowtable_cnt_active_flows_total' % switch_name] = cnt_active_flows_total
        ctx.statistics['metrics.%s.flowtable_cnt_backdelegations' % switch_name] = cnt_backdelegations 
        if len(cnt_active_flows_evicted) > 0 and len(cnt_active_flows) > 0 and sum(cnt_active_flows) > 0:
            ctx.statistics['metrics.%s.flowtable_tr' % switch_name] = float(sum(cnt_active_flows_evicted))/float(sum(cnt_active_flows))
            #avg1 = float(sum(cnt_active_flows_evicted))/float(len(cnt_active_flows_evicted))
            #avg2 = float(sum(cnt_active_flows))/float(len(cnt_active_flows))
            #ctx.statistics['metrics.ds.flowtable_tr_avg'] = avg1/avg2

        # Underutilization
        underutil_max = 0
        underutil_cnt = [1 for x, e in zip(cnt_active_flows, cnt_active_flows_evicted) if x+e > thresh]
        if len(underutil_cnt):
            underutil_max = sum(underutil_cnt)*thresh
        if underutil_max > 0:
            ctx.statistics['metrics.%s.underutil_percent' % switch_name] = (m.underutil.sum/underutil_max)*100
        else:
            ctx.statistics['metrics.%s.underutil_percent' % switch_name] = 0

        # Overutilization
        overutil_max = 0
        overutil_cnt = [x-thresh for x in cnt_active_flows_total if x > thresh]

        if len(overutil_cnt) > 0:
            overutil_max = sum(overutil_cnt)
            ctx.statistics['metrics.%s.overutil_raw' % switch_name] = overutil_max
        if overutil_max > 0:
            ctx.statistics['metrics.%s.overutil_percent' % switch_name] = (m.overutil.sum/overutil_max)*100
        else:
            ctx.statistics['metrics.%s.overutil_percent' % switch_name] = 0

        # Overhead
        if ctx.statistics.get('metrics.flow_first_active') and ctx.statistics.get('metrics.flow_last_active'):
            x1 = math.floor(ctx.statistics['metrics.flow_first_active'])
            x2 = math.floor(ctx.statistics['metrics.flow_last_active'])
            try:
                h = len(ctx.scenario_data.ports)
                overhead_max = (x2-x1)*h
                ctx.statistics['metrics.%s.overhead_max' % switch_name] = overhead_max
                ctx.statistics['metrics.%s.overhead_percent' % switch_name] =  (m.overhead.sum/overhead_max)*100
            except AttributeError:
                pass

        ctx.statistics['metrics.%s.underutil_max' % switch_name] = underutil_max
        ctx.statistics['metrics.%s.overhead' % switch_name] = m.overhead.sum
        ctx.statistics['metrics.%s.overutil' % switch_name] = m.overutil.sum
        ctx.statistics['metrics.%s.underutil' % switch_name] = m.underutil.sum
        ctx.statistics['metrics.%s.backdelegations' % switch_name] = m.backdelegations.sum
        
        if ctx.statistics.get('traffic.cnt_flows') > 0:
            ctx.statistics['metrics.%s.backdelegations_percent' % switch_name] = m.backdelegations.sum/ctx.statistics.get('traffic.cnt_flows')*100

    # the labels of the switches can be useful, so we add them here as well
    # todo: move this to the topo part
    ctx.statistics['topo.switch_names'] = switch_names
    return metrics