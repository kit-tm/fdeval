from pprint import pprint
import json
import logging
import math

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

def calculate_standard_metrics(ctx):
    metrics = {}
    thresh = int(ctx.config.get("param_topo_switch_capacity"))
    switches = [opts['_switch'] for _, opts in ctx.topo.graph.nodes(data=True) if opts.get('_switch')]
    for switch in switches:
        cnt_active_flows = []
        cnt_active_flows_total = []
        cnt_active_flows_evicted = []
        cnt_ports_delegated = []
        for report in switch.reports:
            #print(report.tick, report.cnt_active_flows)
            cnt_active_flows.append(report.cnt_active_flows)
            cnt_active_flows_total.append(report.cnt_active_flows_total)
            cnt_active_flows_evicted.append(report.cnt_active_flows_evicted)
            cnt_ports_delegated.append(report.cnt_ports_delegated)  
        m = Metrics()
        m.overhead = Metric(cnt_ports_delegated)
        m.overutil = Metric([x - thresh  for x in cnt_active_flows if x > thresh])
        m.underutil = Metric([thresh - x  for x, e in zip(cnt_active_flows, cnt_active_flows_evicted) if x < thresh and x+e > thresh])
        
 

        metrics[switch.label] = m
    return metrics

def create_flow_history(ctx):
    history = {}
    for flow in ctx.flows:
        entry = dict(
            spec=flow.flow_gen,
            history=flow.flow_history
        )
        history[str(flow.id)]=entry
    print(json.dumps(history))
    return json.dumps(history)

def get_flow_timings(ctx):
    result = {}
    for link_id in ctx.topo.graph.edges():
        flows = {}
        for flow in ctx.flows:
            data = filter(lambda e: e.get("link") == link_id, flow.flow_history)  
            data = list(sorted(data, key=lambda t: t.get('tick')))
            if len(data) == 0: continue;
            #assert(data[0].get('event') == 'EVLinkFlowAdd' or data[0].get('event') == 'EVLinkFlowAdded')
            assert(data[-1].get('event') == 'EVLinkUpdateOnFinished' or data[-1].get('event') == 'EVLinkFlowStopped')

            flows[flow.label] = [data[0].get('tick'), data[-1].get('tick')]
        result[ctx.topo.print_link(link_id)] = flows
    return json.dumps(result)


def verify_flow_timings(ctx, timings):
    errors = []
    check = json.loads(timings)
    for link_id in ctx.topo.graph.edges():
        link = ctx.topo.print_link(link_id)
        flows = {}
        for flow in ctx.flows:
            data = filter(lambda e: e.get("link") == link_id, flow.flow_history)  
            data = list(sorted(data, key=lambda t: t.get('tick')))
            if len(data) == 0: continue;
            #assert(data[0].get('event') == 'EVLinkFlowAdd' or data[0].get('event') == 'EVLinkFlowAdded')
            assert(data[-1].get('event') == 'EVLinkUpdateOnFinished' or data[-1].get('event') == 'EVLinkFlowStopped')
            got = [data[0].get('tick'), data[-1].get('tick')]
            expected = check.get(link).get(flow.label)
            if not got == expected:
                diff = abs(got[0]-expected[0]) + abs(got[1]-expected[1])
                errors.append('timings for flow=%s on link=%s invalid; expected: %s; got: %s, DIFF=%.2f' % (
                    flow.label, link, str(expected), str(got), diff ))
    return errors

def print_summary(ctx):
    for link_id in ctx.topo.graph.edges():
        for flow in ctx.flows:
            data = filter(lambda e: e.get("link") == link_id, flow.flow_history)  
            data = sorted(data, key=lambda t: t.get('tick'))
            if len(data) == 0: continue;

            print("---------", flow.label, ctx.topo.print_link(link_id))
            pcc = -1

            cnt = 0
            while len(data) > 0 and cnt < 50:
                use = list(filter(lambda e: e.get("pcc") == pcc, data))
                remaining = list(filter(lambda e: e.get("pcc") != pcc, data))
                data = remaining
                pcc += 1
                cnt += 1

                for ev in use:
                    tick = ev.get("tick")
                    pcc = ev.get("pcc")
                    event = ev.get("event")
                    demand = ev.get("demand")

                    print(tick, pcc, event, demand)


def verify_result(ctx, expected_result, description=""):
    expected = json.loads(expected_result)
    history = {}
    errors = 0
    for flow in ctx.flows:
        entry = dict(
            spec=flow.flow_gen,
            history=flow.flow_history
        )
        history[str(flow.id)]=entry
        expected_flow = expected.get(str(flow.id))
        if not expected_flow:
            print("!!! Flow with id=%s not present in EXPECTED" % str(flow.id))
            errors+=1
        else:
            # check flow spec
            expected_spec = expected_flow.get("spec")
            if not expected_spec == flow.flow_gen:
                # check for differences in the spec (some are not critical, e.g. labels)
                for k, v in flow.flow_gen.items():
                    if k in ['fg_label']: continue;
                    if v != expected_spec[k]:
                        errors+=1
                        print("!!! flowspec error for flow id=%s --> expected %s=%s but got %s=%s" %(
                            str(flow.id), str(k), str(expected_spec[k]), str(k), str(v)))
            else:
                # check history results
                expected_history = expected_flow.get("history")
                if not len(expected_history) == len(flow.flow_history):
                    errors+=1
                    print("!!! expected %d events in history of flow id=%s but got %d" % (
                        len(expected_history), str(flow.id), len(flow.flow_history)))
                else:
                    # arrays have the same size, make sure that all events are the same
                    i = 0
                    for event in flow.flow_history:
                        for k, v in event.items():
                            if k == 'link': v = list(v); # link is converted to array while serializing
                            if v != expected_history[i][k]:
                                errors += 1
                                print("!!! history error for flow id=%s --> expected %s=%s for event=%d but got %s=%s" % (
                                    str(flow.id), str(k), expected_history[i][k], i, str(k), str(v)))
                        i += 1
    if errors == 0:
        logger.info("test %s passed with no errors" % description)
    else:
        logger.error("test %s FAILED, %d errors" % (description, errors))
    return errors