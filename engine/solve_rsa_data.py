import logging
import math
import heapq
import statistics

from pprint import pprint
logger = logging.getLogger(__name__)

class RSAData:
    """
    Stores RSA analytics based on raw input scenario
    """
    def __init__(self, ctx, dts_data):
        self.ctx = ctx
        self.dts_data = dts_data

        # data without flow delegation (raw scenario)
        self.link_utilization = {} # <node, node, tick> -> utilization
        self.link_utilization_by_tick = {} # <tick> -> [util_p1, util_p2, ...] in Mbit/s

        # and the same with flow delegation
        self.with_fd_link_utilization_delegated = {} # <delegation_switch, remote_switch, tick> -> delegated demand in bits
        self.with_fd_link_utilization_delegated_by_tick = {} # <tick> -> delegated demand in Mbit/s
        self.with_fd_link_utilization_total = {}

    def update_statistics_with_fd(self):
        yAvg = []
        yMax = []
        for t, data in sorted(self.with_fd_link_utilization_delegated_by_tick.items()):
            yAvg.append(statistics.mean(data))
            yMax.append(max(data))
        try:
            link_util_avg = statistics.mean(yAvg)
        except statistics.StatisticsError:
            link_util_avg = 0
        try:  
            link_util_max = statistics.mean(yMax)
        except statistics.StatisticsError:
            link_util_max = 0          

        self.ctx.statistics['rsa.link_util_delegated_mbit_avg'] = link_util_avg
        self.ctx.statistics['rsa.link_util_delegated_mbit_max'] = link_util_max

    def update_statistics(self):

        # table utilization statistics
        yMax = []
        yAvg = []
        max_all_ports = []
        for switch, dts in self.dts_data.items():
            utils = dts.utils_raw.values()
            if len(utils) > 0:
                yMax.append(max(utils))
                yAvg.append(statistics.mean(utils))

            max_per_port = []
            for port, vals in dts.utils_raw_per_port.items():
                if len(vals) > 0:
                    max_per_port.append(max(vals))
            if len(max_per_port) > 0:
                max_all_ports.append(max(max_per_port))



        self.ctx.statistics['scenario.table_util_max_per_switch'] = yMax
        self.ctx.statistics['scenario.table_util_avg_per_switch'] = yAvg
        try:
            self.ctx.statistics['scenario.table_util_max_total_per_port'] = max(max_all_ports)
        except:
            self.ctx.statistics['scenario.table_util_max_total_per_port'] = 0 
        try:
            self.ctx.statistics['scenario.table_util_max_total'] = max(yMax)
        except:
            self.ctx.statistics['scenario.table_util_max_total'] = 0
        try:  
            self.ctx.statistics['scenario.table_util_avg_total'] = statistics.mean(yAvg)
        except:
            self.ctx.statistics['scenario.table_util_avg_total'] = 0

        # link utilization statistics
        yAvg = []
        yMax = []
        yOverflow = []
        link_over_capacity_cnt = 0
        for t, data in sorted(self.link_utilization_by_tick.items()):
            if len(data) > 0:
                yAvg.append(statistics.mean(data))
                yMax.append(max(data))
                over = [x-1000 for x in data if x >= 1000]
                if len(over) > 0:
                    link_over_capacity_cnt += 1
                    yOverflow.append(sum(over))
                else:
                    yOverflow.append(0)

        try:
            link_util_avg = statistics.mean(yAvg)
        except:
            link_util_avg = 0
        try:   
            link_util_max = statistics.mean(yMax)
        except:
            link_util_max = 0
        try:
            link_over_capacity_sum = statistics.mean(yOverflow)
        except:
            link_over_capacity_sum = 0

        self.ctx.statistics['scenario.link.util_mbit_avg'] = link_util_avg
        self.ctx.statistics['scenario.link.util_mbit_max'] = link_util_max
        self.ctx.statistics['scenario.link.over_capacity_sum'] = link_over_capacity_sum
        self.ctx.statistics['scenario.link.over_capacity_cnt'] = link_over_capacity_cnt

    def calculate_with_fd(self):
        map_delegated = {}
        delegated_total = 0
        delegated_shared = 0
        delegated_backup = 0


        for switch, data in self.dts_data.items():
            sum_out = 0
            check_out = {}
            # with_fd_delegated_demand_out_per_switch: <remote_switch, tick> -> demand
            for assigned_remote_switch, tickdata in data.with_fd_delegated_demand_out_per_switch.items():

                datax = []
                datay = []          
                for t, v in tickdata.items():
                    if not check_out.get(t): check_out[t] = 0
                    check_out[t] += v
                    if not map_delegated.get(switch): map_delegated[switch] = {}
                    if not map_delegated.get(switch).get(assigned_remote_switch): map_delegated[switch][assigned_remote_switch] = {}
                    map_delegated[switch][assigned_remote_switch][t] = v
                    delegated_total += v
                    datax.append(t)
                    datay.append(round(v / 1000000,2))
                    if assigned_remote_switch == 9999:
                        delegated_backup += v
                    else:
                        delegated_shared += v

                # TODO: seems to be wrong!
                #self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datax' % (switch, assigned_remote_switch)] = datax
                #self.ctx.statistics['rsa.link.%d.%d.delegated_mbit.datay' % (switch, assigned_remote_switch)] = datay
            # check flow conservation
            for t, v in check_out.items():
                check = data.with_fd_delegated_demand_in_per_tick.get(t)
                assert(math.isclose(v, check))


        # a last check to verify that delegated_total maps with dts results
        check_total = 0
        for switch, dts in self.dts_data.items():
            for t, v in dts.with_fd_delegated_demand_in_per_tick.items():
                check_total += v
        assert(math.isclose(delegated_total, check_total))


        self.ctx.statistics['rsa.link.util_delegated_bits_total'] = delegated_total
        self.ctx.statistics['rsa.link.util_delegated_bits_shared'] = delegated_shared
        self.ctx.statistics['rsa.link.util_delegated_bits_backup'] = delegated_backup

        #pprint(map_delegated)
        self.with_fd_link_utilization_delegated = map_delegated

        # also provide data by tick in Mbit/s
        self.with_fd_link_utilization_delegated_by_tick = {}
        for source, data in self.with_fd_link_utilization_delegated.items():
            for target, timedata in data.items():   
                for t, v in sorted(timedata.items()):
                    try:
                        self.with_fd_link_utilization_delegated_by_tick[int(t)].append(v/1000000.0)
                    except KeyError:
                        self.with_fd_link_utilization_delegated_by_tick[int(t)] = [v/1000000.0] 


    def calculate_raw(self):
        util_map = {}
        for switch, data in self.dts_data.items():
            source = 'dummy_switch_%d' % switch
            sum_out = 0
            check_out = {}
            for target, by_port in data.demand_out_per_port.items():
                for t, v in by_port.items():
                    if not check_out.get(t): check_out[t] = 0
                    check_out[t] += v
                    if 'dummy_switch_' in target:
                        switch2 = int(target.split('_')[2]) # use node id
                        if not util_map.get(switch): util_map[switch] = {}
                        if not util_map.get(switch).get(switch2): util_map[switch][switch2] = {}
                        #util_map[switch][switch2][t] = v
                
            check_in = {}
            for target, by_port in data.demand_in_per_port.items():
                for t, v in by_port.items():
                    if not check_in.get(t): check_in[t] = 0
                    check_in[t] += v
                    if 'dummy_switch_' in target:
                        switch2 = int(target.split('_')[2]) # use node id
                        if not util_map.get(switch2): util_map[switch2] = {}
                        if not util_map.get(switch2).get(switch): util_map[switch2][switch] = {}
                        util_map[switch2][switch][t] = v       

            # check flow conservation
            for t, v in check_out.items():
                assert(math.isclose(v, check_in[t]))



        for s1, data in util_map.items():
            for s2, data2 in data.items():
                datax = []
                datay = []
                for t, v in data2.items():
                    datax.append(t)
                    datay.append(round(v / 1000000,2))
                if not self.ctx.config.get('param_debug_small_statistics') == 1:
                    self.ctx.statistics['rsa.link.%d.%d.raw_mbit.datax' % (s1, s2)] = datax
                    self.ctx.statistics['rsa.link.%d.%d.raw_mbit.datay' % (s1, s2)] = datay


        #pprint(util_map)
        self.link_utilization = util_map

        # also provide data by tick in Mbit/s
        self.link_utilization_by_tick = {}
        for source, data in self.link_utilization.items():
            for target, timedata in data.items():   
                for t, v in sorted(timedata.items()):
                    try:
                        self.link_utilization_by_tick[int(t)].append(v/1000000.0)
                    except KeyError:
                        self.link_utilization_by_tick[int(t)] = [v/1000000.0] 



        return self.link_utilization


