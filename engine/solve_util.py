import logging

logger = logging.getLogger(__name__)

class VDSSResult:
    """
    Stores DTS results from a scenario with a single switch
    the update functions are called by the DTS algorithms
    (the name "DSS" is an older name of DTS - delegation status selection and means the same)
    """
    def __init__(self, ports, maxtick=450, init=False):
        self.maxtick = maxtick
        self.ports = ports # keys are port ids, values are label of the connected device behind the port such as s1h4 or s2 or dummy_switch_s7
        self.delegation_status = {}
        self.utils = {} # utilization with delegation (no delegation if equal to utils_raw)
        self.utils_raw = {} # full utilization
        self.utils_shared = {} # used by other switch

        # initialize delegation_status
        if init:
            for t in range(0, maxtick):
                self.delegation_status[t] = {}
                self.utils_shared[t] = 0
                for p, label in ports.items():
                    self.delegation_status[t][p] = VDelegationStatus(t, p, 0, 0, 0, 0, 0)
                    self.delegation_status[t][p].port_label = label

    def print_raw_util(self):
        print("")
        print("raw_util")
        for t, v in self.utils_raw.items(): 
            if v > 0:
                logger.info("   -> %.3d  %d" % (t,v))

    def get_metric_underutilization(self, ctx):
        # underutil depends on the capacity
        thresh = ctx.config.get('param_topo_switch_capacity')
        overhead = []
        util = []
        util_raw = []
        for t, v in self.utils.items():
            overhead_now = 0
            for p in self.ports.keys():
                dstat = self.delegation_status[t][p]
                overhead_now += dstat.status
            overhead.append(overhead_now)
            util.append(v)
        for t, v in self.utils_raw.items():
            util_raw.append(v)
        underutil = [thresh - x  for x, raw in zip(util, util_raw) if x <= thresh and raw >= thresh]
        underutil_total = sum(underutil)
        underutil_percent = -1
        underutil_max = 0
        underutil_cnt = [1 for x in util_raw if x > thresh]
        if len(underutil_cnt):
            underutil_max = sum(underutil_cnt)*thresh
        if underutil_max > 0:
            underutil_percent= (sum(underutil)/underutil_max)*100
        else:
            underutil_percent = 0
        print("  Underutilization")
        print("    underutil_total =", underutil_total)
        print("    underutil_max =", underutil_max)
        print("    underutil_percent =", underutil_percent)
        print("    overhead", sum(overhead))
        return underutil_percent, underutil_total, underutil_max

    def plot_utilization(self):
        import matplotlib.pyplot as plt
        plt.close()
        datax = []
        datay = []
        datax_raw = []
        datay_raw = []
        for t, util in self.utils.items():
            datax.append(t)
            datay.append(util)
        for t, util in self.utils_raw.items():
            datax_raw.append(t)
            datay_raw.append(util)
        plt.plot(datax, datay, label="with flow delegation", color="green")
        plt.plot(datax_raw, datay_raw, label="without flow delegation", color="red")
        plt.legend()
        plt.show()

    def update_util(self, tick, util):
        self.utils[tick] = util

    def update_util_per_port(self, tick, port, util, util_raw):
        self.delegation_status[tick][port].util = util   
        self.delegation_status[tick][port].util_raw = util_raw  

    def update_util_raw(self, tick, util):
        self.utils_raw[tick] = util

    def update_demand_raw_per_port(self, tick, port, demand_raw):
        """Store raw demand associated with port/tick, taken from scenario"""
        self.delegation_status[tick][port].demand_raw = demand_raw 

    def update_demand_per_port(self, tick, port, demand):
        """Store demand associated with port/tick, calculated by DTS algorithm"""
        self.delegation_status[tick][port].demand = demand

    def update(self, tick, port, status):
        """Update DSS status for (t,p)"""
        self.delegation_status[tick][port].status = status

    def get(self, tick, port):
        self.delegation_status[tick][port]

    def get_commands(self, port_mappings):
        """
        Calculates the commands for adding and removing delegation rules.
        The port_mappings dict is required because the sub-problems have a different
        port numbering than the global problem
        """
        commands = {}
        for t in range(1, self.maxtick-1):
            commands[t] = dict(remove=[], add=[], change_es=[])          
        for p, label in self.ports.items():
            for t in range(1, self.maxtick-1):
                dstat1 = self.delegation_status[t][p]
                dstat2 = self.delegation_status[t+1][p]
                if self.delegation_status[t][p].status == 0 and self.delegation_status[t+1][p].status == 1:
                    commands[t+1]['add'].append((port_mappings.get(label), self.delegation_status[t+1][p].es))
                if self.delegation_status[t][p].status == 1 and self.delegation_status[t+1][p].status == 0:
                    commands[t+1]['remove'].append((port_mappings.get(label), self.delegation_status[t][p].es))

                if dstat1.status == 1 and dstat2.status == 1:
                    if dstat1.es != dstat2.es:
                        commands[t+1]['change_es'].append((port_mappings.get(label), dstat2.es))
                        #commands[t+1]['add'].append((port_mappings.get(label), dstat2.es))

        return commands

    # obsolete
    def get_debug_link_util_raw(self, port_mappings):
        util = {}
        for t in range(1, self.maxtick-1):
            util[t] = {}
            for p, label in self.ports.items():
                util[t][port_mappings.get(label)] = self.delegation_status[t][p].demand_raw
        return util

    # obsolete
    def get_debug_link_util(self, port_mappings):
        util = {}
        for t in range(1, self.maxtick-1):
            util[t] = {}
            for p, label in self.ports.items():
                demand = 0
                if self.delegation_status[t][p].status == 1:
                    print("get_debug_link_util", t, p, self.delegation_status[t][p].demand)
                    demand = self.delegation_status[t][p].demand
                util[t][port_mappings.get(label)] = demand
        return util

    def apply_port_mappings(self, switch, port_mappings):
        """
        Return portstatus array
        """
        job_id = switch.id * 1000000 # make sure that job ids are unique 
        for p, label in self.ports.items():
            job_id += 1
            for t in range(1, self.maxtick-1):
                if self.delegation_status[t][p]:
                    self.delegation_status[t][p].mapped_port = port_mappings.get(label)
                    self.delegation_status[t][p].job_id = job_id
                    if self.delegation_status[t][p].status == 1 and self.delegation_status[t+1][p].status == 0:
                        job_id+=1

    def get_events(self, port_mappings):
        """
        Return global delegation events for this switch; each events
        consists of delegations that belong together
        """
        events = {} 
        for p, label in self.ports.items():
            events[port_mappings.get(label)] = []
            current_event = []
            for t in range(1, self.maxtick-1):
                if self.delegation_status[t][p].status == 1:
                    current_event.append(self.delegation_status[t][p])
                if self.delegation_status[t][p].status == 1 and self.delegation_status[t+1][p].status == 0:
                    events[port_mappings.get(label)].append(current_event)
                    current_event = []
        return events

    def print_info(self, port_mappings):
        for p, label in self.ports.items(): 
            print("port=%d" % p, label, port_mappings.get(label))
            for t in range(1, self.maxtick-1):
                if self.delegation_status[t][p].status == 0 and self.delegation_status[t+1][p].status == 1:
                    print("add @", t+1)
                if self.delegation_status[t][p].status == 1 and self.delegation_status[t+1][p].status == 0:
                    print("remove @", t+1)  

    def serialize(self):
        result = []
        for t in range(0, self.maxtick):
            for p, port in self.ports.items(): 
                result.append(self.delegation_status[t][p].serialize())
        return dict(
            maxtick=self.maxtick,
            delegation_status=result,
            ports=self.ports,
            utils=self.utils,
            utils_raw=self.utils_raw
            )

    def get_dss_ports(self):
        used_ports = []
        for p, label in self.ports.items(): 
            for t in range(1, self.maxtick-1):
                if self.delegation_status[t][p].status == 1:
                    if (p, label) not in used_ports:
                        used_ports.append((p, label))
        return used_ports


    @staticmethod
    def deserialize(switch, data):
        ports = {}
        for p, label in data.get('ports').items():
            ports[int(p)] = label
        utils = {}
        for t, util in data.get('utils').items():
            utils[int(t)] = util
        utils_raw = {}
        for t, util in data.get('utils_raw').items():
            utils_raw[int(t)] = util

        maxtick = int(data.get('maxtick'))
        result = VDSSResult(ports, maxtick) 
        result.utils = utils
        result.utils_raw = utils_raw
        for entry in data.get('delegation_status'):
            t = int(entry.get('t'))
            p = int(entry.get('p'))
            # set VDelegationStatus entries with results from data
            if not result.delegation_status.get(t):
                result.delegation_status[t] = {} 
            result.delegation_status[t][p] = VDelegationStatus.deserialize(switch, entry)
        return result

class VDelegationStatus:
    """
    Contains DSS details for time slot t and port p
    """
    
    def __init__(self, tick, port, status, util, util_raw, demand, demand_raw):
        # the following variables come from run_dss
        self.tick = int(tick)
        self.port = int(port) # port id in DSS (different from the id used in RSS!)
        self.port_label = None
        self.status = int(status)
        self.util = int(util)
        self.util_raw = int(util_raw)
        self.demand = float(demand)
        self.demand_raw = float(demand_raw)


        self.switch = None # switch this data belongs to
        self.job_id = None # dynamic jobid (neighboring VDelegationStatus objects will have the same jobid )
        self.es_options = [] # list of potential remote switches
        self.es = None # the remote switch determined by the rsa algorithm
        self.es_switch = None # same as es (deprecated)
        self.mapped_port = None # port id in RSS, set by apply_port_mappings()

    def serialize(self):
        return dict(
            t=self.tick, p=self.port, s=self.status, u=self.util, ur=self.util_raw,
            d=self.demand, dr=self.demand_raw
        )

    @staticmethod
    def deserialize(switch, data):
        obj = VDelegationStatus(data.get('t'), data.get('p'), data.get('s'), data.get('u'), data.get('ur'),
            data.get('d'), data.get('dr'))
        obj.switch = switch
        return obj