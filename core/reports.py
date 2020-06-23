import sys

class Report:
    """Used to store information from the simulation. Different from an Event 
    because Reports cannot be injected as events in the simulation!"""
    def __init__(self, tick=sys.maxsize):
        self.tick = tick;

    def __lt__(self, other):
        return self.tick < other.tick

    def __gt__(self, other):
        return self.tick > other.tick

    def __ge__(self, other):
        return self.tick >= other.tick


class FlowDelegationReport(Report):

    def __init__(self, tick=sys.maxsize, source=-1, target=-1, action=None, current_path=None):
        super().__init__(tick);
        self.source = source;
        self.target = target;
        self.action = action; # 1 = add, 0=remove
        self.removed_at = -1; # tick when delegation was removed
        self.current_path = current_path