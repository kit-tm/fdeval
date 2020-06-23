import abc
from heapq import heappush, heappop, nsmallest

from core.events import *

class Consumable: 
    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.kwargs = kwargs
        self.id = kwargs.get("id")
        self._active = [] # a heap storing all currently active consumers
        self._tick_current = 0;
        self.on_simulation_finished = kwargs.get("on_simulation_finished", None); # called once when simulation is finished
        self.on_last_event = kwargs.get("on_last_event", None); # callback option that is executed if there are no further events
        self.last_event_executed = False;
        self.reports = [];

    def verbose(self, *msg):
        if self.ctx.verbose:
            print(*msg)

    def push_report(self, event):
        """Add report events"""
        self.reports.append(event)

    def push_event(self, event):
        """Add new event to the _active heap"""
        return self.ctx.sim.push_event(event, self);

    def register_stats(self, tick):
        """Create a new statistics event for this consumer"""
        ev = EVStats(tick)
        self.ctx.sim.push_event(ev, self)
    
    @abc.abstractmethod
    def on_event(self, event):
        pass


