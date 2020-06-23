from core.consumable import Consumable
from core.events import *

class Host(Consumable):

    def __init__(self, ctx, link, **kwargs):
        super().__init__(ctx, **kwargs);
        self.id = kwargs.get("id") # networkx node id
        self.link = link; # hosts are connected to a switch via exactly one link (multihomed hosts are currently not supported)
        self.label = kwargs.get("label", "NoLabelSet"); # name in topology
        self.x = kwargs.get("x"); # coordinates in topology
        self.y = kwargs.get("y"); # coordinates in topology


    def on_stats(self, ev):
        pass

    def on_event(self, ev):
        pass