import os
import logging
import geopy.distance

from .gml_parser import deserialize

logger = logging.getLogger(__name__)

def create_topo_gml(ctx, graph, **kwargs):
    """Reads content from a .gml file and add the topology information into
    the provided graph object"""
    filename = kwargs.get("filename")
    if not filename:
        raise AttributeError('filename attribute is missing')
    topo = dict()
    nodenames = []
    coordinates = []
    #graph = nx.read_gml(filename)
    nindex = 0
    names = []
    xvalues = []
    yvalues = []   
    name_to_index = {}   

    # gml files must be stored in the gml folder that is located in this directory.
    # filename must point to a file from this folder.
    if not filename.lower().endswith('.gml'):
        filename += '.gml'
    filepath = os.path.join(os.getcwd(), 'topo', filename)

    if not os.path.exists(filepath):
        logger.error('The requested topology gml file was not found: %s' % filepath)
        raise FileNotFoundError(filepath)
    else:
        logger.debug('read from local gml file: %s' % filepath)

    
    with open(filepath, "r") as file:
        data = file.read()
        nodes, edges = deserialize(data)
        
        nindex = 0
        map_indices = {}
        is_logical = {}
        for v_id, v_name, v_lat, v_long, v_country in nodes:
            #print v_id, v_name, v_lat, v_long
            # logical nodes don't have geo coordinates
            if v_long is None or v_lat is None:
                is_logical[v_id] = True
            else:
                is_logical[v_id] = False
                label = v_name + "("+str(nindex)+")"
                xvalues.append(v_long)
                yvalues.append(v_lat)
                names.append(label)
                name_to_index[v_name] = nindex
                map_indices[v_id] = nindex  
                graph.add_node(nindex, label=label, x=v_long, y=v_lat)
                nindex += 1

        for id, name, src, target in edges:
            if is_logical[src] is True or is_logical[target] is True:
                #print "skip edge", id, name, src, target
                pass
            else:
                s = map_indices[src]
                t = map_indices[target]
                assert(xvalues[s] != None)
                assert(yvalues[s] != None)
                assert(xvalues[t] != None)
                assert(yvalues[t] != None)
                dist = geopy.distance.vincenty((yvalues[s], xvalues[s]), (yvalues[t], xvalues[t])).meters
                graph.add_edge(s, t, weight=dist)
                graph.add_edge(t, s, weight=dist)
    """
    for name, node in graph.nodes(data=True):
        y =  node.get("Latitude")
        x =  node.get("Longitude")
        label = name + "("+str(nindex)+")"
        xvalues.append(x)
        yvalues.append(y)
        names.append(label)
        name_to_index[name] = nindex
        graph.add_node(nindex, label=label, x=x, y=y)
        nindex += 1

    for src, target, edge_data in graph.edges(data=True):
        src = name_to_index[src]
        target = name_to_index[target]
        dist = geopy.distance.vincenty((yvalues[src], xvalues[src]), (yvalues[target], xvalues[target])).meters
        graph.add_edge(src, target, weight=dist)
        graph.add_edge(target, src, weight=dist)
    """

    #self.nodecnt = len(names)
    #self.size = self.nodecnt
    #self.names = names
    #self.xvalues = xvalues
    #self.yvalues = yvalues
    #self.topo = topo
    #self.graph = G
