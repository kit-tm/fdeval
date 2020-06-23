import os
import logging
import importlib

logger = logging.getLogger(__name__)

def create_topo_custom(ctx, graph, **kwargs):
    """
    Manually create simple topologies by parsing python files where the topologies
    are constructed in a mininet-like fashion
    """

    # make sure that the topology file exists
    filename = kwargs.get("filename")
    if filename.startswith('custom'):
        filepath = os.path.join(os.getcwd(), 'topo', filename)
    else:
        filepath = filename
        filename = filename.replace(os.path.join(os.getcwd(), 'topo'), '')

    if not filepath.endswith(".py"): filepath = filepath + '.py'
    if not os.path.exists(filepath):
        logger.error("Topology file not found: %s" % filepath)
        raise FileNotFoundError(filepath)

    # convert filename to module notation and dynamically load the topology
    # Example: custom/simple/simple1.py --> topo.custom.simple.simple1
    modulepath = "topo.%s" % filename.replace('.py', '').replace(os.sep, '.')
    modulepath = modulepath.replace('..', '.')
    logger.info("loading custom topology: %s" % modulepath)
    topo = importlib.import_module(modulepath).get_topo(ctx)

    # add created topology to graph
    nindex = 0
    indexByName = {}
    for node in topo.g.nodes(data=True):
        logger.debug('add node', node)
        name = node[0]
        opts = node[1]
        if not 'label' in opts:
            opts.update({'label' : name})
        graph.add_node( nindex, **opts )
        indexByName[name]= nindex
        nindex += 1

    for edge in topo.g.edges(data=True):
        logger.debug('add edge %s' % str(edge))
        n1 = edge[0]
        n2 = edge[1]
        opts = edge[2]
        graph.add_edge( indexByName[n1], indexByName[n2], **opts )
        graph.add_edge( indexByName[n2], indexByName[n1], **opts )
        #graph.add_edges_from( topo.g.edges( data=True, keys=True ) )

    # add the traffic specification to ctx
    ctx.traffic = topo.traffic