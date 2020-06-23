import networkx as nx
import logging
import matplotlib.pyplot as plt
import numpy as np
import math 

from topo.gml.create_topo import create_topo_gml
from topo.custom.create_topo import create_topo_custom
from topo.custom.helper.global_single_v3 import get_topo as get_topo_single_v3

logger = logging.getLogger(__name__)

class Topology:
    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.ctx.topo = self
        self.filename = kwargs.get('filename') 
        self.graph = nx.DiGraph()
        self.xvalues = [] # plain list with all x values sorted by node id
        self.yvalues = [] # plain list with all y values sorted by node id
        self.labels = [] # plain list with all node labels sorted by node id
        logger.debug("load topology", kwargs)

        # Topologies are usually stored in a file inside the topo directory
        if 'filename' in kwargs:
            filename = kwargs.get('filename')

            # points to a topology stored in gml format
            if filename.startswith("gml/"):
                create_topo_gml(self.ctx, self.graph, **kwargs)
            else:   
                # points to a topology defined in mininet style
                create_topo_custom(self.ctx, self.graph, **kwargs)
        else:
            logger.error("Topology() has no filename attribute in kwargs")
            raise RuntimeError()

        logger.debug("done, nodes in topology: %d" % len(self.graph))
        
        # create a list with plain x and y values (used for plotting)
        for node in self.graph.nodes(data=True):
            self.xvalues.append(node[1]['x'])
            self.yvalues.append(node[1]['y'])
            self.labels.append(node[1]['label'])

    def get_node_by_attr(self, attr_name, attr_value):
        """Returns the node id of a node with a given attribute (such as label==h1)"""
        selected_nodes = [n for n,v in self.graph.nodes(data=True) if v[attr_name] == attr_value]  
        if len(selected_nodes) == 1:
            return selected_nodes[0]
        else:
            logger.warn("used get_node_by_attr() with non-unique attribute (attr_name=%s, attr_value=%s); returned None instead of %s" % (
                attr_name, str(attr_value), selected_nodes))
            return None

    def get_host_by_label(self, label):
        id = self.get_node_by_attr('label', label)
        if id is not None:
            return self.graph.nodes[id]['_host']

    def get_switch_by_id(self, id):
        return self.graph.nodes[id]['_switch']

    def get_label_by_id(self, id):
        try:
            return self.graph.nodes[id]['_switch'].label
        except:
            return self.graph.nodes[id]['_host'].label        

    def get_switch_by_label(self, label):
        id = self.get_node_by_attr('label', label)
        if id is not None:
            #print("get_switch_by_label", label)
            return self.graph.nodes[id]['_switch']

    def path_to_id(self, path):
        """resolve [h1, s1, h2] to [0,1,2]"""
        return [ self.get_node_by_attr('label', x) for x in path]

    def print_link_tuples(self, tuples):
        """resolve (0,1), (1,2) to h1->h2 | h2->h3"""
        return ' | '.join([self.print_link(x) for x in tuples])

    def print_link(self, link_id):
        """resolve (0,1) to h1->h2"""
        r = self.graph.nodes[link_id[0]]['label']    
        l = self.graph.nodes[link_id[1]]['label'] 
        return '%s->%s' % (r, l)

    def print_topo(self):
        for node in self.graph.nodes(data=True):
            print(node)

    def plot(self, highlights=[], edge_coloring=None, title=None):
        """
        highlights : array with node ids that should be highlighted
        """

        for node, opts in self.graph.nodes(data=True):
            plt.text(opts['x'], opts['y'], opts['label'] + ' (%d)' % node)   
      

        if edge_coloring:
            # http://jdherman.github.io/colormap/
            colors = []
            with open("colormap", "r") as file:
                data = file.read()
                for i, row in enumerate(data.split(os.linesep)):
                    r,g,b = row.split()
                    colors.append((float(r)/255, float(g)/255, float(b)/255))

            # normalize edge colors (0..1)
            max_value = np.max(edge_coloring.values())
            for k, v in edge_coloring.iteritems():
                normalized = float(v)/float(max_value)
                #print k, normalized, "-->", int(normalized*170)
                edge_coloring[k] = int(normalized*170)
                #print k, colors[int(normalized*170)]

        cnt = 0
        for i in range(len(self.graph)):
            for j in range(len(self.graph)):   
                if self.graph.has_edge(i, j):
                    plt.plot([self.xvalues[j], self.xvalues[i]], [self.yvalues[j], self.yvalues[i]], 
                        'k-', color='lightgrey', linewidth=6)
                    if edge_coloring:
                        color_index = edge_coloring.get((i, j))
                        if not color_index: color_index = edge_coloring.get((j, i))
                        if not color_index: color_index = 0
                        plt.plot([self.xvalues[j], self.xvalues[i]], [self.yvalues[j], self.yvalues[i]], 
                        'k-', color=colors[color_index], linewidth=3)
                    else:
                        plt.plot([self.xvalues[j], self.xvalues[i]], [self.yvalues[j], self.yvalues[i]], 
                        'k-', linewidth=3)                  
                    cnt +=2

        plt.plot(self.xvalues, self.yvalues, 'ro', markersize=8, c='black')

        #highlights = [((3,), 'blue')]
        for highlight, color in highlights:
            for h in highlight:
                x = self.xvalues[h]
                y = self.yvalues[h]
                c = plt.Circle((x, y), 0.7, color=color, fill=False, zorder=4)   
                plt.gcf().gca().add_artist(c)

        # see https://stackoverflow.com/questions/18873623/matplotlib-and-apect-ratio-of-geographical-data-plots
        # either of these will estimate the "central latitude" of your data
        # 1) do the plot, then average the limits of the y-axis    
        #central_latitude = sum(plt.axes().get_ylim())/2.
        # 2) actually average the latitudes in your data
        central_latitude = np.average(self.yvalues)
        # calculate the aspect ratio that will approximate a 
        # Mercator projection at this central latitude 
        mercator_aspect_ratio = 1/math.cos(math.radians(central_latitude))

        #plt.gcf().set_size_inches(8,8)
        plt.gcf().gca().set_aspect(mercator_aspect_ratio)
        plt.show()


    def plot2(self, ax, highlights=[], edge_coloring=None, title=None):
        """
        highlights : array with node ids that should be highlighted
        """
      
        # print labels
        #for name, xpt, ypt in zip(self.labels, self.xvalues, self.yvalues):
        #    ax.text(xpt, ypt, name)

        # http://jdherman.github.io/colormap/

        # calculate colors
        if edge_coloring:
            colors = []
            with open("colormap", "r") as file:
                data = file.read()
                for i, row in enumerate(data.split(os.linesep)):
                    r,g,b = row.split()
                    colors.append((float(r)/255, float(g)/255, float(b)/255))
            #print colors
            # normalize edge colors (0..1)
            if edge_coloring:
                max_value = np.max(edge_coloring.values())
                for k, v in edge_coloring.iteritems():
                    edge_coloring[k] = 20+int(v*150)

                edge_number = dict()
                cnt = 0
                def sort_by_xvalue(e1):
                    x1, x2 = e1
                    return min(self.xvalues[x1], self.xvalues[x2])

                for k, v in sorted(edge_coloring, key=sort_by_xvalue):
                    cnt+=1
                    edge_number[(k, v)] = cnt
            else:
                edge_coloring = dict()

        # print edges
        cnt = 0
        for i, j in self.graph.edges():
            ax.plot([self.xvalues[j], self.xvalues[i]], [self.yvalues[j], self.yvalues[i]], 
                'k-', color='lightgrey', linewidth=6)

            color_index = edge_coloring.get((i, j))
            if not color_index: color_index = edge_coloring.get((j, i))
            if not color_index: color_index = 0

            edge_mid_x = 0
            edge_mid_y = 0
            if self.xvalues[j] > self.xvalues[i]:
                edge_mid_x = self.xvalues[j] - ((self.xvalues[j]-self.xvalues[i])/2)
            else:
                edge_mid_x = self.xvalues[i] - ((self.xvalues[i]-self.xvalues[j])/2)
            if self.yvalues[j] > self.yvalues[i]:
                edge_mid_y = self.yvalues[j] - ((self.yvalues[j]-self.yvalues[i])/2)
            else:
                edge_mid_y = self.yvalues[i] - ((self.yvalues[i]-self.yvalues[j])/2)                     

            if edge_coloring:
                ax.plot([self.xvalues[j], self.xvalues[i]], [self.yvalues[j], self.yvalues[i]], 
                    'k-', linewidth=3)     
            else:
                ax.plot([self.xvalues[j], self.xvalues[i]], [self.yvalues[j], self.yvalues[i]], 
                    'k-', color=colors[color_index], linewidth=3)
            if edge_coloring:
                ax.text(edge_mid_x, edge_mid_y, edge_number[(i, j)], fontsize=8)
            cnt +=2
                    #print cnt

        ax.plot(self.xvalues, self.yvalues, 'ro', markersize=6, c='lightgrey')

        #highlights = [((3,), 'blue')]
        for highlight, color in highlights:
            for h in highlight:
                x = self.xvalues[h]
                y = self.yvalues[h]
                #c = plt.Circle((x, y), 0.1, color=color, fill=False, zorder=4)   
                c = plt.Circle((x, y), 0.3, color=color, fill=True, zorder=4)   
                ax.add_artist(c)

        # see https://stackoverflow.com/questions/18873623/matplotlib-and-apect-ratio-of-geographical-data-plots
        # either of these will estimate the "central latitude" of your data
        # 1) do the plot, then average the limits of the y-axis    
        #central_latitude = sum(plt.axes().get_ylim())/2.
        # 2) actually average the latitudes in your data
        central_latitude = np.average(self.yvalues)
        # calculate the aspect ratio that will approximate a 
        # Mercator projection at this central latitude 
        mercator_aspect_ratio = 1/math.cos(math.radians(central_latitude))
        #plt.gcf().set_size_inches(8,8)
        ax.set_aspect(mercator_aspect_ratio)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

class TopologyFromDTSResult(Topology):
    """
    Special topology used for verification with simulator (see solve_rsa.py)
    """
    def __init__(self, ctx, switch, **kwargs):
        self.ctx = ctx
        self.ctx.topo = self
        self.filename = 'gobal_single.py'
        self.graph = nx.DiGraph()
        logger.debug("load TopologySingle", kwargs)

        topo = get_topo_single_v3(ctx, switch, **kwargs)
        
        # add created topology to graph
        nindex = 0
        indexByName = {}
        for node in topo.g.nodes(data=True):
            logger.debug('add node', node)
            name = node[0]
            opts = node[1]
            if not 'label' in opts:
                opts.update({'label' : name})
            self.graph.add_node( nindex, **opts )
            indexByName[name]= nindex
            nindex += 1

        for edge in topo.g.edges(data=True):
            logger.debug('add edge %s' % str(edge))
            n1 = edge[0]
            n2 = edge[1]
            opts = edge[2]
            self.graph.add_edge( indexByName[n1], indexByName[n2], **opts )
            self.graph.add_edge( indexByName[n2], indexByName[n1], **opts )
            #graph.add_edges_from( topo.g.edges( data=True, keys=True ) )
        
        # add the traffic specification to ctx
        ctx.traffic = topo.traffic
