import networkx as nx
import logging
import matplotlib.pyplot as plt
import numpy as np
import math 
import os
import importlib
import dataset
import itertools
import pickle
import json

from .plotutils import ResultSet
from .agg_preprocessor import AggregatePreprocessor

logger = logging.getLogger(__name__)

class open_table():

    def __init__(self, path, table_name):
        self.path = path
        self.table_name = table_name
        return None;

    def __enter__(self):
        self.db = dataset.connect('sqlite:///%s' % self.path)
        return self.db[self.table_name]

    def __exit__(self, type, value, traceback):
        self.db.executable.close()

class AggregatePlotter:

    def __init__(self, path_series, plotter_name, export, **kwargs):

        # make sure the folder with the db exists
        if not os.path.isabs(path_series):
            path_series = os.path.join(os.getcwd(), path_series)    
        if not os.path.exists(path_series):
            raise RuntimeError("AggregatePlotter: file=%s not found" % path_series)
        if path_series.endswith('series.db'):
            path_series = path_series.replace('series.db', '')
        self.export = export
        self.path_series = path_series # points to a folder with a series.db file
        self.plotter_name = plotter_name # name of the plotter to be executed
        self.db = os.path.join(path_series, 'series.db')

        #if not os.path.exists(self.db):
        #    raise RuntimeError("AggregatePlotter: did not find a series.db file in %s" % path_series)

        self.plotter = self.load_plotter()
        self.run_aggregator()

    def run(self):
        pass

    def load_plotter(self):
        # convert filename to module notation and dynamically load the topology
        # Example: custom/simple/simple1.py --> topo.custom.simple.simple1
        modulepath = "plotter.%s" % self.plotter_name.replace('.py', '').replace(os.sep, '.')
        modulepath = modulepath.replace('..', '.')
        logger.info("loading custom plotter: %s" % modulepath)
        return importlib.import_module(modulepath)

    def run_process_ctx(self, result, fsource, ftarget, plotter):
        pass
        logger.info(".. processing: %s"  % fsource)
        ctx = pickle.load(open(fsource, "rb"))
        result_data = plotter.process_ctx(ctx)
        result['result'] = result_data
        with open(ftarget, 'w') as file:
            file.write(json.dumps(result))



    def run_aggregator(self):
        """the main function that does all the aggregation and caching"""

        filtered_results = None
        if hasattr(self.plotter, 'process_ctx'):
            logger.info("plotter has a process_ctx() function, prepare data")
            filtered_results = self.handle_ctx()

        # prepare default result blob with general statistics
        #AggregatePreprocessor(self.db)

        logger.info("prepare ResultSet blob with db=%s" % str(self.db))

        blob = ResultSet(self.db, filtered_results)
        kwargs = dict(
            exportdir='/home/bauer/Repos/diss/part2/figures/eval/',
            param=self.export
        )

        # call plotter
        self.plotter.plot(blob, **kwargs)


    def handle_ctx(self):
        # get all entries stored in db->info
        entries = []
        with open_table(self.db, 'info') as table:
            entries = table.all()

        # extract the intended parameters in db->info and unroll them
        scripts = {} # contains all script entries from series.db->info
        allwork =  [] # an array that stores all work that has to be done
        valid_uuids = []
        for entry in entries:
            id = entry.get('id')
            if not id:
                raise RuntimeError('there should always be an id...')
            # we add a new entry for this dict to the global scripts array
            scripts[id] = dict(entry)

            # we also calculate the unrolled params for this entry of the series
            unrolled_params = {}
            for k, v in entry.items():
                if k.startswith('param_'):
                    # params can be given in the form a-b-c which unrolls
                    # to range(a,b+1,c)
                    if len(v.split('-')) == 3:
                        lower, upper, step = v.split('-')
                        unrolled_params[k] = []
                        for i in range(int(lower), int(upper)+1, int(step)):
                            unrolled_params[k].append((id, k,i))
                        continue;
                    # params can also be given in the form a,b,c which
                    # unrolls to [a,b,c]
                    if len(v.split(',')) >= 2:
                        unrolled_params[k] = []
                        for i in v.split(','):
                            unrolled_params[k].append((id, k, int(i)))
                        continue;
                    # default case (static value)
                    unrolled_params[k] = [(id, k, int(v))]

            combinations = list(itertools.product(*unrolled_params.values()))

            # convert to an array of plain dicts (easier to handle)
            dictarray = []
            
            for c in combinations:
                newdict =  dict()

                # generate a unique experiment name based on all sorted 
                # parameters values
                nameparts = []
                for id, k, v  in sorted(c, key=lambda x: x[1]):
                    nameparts.append(str(v))
                uuid = '.'.join(nameparts)
                newdict['uuid'] = uuid
                if not uuid in valid_uuids: valid_uuids.append(uuid)
               
                for id, k, v in c:
                    newdict[k] = v
                    newdict['id'] = id # reference to the entry in the info table

                dictarray.append(newdict)
            allwork += dictarray
        logger.info('found %d matching parameterizations in db->info' % len(allwork))

        # get all entries from db->results (the actual runs)
        results = []
        with open_table(self.db, "results") as table:
            data = table.all()
            for d in data:
                results.append(dict(d))
        logger.info('found %d entries stored in db->results' % len(results))

        # filter the results (i.e., remove parameters that might be in db->results but not in db->info)
        filtered_results = []
        for result in results:
            if result.get('uuid') in valid_uuids:
                filtered_results.append(result)
        logger.info('found %d results after filtering (ignore all parameters not present in db->info' % len(filtered_results))
        

        # start processing for calling process_ctx()
        skipped_to_error = []
        from_cache = []
        if hasattr(self.plotter, 'process_ctx'):
            logger.info('start preprocessing for process_ctx (this may take a while if no data is cached)')
            # define the main cache folder
            cache = os.path.join(self.path_series, '.cache', self.plotter_name.replace('.py', ''))
            logger.info('cachefolder: %s' % cache)
            try:
                os.makedirs(cache) # create cache
            except:
                pass  

            for result in filtered_results:
                path = result.get('path')

                picklefile = os.path.join(result.get('path'), 'result.pickle')
                if not os.path.exists(picklefile):
                    logger.error('picklefile not found: %s; skipping entry' % picklefile)
                    skipped_to_error.append(picklefile)
                    continue;
                # get filestats to create a unique name for the cached entry
                filestats = os.stat(picklefile)
                cachefile_name = '%s.%f.%d.json' % (result.get('uuid'), filestats.st_mtime, filestats.st_size)
                cachefile_path = os.path.join(cache, cachefile_name)
                if not os.path.exists(cachefile_path):
                    # results are not yet cached, a new call to process_ctx() is required
                    self.run_process_ctx(result, picklefile, cachefile_path, self.plotter)
                    from_cache.append(picklefile)
                else:
                    # read data from cachefile
                    with open(cachefile_path, 'r') as file:
                        result['result'] = json.loads(file.read()).get('result')

        return filtered_results