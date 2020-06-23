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
import sys
import asyncio
import time
import gzip
import io
import glob
import gc

from pathlib import Path
from functools import partial
from typing import Sequence, Any
from asyncio import async as ensure_future  # async is deprecated in Python 3.4

logger = logging.getLogger(__name__)
logging.getLogger('alembic').setLevel(logging.CRITICAL) # avoid alembic infos

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

# see https://gist.github.com/Garrett-R/dc6f08fc1eab63f94d2cbb89cb61c33d
def gzip_str(string_):
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode='w') as fo:
        fo.write(string_.encode())
    bytes_obj = out.getvalue()
    return bytes_obj

def gunzip_bytes_obj(bytes_obj):
    in_ = io.BytesIO()
    in_.write(bytes_obj)
    in_.seek(0)
    with gzip.GzipFile(fileobj=in_, mode='rb') as fo:
        gunzipped_bytes_obj = fo.read()
    return gunzipped_bytes_obj.decode()

def cmd_read(filename, db_statistics, global_state):
    """
    Read data from pickled file
    """
    items_for_db = []
    converted_all = []
    converted_json = [] # list entries
    converted_djson = [] # dict entries

    DELETE = [
        'solver.stats_itercount',
        'solver.stats_linear_constraint_matrix',
        'solver.stats_nodecount',
        'solver.stats_nz_variables',
        'solver.stats_timelimit'
        'solver.class',
    ]
    with open(filename,'rb') as file:
        all_items = pickle.load(file)
        file.close()
        logger.info('read %s containing %d items' % (os.path.basename(filename), len(all_items)))
        cnt = 0
        for item in all_items:
            cnt +=1
            if cnt % 100 == 0:
                logger.info("%d/%d" % (cnt, len(all_items)))
            json_string = gunzip_bytes_obj(item)
            obj = json.loads(json_string)

            # ---------------
            # cleanup2
            # perform additional cleanup for the scalability experiments
            # uncomment this if required, is not done automatically
            # ---------------
            if 0:
                delete = []
                for c in obj.keys():
                    hasnumber = any(char.isdigit() for char in c) 
                    if hasnumber:     
                        delete.append(c) 
                for c in delete:
                    del obj[c]

            # ---------------
            # cleanup1 the file (there are too many columns otherwise)
            # this will remove columns from the result file that are rarely used
            # (if at all)
            if 1:
                delete = []
                switches = obj.get('scenario.switch_cnt')
                if switches:
                    for switch in range(0, switches):
                        # the scenario generator parameters are identical
                        # for each switch and can be removed (are still present at
                        # the scenario.*** namespace)
                        key = "dts.%d.scenario." % switch
                        for c in obj.keys():
                            if c.startswith(key):
                                delete.append(c)
                            for k in DELETE:
                                if c.startswith(k):
                                    delete.append(c)   
                for c in delete:
                    del obj[c]
            # cleanup1 end
            # ---------------

            plain_item = dict(result_id=obj.get('id')) # cross reference to result table
            for k, v in obj.items():
                if isinstance(v, list):
                    # lists are saved in json format in the db
                    plain_item['json_%s' % k.replace('.', '_')] = json.dumps(v)
                    if not k in converted_json: converted_json.append(k)
                elif  isinstance(v, dict) :       
                    plain_item['json_%s' % k.replace('.', '_')] = json.dumps(v)
                    if not k in converted_djson: converted_djson.append(k)
                else:
                    plain_item[k.replace('.', '_')] = v
                    if not k in converted_all: converted_all.append(k)   
            items_for_db.append(plain_item)

        del all_items
        gc.collect(2)

    logger.debug(' ** parameters total: %d' % (len(converted_all)))
    logger.debug(' ** converted lists:  %d' % (len(converted_json)))
    logger.debug(' ** converted dicts:  %d' % (len(converted_djson)))
    logger.debug(' ** add data to database..')

    #dbname = filename.replace('.pickle', '.db')
    #if os.path.exists(dbname):
    #    os.remove(dbname)
    
    with open_table(db_statistics, "statistics") as table:  
        table.insert_many(items_for_db, chunk_size=100)          
    processed = len(items_for_db)
    
    items_for_db = []
    del items_for_db
    gc.collect(2)
    logger.debug(' ** added %d entries to db.statistics' % processed)
    return filename, processed


def cmd_write(entry, scripts, global_state):
    """
    Write data from file
    """
    statisticsfile = os.path.join(entry, 'statistics.json')
    if not os.path.exists(statisticsfile):
        global_state.get('errors').append('statisticsfile not found: %s; skipping entry' % statisticsfile)
        return None, None

    configfile = os.path.join(entry, 'config.json')
    if not os.path.exists(configfile):
        global_state.get('errors').append('configfile not found: %s; skipping entry' % configfile)
        return None, None

    total = 0
    compressed = 0
    config = None
    with open(configfile, 'r') as file:
        config = json.loads(file.read())

    with open(statisticsfile, 'r') as file:
        data = file.read()
        try:
            data = json.loads(data)
        except json.decoder.JSONDecodeError as e:
            logger.error("parsing %s failed: %s" % (statisticsfile, str(e)))
            global_state.get('errors').append('skipped %s: %s' % (statisticsfile, str(e)))
            return None, None   

        data.update(config)
        del data['id'] # will mess up database
        combined = json.dumps(data)
        l1 = len(combined)
        _string = gzip_str(combined)
        l2 = len(_string)
        compressed += l2    
        total += l1
        return _string, (total, compressed)

@asyncio.coroutine
def write_pickled_data(command_list: Sequence[dict], semaphore, global_state):
    """
    Write results in compressed form on disk (for storage/exchange)
    """
    start = time.time()
    loop = asyncio.get_event_loop()
    fs = [run_command_on_loop(loop, cmd_write, command, semaphore, global_state) for command in command_list]
    cnt = 0
    all_items = []
    total = 0
    compressed = 0
    cutoff = 0 # already wrote to disk
    last_cutoff_index = 0
    for f in asyncio.as_completed(fs):
        result, stats = yield from f
        if result:
            cnt += 1
            logger.info("%d/%d  compression=%.2fMb->%.2fMb" % (cnt, len(command_list), 
                total/1000000.0, compressed/1000000.0))
            all_items.append(result)
            if compressed/1000000.0 - cutoff > global_state.get('maxsize'):
                ensure_future(write_to_disk(all_items, (last_cutoff_index+1, cnt, global_state.get('path_series'))))  
                last_cutoff_index = cnt
                cutoff = compressed/1000000.0 
                all_items = []
            t, c = stats
            total += t
            compressed += c

    if len(all_items) > 0:
        ensure_future(write_to_disk(all_items, 
            (last_cutoff_index+1, cnt, global_state.get('path_series'))))  

    logger.info("done after: %f" % (time.time() - start))


@asyncio.coroutine
def write_to_disk(all_items, stats):
    """
    Write pickle file to disk
    """
    if all_items:
        ix1, ix2, path = stats
        filename = os.path.join(path, '%d-%d.pickle' % (ix1, ix2))
        sys.setrecursionlimit(100000)
        pickle.dump(all_items, open(filename, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("             => wrote file to disk (%d-%d)" % (ix1, ix2))

@asyncio.coroutine
def read_pickled_data(command_list: Sequence[dict], semaphore, global_state):
    """
    Run all commands in a list
    """
    start = time.time()
    loop = asyncio.get_event_loop()
    fs = [run_command_on_loop(loop, cmd_read, command, semaphore, global_state) for command in command_list]
    cnt = 0
    for f in asyncio.as_completed(fs):
        filename, items = yield from f
        print('%d/%d' % (cnt, len(fs)), filename, items)
        cnt += 1
        #ensure_future(process_result(result, compressed))
    logger.info("done after: %f" % (time.time() - start))

@asyncio.coroutine
def run_command_on_loop(loop: asyncio.AbstractEventLoop, fn, command, semaphore, global_state):
    """
    Run command "fn" in loop
    """
    with (yield from semaphore):
        runner = partial(fn, command, semaphore, global_state)
        output = yield from loop.run_in_executor(None, runner)
        #yield from asyncio.sleep(0.01)  # Slowing a bit for demonstration purposes
        return output


class AggregatePreprocessor:

    def __init__(self, read=None, write=None, threads=4, maxsize=500, **kwargs):

        path_series = None
        if read:
            path_series = read
        if write:
            path_series = write
        if not path_series:
            raise RuntimeError('AggregatePreprocessor: no file specified! (-r/-w)')


        # make sure the folder with the db exists
        if not os.path.isabs(path_series):
            path_series = os.path.join(os.getcwd(), path_series)    
        if not os.path.exists(path_series):
            raise RuntimeError("AggregatePreprocessor: file=%s not found" % path_series)
        if path_series.endswith('series.db'):
            path_series = path_series.replace('series.db', '')

        self.threads = threads
        self.maxsize = maxsize
        self.path_series = path_series # points to a folder with a series.db file
        self.db = os.path.join(path_series, 'series.db') # database for experiment parameters
        self.db_statistics = os.path.join(path_series, 'statistics.db') # database for statistics

        if not os.path.exists(self.db):
            logger.warn("AggregatePreprocessor: did not find a series.db file in %s" % path_series)

        if not self.threads:
            self.threads = 4

        if not self.maxsize:
            self.maxsize = 500
        """
        # check metadata
        with open_table(self.db, "metadata") as table:             
            entries = list(table.find())
            if len(entries) == 1:
                # seems ok
                meta = entries[0]
                print("Name:", meta.get('name'))
                print("Folder:", meta.get('folder'))
                print("DB:", meta.get('db'))

        # check info
        with open_table(self.db, "info") as table:             
            entries = list(table.find())
            for entry in entries:
                for k, v in dict(entry).items():
                    print(k, v)
        # check results
        with open_table(self.db, "results") as table:             
            entries = list(table.find())
            print('found %d entries in db.results' % len(entries))
        

        # check results
        with open_table(self.db, "statistics") as table:             
            entries = list(table.find())
            print('found %d entries in db.statistics' % len(entries))
        """
           
    def read_pickled_data(self):
        """
        if os.path.exists(self.db_statistics):
            os.remove(self.db_statistics)
            logger.info("deleted old statistics.db")
        if not os.path.exists(self.db_statistics):
        """

        # create new statistics.db if necessary
        try:
            db = dataset.connect('sqlite:///%s' % self.db_statistics)
            logger.info("created new statistics.db")
        except:
            logger.error("error while creating empty statistics.db")
            exit(1)
        
        # now run through all the pickle files in the folder where series.db is stored
        all_items = []
        for name in glob.glob('%s/*.pickle' % self.path_series):
            all_items.append(name)

        if len(all_items) > 0:
            print(len(all_items), self.threads)
            logger.info("run preprocessor -r with %d files using up to %d processors" % (
                len(all_items), self.threads))
        else:
            logger.info("no .pickle files found in folder: %s; abort" % self.path_series)
            return

        # sort items by name
        all_items = sorted(all_items)
        # there is a memory leak with asyncio here, no idea; using simple process instead
        from multiprocessing import Process, Queue
        cnt = 1
        for file in all_items:
            logger.info("=== read file %d/%d (current=%s)" % (cnt, len(all_items), file))
            p = Process(target=cmd_read, args=(file, self.db_statistics, 1))
            p.start()
            p.join()
            print("exitcode", p.exitcode)
            if p.exitcode == 1:
                exit(1)
            cnt += 1

        return


        # limit number of processes
        semaphore = asyncio.Semaphore(self.threads)

        if len(all_items) > 0:
            global_state = dict(
                path_series=self.path_series,
                db=self.db,
                db_statistics=self.db_statistics,
                started=time.time(),
                todo=len(all_items),
                errors=[],
                data=[],
                done=0,
                timings=[]  
            )
            loop = asyncio.get_event_loop()
            loop.run_until_complete(read_pickled_data(all_items, semaphore, global_state))

    def write_pickled_data(self):
        logger.info("run preprocessor -w")

        all_items = []
        for filename in Path(self.path_series).glob('**/config.json'):
            all_items.append(os.path.dirname(filename))
        logger.info("found %d config.json files on disk" % len(all_items))

        # important to use get_entries() here because we want to only
        # include statistics for values that are consistent with the current
        # parameterization
        #all_items = self.get_entries()
        #all_items = sorted(all_items, key=lambda x: x.get('uuid'))[0:50]

        # limit number of processes
        semaphore = asyncio.Semaphore(self.threads)

        if len(all_items) > 0:
            global_state = dict(
                path_series=self.path_series,
                maxsize=self.maxsize,
                started=time.time(),
                todo=len(all_items),
                errors=[],
                data=[],
                done=0,
                timings=[]  
            )
            loop = asyncio.get_event_loop()
            loop.run_until_complete(write_pickled_data(all_items, semaphore, global_state))

        return
        raise RuntimeError()

        start = time.time()
        errors = []
        cnt = 0
        for entry in entries:
            cnt += 1
            if cnt % 1000 == 0:
               logger.info("processing.. (%d/%d)" % (cnt, len(entries))) 

            #print("update", entry.get("path"))

            jsonfile = os.path.join(entry.get('path'), 'statistics.json')
            if not os.path.exists(jsonfile):
                errors.append('jsonfile not found: %s; skipping entry' % jsonfile)
                continue

            converted_all = []
            converted_json = [] # list entries
            converted_djson = [] # dict entries
            with open(jsonfile, 'r') as file:
                stats = json.loads(file.read())
                plain_item = dict(result_id=entry.get('id')) # cross reference to result table
                #print(stats)
                for k, v in stats.items():
                    if isinstance(v, list):
                        # lists are saved in json format in the db
                        plain_item['json_%s' % k.replace('.', '_')] = json.dumps(v)
                        if not k in converted_json: converted_json.append(k)
                    elif  isinstance(v, dict) :       
                        plain_item['json_%s' % k.replace('.', '_')] = json.dumps(v)
                        if not k in converted_djson: converted_djson.append(k)
                    else:
                        plain_item[k.replace('.', '_')] = v
                        if not k in converted_all: converted_all.append(k)

                all_items.append(plain_item)

                assert(plain_item.get('id') is None)

        logger.info("done after: %f" % (time.time() - start))
        logger.info(' ** parameters total: %d' % (len(converted_all)))
        logger.info(' ** converted lists:  %d' % (len(converted_json)))
        logger.info(' ** converted dicts:  %d' % (len(converted_djson)))

        if len(errors) > 0:
            logger.error("errors occured while preprocessing: %d" % len(errors))
            if len(errors) > 10:
                logger.error("> 10 errors, SKIPPED further processing!")
                return
            else:
                for e in errors:
                    logger.error('  ->%s' % e)

        logger.info('create json backup file of preprocessor data..')
        backup = os.path.join(self.path_series, 'backup_preprocessor.json')
        with open(backup, 'w+') as file:
            file.write(json.dumps(all_items))

        logger.info('add data to database..')
        with open_table(self.db, "statistics") as table:   
            table.delete()
            table.insert_many(all_items, chunk_size=500)          
        logger.info('added %d entries to db->statistics' % len(all_items))

    def get_entries(self):
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
                    if v == '': continue;
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
        

        return filtered_results