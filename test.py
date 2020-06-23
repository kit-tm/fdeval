#!/usr/bin/env python
#
# This script can execute multiple experiments in parallel
# controlled by a database called series.db that has to be 
# created beforehand. Use the -i flag to create a new demo 
# series.db file. Customize the init_new_experiment_series()
# function or use external tools to populate the database in
# order to execute real experiments. init_new_experiment_series()
# contains a small dummy example and additional explanations. 

import dataset
import logging
import asyncio
import json
import os
import subprocess
import itertools
import shutil
import argparse
import time, datetime

from functools import partial
from typing import Sequence, Any

from asyncio import async as ensure_future  # async is deprecated in Python 3.4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='parallel sim runner')
parser.add_argument('-f', dest='folder', action='store', type=str, 
    help='Points to a folder with a series.db file in it. Required argument.')

parser.add_argument('-p', dest='plotter', action='store', type=str, 
    help='Defines a plotter to be executed after the simulations.')

parser.add_argument('-t', dest='threads', action='store', type=int, 
    help='Number of parallel processes used for simulation')

parser.add_argument('-n', dest='nopickle', action='store_true', default=False, 
    help='Create no pickle files')

parser.add_argument('-i', dest='init', action='store', type=str, 
    help='Initialize a new empty experiment', default=False)

parser.add_argument('-d', dest='delete_before_init', action='store_true',
    help='Can be used together with -i to delete the existing experiment first (use with caution!)', default=False)


args = parser.parse_args()

# configure asyncio
MAX_RUNNERS = 4
if args.threads: MAX_RUNNERS = args.threads
semaphore = asyncio.Semaphore(MAX_RUNNERS)

# helper for accessing the content in series.db
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

intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )

# from https://stackoverflow.com/questions/4048651/python-function-to-convert-seconds-into-minutes-hours-and-days
def display_time(seconds, granularity=2):
    result = []
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

def init_new_experiment_series(folder):
    """
    Add a new dummy experiment into the specified folder. This will
    create a series.db sqlite file. Use the -d flag to overwrite the old 
    series.db file.

    To execute the created series.db file, use test.py -f /path/to/experiment
    (same path that was used with -i). This will check the content of the 
    series.db file and execute all missing experiments. The results will be
    stored in a new folder in the same directory as the series.db file. In order
    to re-execute all experiments, just delete this folder and re-run test.py -f.
    You can also remove individual results from the result folder and test.py 
    will only re-execute the missing experiments. You can stop and restart the 
    experiments at any time (just hit ctrl+c in the terminal and re-run test.py -f
    to restart).
    """
    db = os.path.join(folder, 'series.db')  
    logger.info("Create a new empty experiment: %s" % db)
    if args.delete_before_init:
        if os.path.exists(db):
            os.remove(db)
            logger.info("..deleted old series.db file because -d flag was set")    
        
    if not os.path.exists(db):
        try:
            os.makedirs(folder)
            logger.info("..created directory")
        except:
            pass    

        # You can add more elements with insert() below if you want to. It is 
        # recommended, however, to use an external tool / gui to prepare and
        # manage experiments. The following gives some simple 
        # examples of how to add content to series.db
        experiment1 = dict(
            name='myexperiment', # folder for the experiment
            param_topo_scenario_generator=1,
            param_topo_seed=100,
            param_debug_verify_with_simulator=1 # verify this experiment with simulator
        ) 
        # You can specify multiple experiments by adding entries with different
        # parameters. In this case, a second experiment is added with a different
        # seed. See topo/custom/pbce/exp800-2.py for all possible parameters.
        experiment2 = dict(
            name='myexperiment', # folder for the experiment
            param_topo_scenario_generator=1,
            param_topo_seed=101 # different parameter for the second experiment
        ) 
        # You can specify multiple parameters by using a string and separating
        # the experiments with a comma
        experiment3 = dict(
            name='myexperiment',
            param_topo_scenario_generator=1,
            param_topo_seed="150,155,160" 
        )
        # You can also specify ranges of parameters by using x-y-z where x is the
        # start delimiter, y is the end delimiter and z is the increment. The following
        # example will execute experiments with seed 200, 220, ..., 280, 300
        experiment4 = dict(
            name='myexperiment',
            param_topo_scenario_generator=1,
            param_topo_seed="200-300-20" 
        )     
        logger.info("..created new series.db file with some dummy entries")
        with open_table(db, "info") as table:             
            table.insert(experiment1)
            table.insert(experiment2)
            table.insert(experiment3)
            table.insert(experiment4)
    else:
        logger.info("..database seems to already exist, exit now")
        exit(0)

def run_command(params, scripts, global_state):
    """
    Run the simulator with params; script is used to get the script infos
    """
    script = scripts.get(params.get('id'))
    script_name = script.get('name')

    # scripts are stored in ./topo/script_name
    path_script = os.path.join(os.getcwd(), 'topo', script_name)

    # we use the main.py simulator script to start the individual experiments.
    # The scripts are passed via the -f parameter. The configuration is passed
    # via the -c parameter (but we have to write the config file first!)
    path_sim = os.path.join(os.getcwd(), 'main.py')
    if not os.path.exists(path_sim):
        logger.error('path_sim %s not found' % path_sim)

    # construct path for the results
    path_series = script.get('path_series')
    path_results = os.path.join(path_series, script_name.replace('.py', ''), params.get('uuid'))
    try:
        os.makedirs(path_results)
    except:
        pass

    # get the new run number based on the existing folders in the data directory
    blacklist = ['.cache']
    old_runs = []
    for root, dirs, files in os.walk(path_results):
        for d in dirs:
            if d not in blacklist: old_runs.append(d)
    ids = [int(''.join(filter(lambda x: x.isdigit(), r))) for r in old_runs]
    runid = 1
    if len(ids) > 0:
        runid = max(ids)+1

    path_experiment = os.path.join(path_results, 'run%.4d' % runid)
    try:
        os.makedirs(path_experiment)
    except:
        pass 

    # write config file
    path_config = os.path.join(path_experiment, 'config.json')
    with open(path_config, 'w') as file:
        file.write(json.dumps(params, indent=4, sort_keys=True))

    path_interpreter = os.path.join(os.getcwd(), 'venv', 'bin', 'python')
    command = '%s %s -f %s -c %s' % (path_interpreter, path_sim, path_script, path_config)
    if args.nopickle:
        command += ' -n'

    timings = global_state['timings'][:]
    avg = 0
    eta = 0
    left = 'N/A'
    passed = display_time(int(time.time() - global_state.get('started')))
    if len(timings) > 0:
        avg = sum(timings)/len(timings)/MAX_RUNNERS
        todo = global_state.get('todo')
        done = global_state.get('done')
        eta = int((todo-done)*avg)
        left = display_time(eta)

    print("run %s.%s [%d/%d] rt=%s avg=%.2fs eta=%s" % (script_name.replace('.py', ''), params.get('uuid'), 
        global_state.get('done'), global_state.get('todo'), passed, avg, left))

    try:
        st = time.time()
        output = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True,
            cwd=os.getcwd(),
        )
        outfile = path_config.replace('config.json', 'out.txt')
        with open(outfile, 'w') as file:
            file.write(output)
        params['path'] = path_experiment
        
        global_state['done'] += 1
        global_state['timings'].append(time.time()-st)
        return params, scripts

    except subprocess.CalledProcessError as e:
        print(e)
        for k, v in params.items():
            print('  ', k, v)
        output = e.output

    return None, None


@asyncio.coroutine
def run_command_on_loop(loop: asyncio.AbstractEventLoop, command, script, global_state):
    """
    Run commands in loop
    """
    with (yield from semaphore):
        runner = partial(run_command, command, script, global_state)
        output = yield from loop.run_in_executor(None, runner)
        yield from asyncio.sleep(0.1)  # Slowing a bit for demonstration purposes
        return output


@asyncio.coroutine
def run_all_commands(command_list: Sequence[dict], script, global_state):
    """
    Run all commands in a list
    """
    loop = asyncio.get_event_loop()
    fs = [run_command_on_loop(loop, command, script, global_state) for command in command_list]
    for f in asyncio.as_completed(fs):
        result, scripts = yield from f
        ensure_future(process_result(result, scripts))


@asyncio.coroutine
def process_result(result, scripts):
    """
    Update database with new results. This is only required for the gui that is not part
    of this repository and can be ignored or removed.
    """
    if result:
        #print("result", result)
        if result.get("info_id"):
            raise RuntimeError("info_id found in result, should not happen; Exit now.")
        script = scripts.get(result.get('id'))
        path_series = script.get('path_series')
        series_db = os.path.join(path_series, 'series.db')
        with open_table(series_db, 'results') as table:
            result['info_id'] = result['id']
            del result['id']
            id = table.upsert(result, ['path'])

        # update the todo table in the database
        db = dataset.connect('sqlite:///%s' % series_db)
        result = db.query('UPDATE todo SET runs_total = runs_total - 1 WHERE todo = 1')

def main():

    # -i (new empty experiment)
    if args.init:
        init_new_experiment_series(args.init)
        exit(0)

    # make sure a folder was given via -f and this path exists
    if not args.folder:
        logger.error('The -f options is required. Exit now.')
        exit(1)

    path_series = args.folder
    if not os.path.exists(args.folder):
        # handle relative paths
        if not os.path.isabs(args.folder):
            path_series = os.path.join(os.getcwd(), args.folder)
        try:
            os.makedirs(path_series)
            logger.info('..the provided folder did not exist and was created')
        except:
            logger.error('..there was an error creating folder=%s. Exit now.' % args.folder)
            exit(1)
            pass 

    # Progress of a series is stored in a sqlite database 
    # so that other processes can access it (the data of the running experiments,
    # NOT the actual results!). If no such database exists, the series was not yet started
    series_db = os.path.join(path_series, 'series.db')
    if not os.path.exists(series_db): 
        logger.info("..no database exists for this experiment series")

    # get all entries (stored in info table)
    entries = []
    with open_table(series_db, 'info') as table:
        entries = table.all()

    scripts = {} # contains all script entries from series.db->info
    allwork =  [] # an array that stores all work that has to be done
    for entry in entries:
        id = entry.get('id')
        if not id:
            raise RuntimeError('there should always be an id...')
        # we add a new entry for this dict to the global scripts array
        scripts[id] = dict(entry)
        scripts[id]['path_series'] = path_series # store path to series folder

        
        # we also calculate the unrolled params for this entry of the series
        unrolled_params = {}
        for k, v in entry.items():
            if k.startswith('param_'):
                # params can be given in the form a-b-c which unrolls
                # to range(a,b+1,c)
                if v is None:
                    continue;
                if type(v) is int:
                    unrolled_params[k] = []
                    unrolled_params[k].append((id, k,v))
                    continue;  
                if len(v) == 0:
                    continue;
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

        # after that, we create the param dicts for all unrolled runs
        from pprint import pprint
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
           
            for id, k, v in c:
                newdict[k] = v
                newdict['id'] = id # reference to the entry in the info table

            dictarray.append(newdict)
        allwork += dictarray

    # We now have an array where all the runs that are currently associated with
    # series.db are stored. This looks like:
    # [ 
    #  {'id': 2, 'param_pbce_threshold': 500, 'param_pbce_threshold_offset': 50},
    #  {'id': 2, 'param_pbce_threshold': 525, 'param_pbce_threshold_offset': 50},
    #  {'id': 2, 'param_pbce_threshold': 625, 'param_pbce_threshold_offset': 50},
    #  {'id': 2, 'param_pbce_threshold': 650, 'param_pbce_threshold_offset': 50}, 
    #  ... potentially many more entries ... 
    # ]
    #pprint(allwork)
    logger.info(".. found %d entries in total" % len(allwork))

    filtered_work = []
    
    # Next we get all the entries from the results table in the database;
    # Because every single experiment is stored in one folder, we can use this
    # path as a unique identifier for the dict
    database = {}
    with open_table(series_db, 'results') as table:
        items = table.all()
        for item in items:
            database[item.get('path')] = dict(item)

    # check existing data on filesystem 
    for params in allwork:
        script = scripts.get(params.get('id'))
        iterations = script.get('iterations', 1)
        if not iterations:
            iterations = 1
        path_results = os.path.join(path_series, script.get('name').replace('.py', ''), params.get('uuid'))
        found_past_runs = []
        if os.path.exists(path_results):
            # get the new run number based on the existing folders in the data directory
            blacklist = ['.cache']
            old_runs = []
            to_deletes = []
            for root, dirs, files in os.walk(path_results):
                if len(os.listdir(root)) == 0:
                    logger.info(".. delete empty directory: %s" % root)  
                    os.rmdir(root)
                for d in dirs:
                    if d not in blacklist: 
                        checkdir = os.path.join(root, d)
                        if 'switch_' in checkdir: continue;
                        f1 = os.path.join(checkdir, 'config.json')
                        f2 = os.path.join(checkdir, 'result.pickle')
                        f3 = os.path.join(checkdir, 'statistics.json')
                        for f in [f1, f3]:
                            if not os.path.exists(f):
                                if not checkdir in to_deletes: to_deletes.append(checkdir)
                        if not checkdir in to_deletes:
                            # even if all files are present some of them could be broken
                            # check statistics in that case
                            with open(f3, 'r') as file:
                                try:
                                    data = json.loads(file.read())
                                except:
                                    if not checkdir in to_deletes: to_deletes.append(checkdir)
                                    continue   
                            found_past_runs.append(checkdir)
            
            # run through all folders that are not deleted and check whether they are present
            # in the database; if not, the database is updated
            for path in found_past_runs:
                if not database.get(path):
                    logger.info(".. update database: "  + path)
                    config = os.path.join(path, 'config.json')
                    try:
                        with open(config, 'r') as file:
                            data = json.loads(file.read())
                            with open_table(series_db, 'results') as table:
                                data['info_id'] = data['id']
                                data['path'] = path
                                del data['id']
                                id = table.upsert(data, ['path'])
                    except Exception as e:
                        logger.error('exception while updating database: %s; remove run;' % str(e))
                        if not path in to_deletes: to_deletes.append(path)

            # run through the folders marked for deletion and actually delete them
            for to_delete in to_deletes:
                logger.info('.. deleting invalid folder: ' + to_delete)
                try:
                    os.rmdir(to_delete) 
                except Exception as e:
                    # better be careful with rmtree...
                    # you can adopt or remove this line
                    #assert(to_delete.startswith(os.path.join(os.getcwd(), 'path')))
                    shutil.rmtree(to_delete)

            # do we have enough iterations?
            if len(found_past_runs) < iterations:
                for i in range(len(found_past_runs), iterations):
                    filtered_work.append(dict(params)) 
        else:
            # these experiments are still missing completely
            for i in range(0, iterations):
                filtered_work.append(dict(params))    

    if len(filtered_work) == 0:
        logger.info(".. all runs specified in the database were executed; nothing to do")
    else:
        logger.info(".. found %d open tasks; starting now.." % len(filtered_work))
        # update the todo table in the database
        with open_table(series_db, 'todo') as table:
            runs_total = len(filtered_work)
            todo_entry = dict(
                todo=1, # make sure we use the same entry
                runs_total=runs_total
            )
            table.upsert(todo_entry, ['todo'])

    # We are now synchronized with the database and the filesystem
    # and can start the actual simulations
    if len(filtered_work) > 0:
        global_state = dict(
            started=time.time(),
            todo=len(filtered_work),
            done=0,
            timings=[]  
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_all_commands(filtered_work, scripts, global_state))
        logger.info(".. simulation part done")

    # run aggregated plotter
    #os.system()
    if args.plotter:
        if not args.plotter.endswith('.py'):
            args.plotter += '.py'
        path_sim = os.path.join(os.getcwd(), 'main.py')
        path_interpreter = os.path.join(os.getcwd(), 'venv', 'bin', 'python')
        command = '%s %s -a %s -u %s' % (path_interpreter, path_sim, path_series, args.plotter)

        print(command)
        try:
            logger.info(".. run plotter")
            output = subprocess.check_output(
                command,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True,
                cwd=os.getcwd(),
            )
            print(output)
            #outfile = path_config.replace('config.json', 'out.txt')
            #with open(outfile, 'w') as file:
            #    file.write(output)
            #params['path'] = path_experiment
            #return params, scripts
        except subprocess.CalledProcessError as e:
            print(e)
            output = e.output


if __name__ == "__main__":
    main()