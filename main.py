#!/usr/bin/env python
#
# Main class for FDeval. Used to execute a single experiment. Experiment series
# that consist of multiple experiments are started with test.py

import argparse
import logging
import os
import json
import pickle
import sys
import importlib
import traceback
import time

from plotter.util.agg_plotter import AggregatePlotter
from plotter.util.agg_preprocessor import AggregatePreprocessor

from engine.errors import TimeoutError
from core.statistics import Timer
from core.context import Context
from topo.topology import Topology
from core.simulator import Simulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='SDN RuleSim CLI')

parser.add_argument('-f', dest='filename', action='store', type=str, 
    help='Filename of the simulation script to be executed')

parser.add_argument('-s', dest='seed', action='store', type=str, 
    help='Global seed value to be injected (for running from CLI, deprecated)')

parser.add_argument('-c', dest='config', action='store', type=str, 
    help='Points to a configuration .yaml file that contains the simulation parameters')

parser.add_argument('-p', dest='picklefile', action='store', type=str, 
    help='Picklefile to be used for plotting (requires -u flag)')

parser.add_argument('-u', dest='plotters', action='store', type=str, 
    help='Plotters to be used with the -p flag')

parser.add_argument('-T', dest='run_tests', action='store_true', 
    help='Run all files in the test folder', default=False)

parser.add_argument('-g', dest='run_scenario_generator', action='store_true', 
    help='Run scenario generator instead of simulator', default=False)

parser.add_argument('-a', dest='aggregated_plots', action='store', type=str,
    help='Run plotters with series.db', default=False)

parser.add_argument('-w', dest='run_preprocessor', action='store', type=str,
    help='Run preprocessor with series.db and write output to *.pickle on disk', default=False)

parser.add_argument('-r', dest='run_preprocessor_read', action='store', type=str,
    help='Read pre-processed *.pickle files from disk and store results in statistics.db', default=False)

parser.add_argument('-t', dest='threads', action='store', type=int, 
    help='Number of cores for processing')

parser.add_argument('-e', dest='export', action='store', type=str,
    help='Export plot as pdf to the file provided (deprecated)', default=False)

parser.add_argument('-n', dest='nopickle', action='store_true', default=False, 
    help='Create no pickle files (save disk storage for large experiment series)')

args = parser.parse_args()

def run_plotter():
    """
    Run a plotter for a specified pickle file
    """
    logger.info("run plotter from pickle file: %s" % args.picklefile)
    ctx =  pickle.load(open(args.picklefile,"rb"))   
    plotters = args.plotters.split(',')
    # handle plotters
    for plotter in plotters:
        modulepath = "plotter.%s" % plotter.replace('.py', '').replace(os.sep, '.')
        modulepath = modulepath.replace('..', '.')
        logger.info("loading plotter: %s" % modulepath)
        importlib.import_module(modulepath).plot(ctx)

def run_aggregated_plotter():
    """
    Run a plotter using a series.db file
    """
    logger.info("run aggregated plotter with folder=%s" % args.aggregated_plots)
    plotter_name = args.plotters
    logger.info("use plotter: %s" % plotter_name)
    AggregatePlotter(args.aggregated_plots, plotter_name, args.export)

def run_preprocessor():
    logger.info("run preprocessor -w with folder=%s" % args.run_preprocessor)
    ap = AggregatePreprocessor(read=args.run_preprocessor_read, 
        write=args.run_preprocessor, threads=args.threads, maxsize=args.seed)
    ap.write_pickled_data()

def run_preprocessor_read():
    logger.info("run preprocessor -r with folder=%s" % args.run_preprocessor_read)
    ap = AggregatePreprocessor(read=args.run_preprocessor_read, 
        write=args.run_preprocessor, threads=args.threads, maxsize=args.seed)
    ap.read_pickled_data()

def run_tests():
    """
    Run some basic testfiles for the simulator backend in /topo/custom/tests folder;
    Note that this code is partially obsolete, dig into the code for details. Also
    note that test_routing_with_overutil.py and test_changepath_stacked.py are supposed 
    to fail at the moment (represent corner cases that are çurrently not properly supported).
    """
    success = []
    failure = []

    # get all test files
    testfiles = []
    blacklist = ["__init__.py", "testutil.py"]
    testdir = os.path.join(os.getcwd(), "topo", "custom", "tests")
    for root, dirs, files in os.walk(testdir):
        for file in files:
            if file in blacklist: continue;
            if file.endswith(".py"):
                 testfiles.append(file)
    logger.info("tests: %s" % ",".join(testfiles))

    # run them one by one and print a summary afterwards
    for testname in testfiles:
        # please not that print is used because it logs to stdout
        print("+------------------------+")
        print("| %s" % testname)
        print("+------------------------+")
        ctx = Context() 
        topo = Topology(ctx, filename='custom/tests/%s' % testname)
        try:
            sim = Simulator(ctx)
            errors = sim.run()
        except Exception as e:
            failure.append((testname, ['test crashed: %s' % str(e)]))
            continue;        
        if errors == None:
            # if the simulator didn't return an array the 
            # on_test_finished method is not implemented properly
            failure.append((testname, ['test did not return an array, prob. not implemented']))
            continue;
        if len(errors) == 0:
            success.append(testname)
        else:
            failure.append((testname, errors))

    logger.info("+------ Test Results ----+")
    for test in success:
        logger.info("| %30s PASSED" % test)
    if len(failure) > 0: logger.info("|--->")
    for test, errors in failure:
        logger.info("| %30s FAILED" % test)
    logger.info("+------------------------+")
    if len(failure) == 0:
        logger.info("| ALL TESTS PASSED")  
        logger.info("+------------------------+")

def get_scripts(folder):   
    """
    Return all potential experiment script files in folder
    """
    blacklist = ['__init__.py', 'topology.py', 'gml_parser.py', 'topo.py', 'testutil.py',
        'util.py', 'create_topo.py']
    scripts = []
    allfiles = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".py") and file not in blacklist:
                 allfiles.append(os.path.join(root, file))
                 filename = os.path.join(root.replace(folder, '')[1:], file)
                 scripts.append(filename)
    return scripts

def main():

    # -w (run preprocessor write)
    if args.run_preprocessor:
        run_preprocessor()
        exit(0)

    # -r (run preprocessor read)
    if args.run_preprocessor_read:
        run_preprocessor_read()
        exit(0)

    # -a (run aggregated plotter)
    if args.aggregated_plots:
        if not args.plotters:
            logger.error("The -u option is required for -a. Exit now.")
            exit(1)  
        if args.plotters == 'util/agg_preprocessor.py':
            args.run_preprocessor = args.aggregated_plots
            run_preprocessor()
            exit(0)
        run_aggregated_plotter()
        exit(0)
    
    # -p and -u (run plotter from pickle file)
    if args.picklefile:
        run_plotter()
        exit(0)

    # -t (run tests)
    if args.run_tests:
        run_tests()
        exit(0)

    # make sure the file that should be run exists
    if not args.filename:
        logger.error('The -f options is required. Exit now.')
        exit(1)

    try:
        # TODO: currently, filenames are relative to the topo folder
        topo_folder = os.path.join(os.getcwd(), 'topo')
        filename = args.filename.strip()
        if not filename.endswith('.py'): filename = filename + '.py'

        # make sure the file that should be run exists
        if not os.path.exists(os.path.join(topo_folder, filename)):
            scripts = get_scripts(topo_folder)
            print(scripts)
            for script in scripts:
                if filename == os.path.basename(script):
                    logger.info('use file: %s' % script)
                    filename = script
                    break;
            else:
                logger.error('The file specified via -f was not found: %s' % str(filename))
                # fallback to default
                topo_folder = os.path.join(os.getcwd(), 'topo')
                filename = os.path.join(topo_folder, 'custom/pbce/exp800-2.py')
                logger.error('Using fallback: %s' % filename)
                if not os.path.exists(os.path.join(topo_folder, filename)):
                    logger.error('Fallback failed')
                    exit(1)

        # create a new experiment context that stores all the data; this is
        # the central object that is basically injected everywhere from now on; try not
        # to add anything here except raw objects (i.e., no functions) due to
        # serialization issues
        ctx = Context()
        ctx.started = time.time()
        timer = Timer(ctx, 'main')
        timer.start('total_time')

        # enforce specific seed (stored in ctx)
        if args.seed:
            logger.info('Global seed set to %d via CLI' % int(args.seed))
            ctx.seed = int(args.seed)

        # only run the scenario generator (neither the simulator nor the DTS/RSA algorithms
        # are executed)
        if args.run_scenario_generator:
            logger.info('Note: -g flag is set, i.e., the scenario generator ctx flag is active and the simulator will not be executed')
            ctx.run_scenario_generator = True

        # inject config from configfile into ctx
        # this is useful if the tool is used in an automated fashion
        if args.config:
            with open(args.config, 'r') as file:
                config = json.loads(file.read())
                ctx.configfile = args.config
                ctx.config = config
                #logger.info(str(config) + " " + args.filename)

        # create topology 
        topo = Topology(ctx, filename=filename)

        if args.run_scenario_generator:
            logger.info('Run scenario generator instead of simulator (-g flag was set)')
            ctx.statistics['sim.skipped'] = 1

        # finally run the simulator if requested
        if not ctx.skip_main_simulation == True:
            sim = Simulator(ctx)
            sim.run()

        # print the statistics
        print_statistics(ctx)

        # save the aggregated statistics (statistics.json)
        if ctx.configfile:
            statistics_file = os.path.join(os.path.dirname(ctx.configfile), 'statistics.json')
            with open(statistics_file, 'w') as file:
                file.write(json.dumps(ctx.statistics)) 

        # saves a pickle result file from ctx to access all raw data later on
        # (skipped if -n flag is used)
        if not args.nopickle:
            if ctx.scenario is not None:
                timer.stop()
                return
                #raise Exception("ctx.scenario is set -> not possible to create a pickle file (-n option is mandatory here!")
            if ctx.configfile:
                sys.setrecursionlimit(100000)
                result_file = os.path.join(os.path.dirname(ctx.configfile), 'result.pickle')
                pickle.dump(ctx, open(result_file, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

        # deprecated
        if args.plotters:
            logger.info("run plotters..")
            plotters = args.plotters.split(',')
            # handle plotters
            for plotter in plotters:
                modulepath = "plotter.%s" % plotter.replace('.py', '').replace(os.sep, '.')
                modulepath = modulepath.replace('..', '.')
                logger.info("loading plotter: %s" % modulepath)
                importlib.import_module(modulepath).plot(ctx)

        timer.stop()

    except TimeoutError as e:
        timer.stop() 
        # maximum time has exceeded
        logger.info("Timelimit exceeded, abort")
        # create statistics file (this is not technically an error, some experiments are just
        # running too long)
        if ctx.configfile:
            statistics_file = os.path.join(os.path.dirname(ctx.configfile), 'statistics.json')
            ctx.statistics['hit_timelimit'] = time.time() - ctx.started
            print_statistics(ctx)
            with open(statistics_file, 'w') as file:
                file.write(json.dumps(ctx.statistics)) 
        # still create an error message for quick checks
        exc_string = traceback.format_exc()
        if ctx.configfile:
            error_file = os.path.join(os.path.dirname(ctx.configfile), 'timeout-error.txt')
            with open(error_file, 'w') as file:
                file.write(exc_string)
        # finally print the exception and exit
        print("Exception:")
        print('-'*60)
        print(exc_string)
        print('-'*60)
        exit(0)

    except Exception as e:
        timer.stop()
        print("Exception:")
        print('-'*60)
        exc_string = traceback.format_exc()
        print(exc_string)
        print('-'*60)
        # save the aggregated statistics (statistics.json)
        if ctx.configfile:
            error_file = os.path.join(os.path.dirname(ctx.configfile), 'error.txt')
            with open(error_file, 'w') as file:
                file.write(exc_string) 
        raise e

def print_statistics(ctx):
    print("")
    for k, v in sorted(ctx.statistics.items()):
        if k == 'dss_result':
            # many entries here
            print('dss_result (%d entries)' % len(v.get('delegation_status')))
        elif k == 'metrics.link_utils':
            # many entries here
            print('metrics.link_utils (%d entries)' % len(v))               
        else:
            if isinstance(v, list):
                if len(v) >= 2:
                    try:
                        print(k, '[List] min=%.4f max=%.4f avg=%.4f' % (min(v), max(v), sum(v)/len(v)))
                    except:
                        print(k, '[List]', v[0], '...', v[-1])
                if len(v) == 1:
                    print(k, '[List]', v[0])     
            else:
                print(k, v)

if __name__ == "__main__":
    main()

