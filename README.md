# Evaluation Environment for Flow Delegation (FDeval)

FDeval is a combined solver/simulator for *Flow Delegation* in the context of software-defined networks. The basic idea behind flow delegation is as follows: In case of a flow table capacity bottleneck, a subset of the flow rules in the bottlenecked delegation switch is "relocated" to a neighboring remote switch with spare capacity. If a flow rule is relocated, all traffic for that rule is redirected towards and processed within the remote switch. 

This evaluation environment can create different flow table utilization scenarios based on a wide range of configurable parameters. These scenarios can then be analyzed with a set of flow delegation algorithms. The simulator part in the /core folder works like a high-level flow simulator and can also be used for other purposes. The solver part in the /engine folder is dedicated to the flow delegation concept.

## Install the FDeval environment

First clone the git repository and create a new virtual environment.

`$ git clone ... FDeval`
`$ cd FDeval`
`$ python3 -m venv venv`
`$ source venv/bin/activate`

Then install the required packages inside the virtual environment via pip.

`(venv) $ pip install -r requirements.txt`

After that, install the gurobi solver by following the instructions in https://www.gurobi.com/documentation/8.1/quickstart_linux/software_installation_guid.html. While it would be possible to refactor parts of the code to be used without gurobi, the current code base depends on importing the gurobi module. You will receive an error like the following if you skip this step: ModuleNotFoundError: No module named 'gurobipy'.

It is assumed that gurobi is installed to /opt/gurobi810 (the 810 refers to the gurobi version and can of course be updated). It is further assumed that the python interface is installed, see http://www.gurobi.com/documentation/7.5/quickstart_mac/the_gurobi_python_interfac.html. This is achieved by:

`cd /opt/gurobi810/linux64`
`python setup.py install`

Next, it is required to copy gurobi.so to the venv folder (if using venv, you can also install it globally and use it without venv).

`cp /opt/gurobi810/linux64/lib/python3.6_utf32/gurobipy.so venv/lib/python3.6/site-packages/`

Finally, gurobi requires a licence which can be obtained on the gurobi website and enabled with grbgetkey (gurobi provides free licences for academic purposes). After the licence is installed, you should be able to run main.py and test.py without errors.

## How to create, run and analyze flow delegation experiments using FDeval

First, create a new experiment series which is stored in a series.db file (sqlite3 database). An example series.db filled with a couple of working dummy experiments can be created with the following command:

`$ python3 test.py -i data/test`

It is explained in detail in test.py (see function init_new_experiment_series) how this works and how to customize the parameters inside series.db. The command will create a new folder *data* and a subdirectory *test* inside the data folder. It will also create a new series.db file that contains the parameters for the experiment. Use the -d flag to overwrite an existing database. Next, the experiment series in the series.db file has to be executed. This is done with:

`$ python3 test.py -f data/test`

Note that this can run for days or even weeks depended on the number of experiments. Use the -t flag to define the number of CPU cores in order to exploit parallelism (default is only 4 cores). test.py makes use of asyncio and can scale with arbitrary number of cores (tested with 64 cores).

The raw results of the experiment series will be stored in a folder in the same directory as the series.db file created with the -i command.
Next, a first preprocessing step is required that will take the raw data in this folder and write it into a series of .pickle files. This step is introduced here because the raw data requires a large amount of disk space. Aggregating the raw data into a set of equally sized .pickle files makes the final processing step much easier and faster for large experiments.

`$ python3 main.py -w data/test`

The preprocessed data is now stored in one or multiple .pickle files (in the same directory as the series.db file). By default, the tool will split up the raw data into chunks of 500mb so that they can be easily copied to an analysis machine. You can change the default value of 500mb with the maxsize argument in the constructor of AggregatePreprocessor in plotter/util/agg_preprocessor.py if required.

`$ python3 main.py -r data/test`

This will create the final statistics.db file that contains all relevant data needed for plotting and analysis. Note that this step may also take rather long (up to hours) for a large experiment series with 100.000+ experiments. Also note that the final statistics.db file can be several GB in size. 

The final step that includes analysis and plotting is only done on top of statistics.db. None of the other files need to be present in the folder in order to start this process. This is done as follows (using an example analysis script called agg_01_info.py that will plot some basic statistics to the terminal):

`python3 main.py -a data/test -u agg_01_info.py`

The tool supports caching of result queries to speed up analysis tasks. This means that each unique SQL query to statistics.db is only executed once as long as the database is not changed and the cached files are not deleted. Note that it can be benefitial to manually create an index inside statistics.db if the database is too large and the SQL queries take too long. Cached entries are stored as json files on disc and can also be rather large if used without caution. You can clear the cache manually by deleting the .cache folder created in the root directory (the one with series.db and statistics.db). 

## Reproduce results from the existing data sets

All data sets listed below were created with the process described above. In result, each data set is given as a separate statistics.db file and analysis can be done by just executing the last step. The following shows this with an example of data set D21 that consists of handcrafted scenarios with different amounts of switches.

1. Download the data set and put the statistics.db file into an arbitrary folder, e.g., data/d21
2. Execute the analysis script, e.g., `python3 main.py -a data/d21 -u agg_01_info.py`

For this example data set, the output of agg_01_info.py looks as follows:

<pre>
INFO:__main__:run aggregated plotter with folder=data/d21
INFO:__main__:use plotter: agg_01_info.py
INFO:plotter.util.agg_plotter:loading custom plotter: plotter.agg_01_info
INFO:plotter.util.agg_plotter:prepare ResultSet blob with db=/home/bauer/Repos/flow-delegation/data/d21/series.db
INFO:plotter.util.plotutils:using statistics.db instead of series.db
no cache, run
open
INFO:plotter.util.plotutils:------------- query -----------
INFO:plotter.util.plotutils:
                SELECT param_debug_total_time_limit,param_topo_bottleneck_cnt,param_topo_bottleneck_duration,param_topo_bottleneck_intensity,param_topo_iat_scale,param_topo_idle_timeout,param_topo_num_flows,param_topo_num_hosts,param_topo_num_switches,param_topo_scenario_ba_modelparam,param_topo_scenario_generator,param_topo_seed,param_topo_switch_capacity,param_topo_traffic_interswitch,param_topo_traffic_scale,scenario_switch_cnt,scenario_table_capacity,json_scenario_concentrated_switches,json_scenario_edges,json_scenario_bottlenecks,json_scenario_hosts_of_switch,rsa_solver_cnt_infeasable,param_topo_num_switches,param_topo_num_hosts,param_topo_num_flows,param_topo_switch_capacity,param_topo_bottleneck_cnt,param_topo_scenario_ba_modelparam,param_topo_traffic_interswitch,param_topo_traffic_scale,param_debug_total_time_limit
                FROM statistics;
INFO:plotter.util.plotutils:processed entries: 3120
INFO:plotter.util.plotutils:------------- query done -----------
>> dataset d21
>> filesize 330.00
>> experiments 3120
>> time_limit 0
>> unique_seeds 10
>> param_topo_num_switches 10-300
>> param_topo_num_hosts 250-1000
>> param_topo_num_flows 100000-150000
>> param_topo_switch_capacity 30-90
>> param_topo_bottleneck_cnt 1
>> param_topo_concentrate_demand -
>> param_topo_scenario_ba_modelparam 1-2
>> param_topo_traffic_interswitch 75
>> param_topo_traffic_scale 100
>> param_dts_algo -
>> param_dts_look_ahead -
>> param_dts_weight_table -
>> param_dts_weight_link -
>> param_dts_weight_ctrl -
>> param_rsa_look_ahead -
>> param_rsa_max_assignments -
>> param_rsa_weight_table -
>> param_rsa_weight_link -
>> param_rsa_weight_ctrl -
>> param_debug_total_time_limit 5000-50000
>> param_dts_timelimit -
>> param_rsa_timelimit -
</pre>

The SQL query in the top is only executed once. If the same analysis script is executed a second time, the query result is taken from the cached json file instead which is much faster. The part in the the bottom shows a brief overview of the parameters that were used to create this data set. 

The actual results of an experiment series are not yet visible. These are stored in other columns of the databse (those not starting with param). To see how this data can be accessed, have a look at the analysis scripts in the plotter folder. There is one analysis script for each data set listed in the table below. The scripts are used in the same way as before. In this case, for data set D21, the command would look like this:
 
`python3 main.py -a data/d21 -u agg_01_d021_scale_switches`

This will put the evaluation result into the folder cwd/data/plots where cwd is the current working directory of the executing script. In this concrete example, the script should produce four plots named scale_switches_x.pdf with x = {50, 90, 99, 100}. The folder where the results are stored can be changed by overwriting EXPORT_FOLDER in plotter/agg_2_utils.py (e.g., to an absolute path). Note that most analysis scripts require additional libraries (matplotlib and numpy). 


## Available data sets

| Data set | Description |
|-------|-----------------------------------------------------------------|
| D1    | Randomly generated dataset with 500.000 different scenarios used in the scenario generation process.    |
| D2    | Consists of handcrafted scenarios with different scenario generation and bottleneck parameters.              |
| D3    | Consists of handcrafted scenarios with different topology parameters.                                                  |
| D20   | Consists of handcrafted scenarios with different amounts of considered delegation templates.         |
| D21   | Consists of handcrafted scenarios with different amounts of switches in the topology.           |
| D30   | Contains the scenarios from scenario set Z100 with different values for the DT-Select look-ahead factor (L=1 to L=9).    |
| D31   | Same as dataset D30 above but the DT-Select look-ahead factor is varied between 1 and 50.                                          |
| D40   | Contains the scenarios from scenario set Z100 with different values for the RS-alloc look-ahead factor (L=1 to L=9) |
| D41   | Same as dataset D40 above but the RS-Alloc look-ahead factor is varied between 1 and 30                       |
| D50   | Contains the scenarios from scenario set Z100 with different weights for the DT-Select algorithms |
| D60   | Contains the scenarios from scenario set Z100 with different weights for the RS-alloc algorithm |
| D100  | Contains the scenarios from scenario set Z100 with all capacity reduction factors between 1% and 80% |
| D110  | Contains four selected scenarios from Z100 with capacity reduction factors between 1% and 80%. |
| D5000 | Contains the scenarios from scenario set Z5000 where each scenario is assigned a random capacity reduction factor  between 1% and 80%  |
| D5050 | Contains the scenarios from scenario set Z5000 but only with Select-CopyFirst and the execution was restricted to a single CPU core.  |


## Data set characteristics


| Dataset | Size     | Experiments | Timeouts | Scenarios | Switches | Hosts    | Pairs         | Capacity Reduction | Bottlenecks | Hotspots | M-Parameter | Inter Switch Ratio | Traffic Scale | DT-Select      |
|---------|----------|-------------|----------|-----------|----------|----------|---------------|--------------------|-------------|----------|-------------|--------------------|---------------|----------------|
| D1      | 1356.86  | 500000      | 3164     | 500000    | 2-15     | 10-300   | 25000-249999  | 1-80               | 0-20        | 0-4      | 1-5         | 20-80              | 25-12500      | None           |
| D2      | 4807.54  | 64800       | 0        | 10        | 6        | 100      | 100000        | 1                  | 0-5         | 0-2      | 1-3         | 20-75              | 0             | None           |
| D3      | 48.72    | 1458        | 0        | 6         | 4-12     | 100-200  | 150000        | 1                  | 0           | 0        | 1-3         | 0                  | 0             | None           |
| D20     | 1197.71  | 1600        | 0        | 10        | 1        | 25-500   | 100000-200000 | 10-70              | 0           | 0        | 1           | 0                  | 0             | CopyFirst      |
| D21     | 330.00   | 3120        | 0        | 10        | 10-300   | 250-1000 | 100000-150000 | 10-70              | 1           | 0        | 1-2         | 75                 | 100           | CopyFirst      |
| D30     | 354.37   | 1800        | 266      | 100       | 2-15     | 13-244   | 44322-246377  | 1-73               | 0-9         | 0-3      | 1-5         | 20-70              | 25-5250       | Opt, CopyFirst |
| D31     | 1236.34  | 8488        | 0        | 100       | 2-15     | 13-244   | 44322-246377  | 1-73               | 0-9         | 0-3      | 1-5         | 20-70              | 25-5250       | CopyFirst      |
| D32     | 1310.32  | 7200        | 995      | 100       | 2-15     | 13-244   | 44322-246377  | 10-40              | 0-9         | 0-3      | 1-5         | 20-70              | 25-5250       | Opt, CopyFirst |
| D40     | 1096.76  | 5400        | 0        | 100       | 2-15     | 13-244   | 44322-246377  | 1-73               | 0-9         | 0-3      | 1-5         | 20-70              | 25-5250       | CopyFirst      |
| D41     | 699.18   | 15000       | 12       | 100       | 2-15     | 13-244   | 44322-246377  | 1-73               | 0-9         | 0-3      | 1-5         | 20-70              | 25-5250       | CopyFirst      |
| D50     | 15943.81 | 70713       | 52       | 97        | 2-15     | 14-300   | 37339-245537  | 1-69               | 0-10        | 0-4      | 1-5         | 20-70              | 25-8750       | CopyFirst      |
| D60     | 3506.99  | 73000       | 0        | 100       | 2-15     | 13-244   | 44322-246377  | 1-73               | 0-9         | 0-3      | 1-5         | 20-70              | 25-5250       | CopyFirst      |
| D61     | 8223.20  | 39700       | 30       | 100       | 2-15     | 14-300   | 37339-245537  | 1-69               | 0-10        | 0-4      | 1-5         | 20-70              | 25-8750       | CopyFirst      |
| D100    | 2273.61  | 8000        | 93       | 100       | 2-15     | 14-300   | 37339-245537  | 1-80               | 0-10        | 0-4      | 1-5         | 20-70              | 25-8750       | CopyFirst      |
| D101    | 2242.10  | 8000        | 9        | 100       | 2-15     | 13-244   | 44322-246377  | 1-80               | 0-9         | 0-3      | 1-5         | 20-70              | 25-5250       | CopyFirst      |
| D110    | 135.66   | 320         | 0        | 4         | 5-13     | 56-203   | 83758-231821  | 1-80               | 0-4         | 0-1      | 1-2         | 40-70              | 75-500        | CopyFirst      |
| D250    | 6638.71  | 31250       | 121      | 250       | 2-15     | 10-286   | 29677-244196  | 1-79               | 0-20        | 0-3      | 1-5         | 20-80              | 25-9000       | CopyFirst      |
| D5000   | 2984.41  | 15000       | 905      | 5000      | 2-15     | 10-299   | 26129-249948  | 1-80               | 0-20        | 0-4      | 1-5         | 20-80              | 25-12500      | All three      |
| D5050   | 1062.96  | 5000        | 5        | 5000      | 2-15     | 10-299   | 26129-249948  | 1-80               | 0-20        | 0-4      | 1-5         | 20-80              | 25-12500      | CopyFirst      |

## Write new analysis scripts

The statistics.db files can be analyzed with any existing sqlite-capable database browser. However, FDeval supports several helper functions for common analysis tasks. These can be found in the plotter folder, primarily in agg_02_utils.py and the files in the util subfolder. In order to create new analysis scripts to be used with main.py, just add a new file in the plotter folder that starts with the prefix "agg". A minimum working example of such a file looks as follows:

<pre>
# Minimum working example of an analysis script
def plot(blob, **kwargs):
    # Define the columns that should be included in the SQL query; only these columns are
    # queried which is important because the data sets can contain hundreds of these columns.
    # Note that parameters (param_xxx) are always included, i.e., it is not required to include 
    # columns such as "param_topo_seed" manually. In this example, two result columns are
    # queried: scenario_switch_cnt and scenario_table_capacity. These are results of the experiment,
    # not input parameters.
    includes = ['scenario_switch_cnt', 'scenario_table_capacity']

    # Some columns are not present in all data sets. One example is the hit_timelimit column
    # which is only present if at least one experiment experienced a timeout. The find_columns()
    # function will only add such columns to the includes array if it is present in the statistics.db file.
    includes += blob.find_columns('hit_timelimit')

    # This will configure the SQL query to perform a SELECT on all columns specified
    # by the includes array + all columns that start with param_. 
    blob.include_parameters(**dict.fromkeys(includes, 1))

    # This will trigger the SQL query and the caching mechanism. The result is an array
    # of python dicts.
    runs = blob.filter(**dict())

    # Do something with the results...
    for i, run in enumerate(runs):
        print("run_%d" % i, run.get('param_topo_seed'), run.get('scenario_switch_cnt'))
</pre>

The plot() function will be called by main.py and inject the data into the blob datastructure which is a ResultSet class instance as defined in plotter/util/plotutils.py. The new analysis script can than be executed like above

`python3 main.py -a data/test -u agg_01_minimum_example.py`

This will result in the following output:

<pre>
INFO:__main__:run aggregated plotter with folder=data/test
INFO:__main__:use plotter: agg_01_minimum_example.py
INFO:plotter.util.agg_plotter:loading custom plotter: plotter.agg_01_minimum_example
INFO:plotter.util.agg_plotter:prepare ResultSet blob with db=/home/bauer/Repos/flow-delegation/data/test/series.db
INFO:plotter.util.plotutils:using statistics.db instead of series.db
no cache, run
open
INFO:plotter.util.plotutils:------------- query -----------
INFO:plotter.util.plotutils:
                SELECT param_topo_scenario_generator,param_topo_seed,scenario_switch_cnt,scenario_table_capacity
                FROM statistics;
INFO:plotter.util.plotutils:processed entries: 12
INFO:plotter.util.plotutils:------------- query done -----------
run_0 280 3
run_1 220 3
run_2 200 3
run_3 260 3
run_4 155 3
run_5 101 3
run_6 160 3
run_7 150 3
run_8 240 3
run_9 100 3
run_10 1 3
run_11 300 3
</pre>
