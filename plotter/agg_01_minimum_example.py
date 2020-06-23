# Minimum example for an analysis script

def plot(blob, **kwargs):
    "Minimum working example of an analysis script"

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
