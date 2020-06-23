# small helper script to check results

import fnmatch
import os
import shutil
import time
import json
import sys

CHECKFOLDER = sys.argv[1]

todo = []
missing = []
for root, dirnames, filenames in os.walk(CHECKFOLDER):
    if not 'errors' in root:
        for filename in fnmatch.filter(filenames, 'config.json'):
            # there should be a statistics json file here or something went wrong
            statistics = os.path.join(root, 'statistics.json')
            if not os.path.exists(statistics):
                missing.append(root)
            else:
                todo.append(statistics)

print(".. found %d statistics.json files" % len(todo))

major_error_cnt = 0
total_minor_errors = 0
total_error_sum = []
total_files_checked = 0
total_switches_checked = 0
total_result_none = 0

metrics = {}
metrics_maxerror = {}
for filename in todo:

    total_files_checked += 1

    with open(filename, 'r') as file:
        data = json.loads(file.read())

        switch_cnt = data.get('scenario.switch_cnt')

        for node in range(0, switch_cnt):
            total_switches_checked += 1

            d1 = 'dts.%d.verify.diff.ctrl_overhead' % (node)
            d2 = 'dts.%d.verify.diff.total_delegated_demand_in' % (node)
            d3 = 'dts.%d.verify.diff.total_demand_in' % (node)

            # check simulator diffs
            check = [d1, d2, d3]
            for c in check:
                result = data.get(c)
                if result == None:
                    total_result_none += 1
                if result > 0:
                    total_minor_errors += 1
                    total_error_sum.append(result)
                    #print("    >>", c, result)
                if result > 0.1:
                    path = os.path.join(CHECKFOLDER, 'verify_errors')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    configfile = filename.replace('statistics.json', 'config.json')
                    with open(configfile, 'r') as file2:
                        config = json.loads(file2.read())
                        rerun = "\n"
                        for k,v in config.items():
                            if 'param' in k:
                                rerun += "%s %d\n" % (k, v)
                    new_filename = os.path.join(path, 'error_%d.txt' % major_error_cnt)
                    major_error_cnt += 1
                    print("major error: ", new_filename)
                    with open(new_filename, 'w') as file3:
                        file3.write(rerun)

            # check metric results
            ctrl = 'dts.%d.ctrl_overhead_percent' % (node)
            link = 'dts.%d.link_overhead_percent' % (node)
            table = 'dts.%d.table_overhead_percent' % (node)
            under = 'dts.%d.underutil_percent' % (node)

            ctrl2 = 'dts.%d.verify_ctrl_overhead_percent' % (node)
            link2 = 'dts.%d.verify_link_overhead_percent' % (node)
            table2 = 'dts.%d.verify_table_overhead_percent' % (node)
            under2 = 'dts.%d.verify_underutil_percent' % (node)     

            check = [(ctrl, ctrl2), (link, link2), (table, table2), (under, under2)]
            for v1, v2 in check:
                r1 = data.get(v1)
                r2 = data.get(v2)
                if r1 is None or r2 is None:
                    total_result_none += 1 
                    continue
                m = v1.split('.')[2]
                error = abs(r1-r2)
                try:
                    metrics[m].append(error)
                except KeyError:
                    metrics[m] = [error]

                if error > 0.001:
                    try:
                        if error > metrics_maxerror[m][0]:
                            metrics_maxerror[m] = (error, filename)
                    except KeyError:
                        metrics_maxerror[m] = (error, filename)


print(".. %-25s %d" % ('total_files_checked', total_files_checked))
print(".. %-25s %d" % ('total_switches_checked', total_switches_checked))
print(".. %-25s %d" % ('total_result_none', total_result_none))
print(".. %-25s %d  (rounding errors)" % ('total_minor_errors', total_minor_errors))
print(".. %-25s %d" % ('total_major_errors', major_error_cnt))
print(".. %-25s %f" % ('total_error_sum', sum(total_error_sum)))

for metric, data in metrics.items():
    print(".. %-25s %-10s   [checked=%d  ok=%d sumError=%f maxError=%f]" % (metric, 
        str(sum([1  for x in data if x > 0.001])), len(data), 
        sum([1  for x in data if x < 0.001]), sum(data), max(data)))

for metric, filename in metrics_maxerror.items():
    print("-------- max error: %s ----------" % metric)
    print("  file: %s" % filename[1])
    print("  error: %f" % filename[0])
    print("---------------------------------" + "-"*len(metric))

    filename = filename[1].replace('statistics.json', 'config.json')
    with open(filename, 'r') as file:
        config = json.loads(file.read())
        rerun = "\n"
        for k,v in config.items():
            if 'param' in k:
                rerun += "%s %d\n" % (k, v)
        print(rerun)