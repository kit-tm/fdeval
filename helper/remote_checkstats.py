# small helper script to check results

import fnmatch
import os
import shutil
import time
import json
import sys

try:
    CHECKFOLDER = sys.argv[1]
except:
    CHECKFOLDER = './custom'
    if not os.path.exists(CHECKFOLDER):
        print("folder %s not found" % CHECKFOLDER)
        exit(1)

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
total_error_sum = 0
total_files_checked = 0
total_switches_checked = 0
total_result_none = 0

metrics = {}
metrics_maxerror = {}
for filename in todo:

    total_files_checked += 1

    with open(filename, 'r') as file:
        raw = file.read()

        """
        data = json.loads(raw)
        switch_cnt = data.get('scenario.switch_cnt')
        check_ctrl = data.get('rsa.ctrl.overhead_from_rsa')
        check_fairness = data.get('rsa.table.fairness_mse')
        assert((check_ctrl != None) == ('rsa.ctrl.overhead_from_rsa' in raw))
        assert((check_fairness != None) == ('rsa.table.fairness_mse' in raw))
        """

        if ('rsa.ctrl.overhead_from_rsa' not in raw) or ('rsa.table.fairness_mse' not in raw):
            total_error_sum += 1
            print("")
            print(filename)

            path = os.path.join(CHECKFOLDER, 'rsa_stat_errors')
            if not os.path.exists(path):
                os.makedirs(path)

            outfile = filename.replace('statistics.json', 'out.txt')
            newoutfile = os.path.join(path, 'error_%d_out.txt' % major_error_cnt)
            if os.path.exists(outfile):
                shutil.copyfile(outfile, newoutfile)

            configfile = filename.replace('statistics.json', 'config.json')
            with open(configfile, 'r') as file2:
                config = json.loads(file2.read())
                rerun = "\n"
                for k,v in config.items():
                    if 'param' in k:
                        rerun += "%s %d\n" % (k, v)
            new_filename = os.path.join(path, 'error_%d.txt' % major_error_cnt)
            major_error_cnt += 1
            with open(new_filename, 'w') as file3:
                file3.write(rerun)

            remove_folder = '/'.join(os.path.dirname(filename).split("/")[:-1])
            assert(sum(1 for i in remove_folder if i =='.') > 10) # experiment folders have lots of dots
            shutil.rmtree(remove_folder)


print(".. %-25s %d" % ('total_files_checked', total_files_checked))
print(".. %-25s %d" % ('total_error_sum', total_error_sum))

"""
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
"""