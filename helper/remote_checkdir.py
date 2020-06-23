# small helper script to check results

import fnmatch
import os
import shutil
import time
import json
import sys
liste = []

CHECKFOLDER = sys.argv[1]
"""
# we do not need the pickle files here
for root, dirnames, filenames in os.walk(CHECKFOLDER):
    for filename in fnmatch.filter(filenames, 'scenario.pickle'):
        #os.remove(os.path.join(root, 'result.pdf'))
        print("remove", os.path.join(root, 'scenario.pickle'))
        os.remove(os.path.join(root, 'scenario.pickle'))

os.system("rm -rf pdfs")
os.system("mkdir pdfs")
"""

missing = []
for root, dirnames, filenames in os.walk(CHECKFOLDER):
    if not 'errors' in root:
        for filename in fnmatch.filter(filenames, 'config.json'):
            #print("found", filename)
            if not os.path.exists(os.path.join(root, 'statistics.json')):
                missing.append(root)


error_folder = os.path.join(CHECKFOLDER, 'errors')
if not os.path.exists(error_folder):
    os.makedirs(error_folder)

for path in missing:
    new_path = path.replace(CHECKFOLDER, '').replace('/custom', '').replace('/run0001', '')
    if new_path.startswith('/'):
        new_path = new_path[1:]
    new_path = os.path.join(error_folder, new_path)

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    

    print("add errorreport: %s" % new_path)

    outfile =  os.path.join(path, 'out.txt')
    outfile2 = os.path.join(new_path, 'out.txt')
    if os.path.exists(outfile):
        shutil.copyfile(outfile, outfile2)

    config = os.path.join(path, 'config.json')
    config2 = os.path.join(new_path, 'config.json')
    if os.path.exists(config):
        shutil.copyfile(config, config2)

        importstring = ""
        with open(config, "r") as file:
            data = json.loads(file.read())
            for k, v in data.items():
                if k.startswith('param_'):
                    importstring += '%s %d\n' % (k, v)

        if len(importstring) > 0:
            importstring_path = os.path.join(new_path, 'import.txt')
            with open(importstring_path, 'wb') as file:
                file.write(importstring)

    scenario = os.path.join(path, 'scenario.pdf')
    scenario2 = os.path.join(new_path, 'scenario.pdf')
    if os.path.exists(scenario):
        shutil.copyfile(scenario, scenario2)


exit(0)
