'''
Name:
    Run All
Description:
    Runs all the test files in the folder, minus those which have been recorded
    as previously run with no errors or failures in the test log file.
    To clear this log file and run all tests again, use the '--new' option on
    the command line. To re-add files that have been modified since their last
    run use the '--add_modified' option on the command line.

    Log file can be set using the variable: LOG_FILE_NAME
'''

# Imports and Initial Setup
# -----------------------------------------------------------------------------
import os
import argparse
import re
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser(description='Select test files to run.')
parser.add_argument(
    '-n', '--new', action='store_true',
    help='Restart the run process, using all test files.')
parser.add_argument(
    '-am', '--add_modified', action='store_true',
    help="Use this flag to automatically add any files that have been "
         "modified since their last run back into testing pool.")
parser.add_argument(
    '-r', '--remove', nargs='+'
)
args = parser.parse_args()

# Directory for package modules.
SRC_DIRECTORY = '../grain_modeller'

# -----------------------------------------------------------------------------

# Log File Setup
# -----------------------------------------------------------------------------
LOG_FILE_NAME = 'test_log.csv'
dash_repeats = 70
equals = '='*dash_repeats
dashes = '-'*dash_repeats
print('\n'+equals, f"\n  Test Log File Name: '{LOG_FILE_NAME}'")

# Output for the console
run_undertaken = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"  Run Undertaken: {run_undertaken}")
print(equals)
print('  Passed Tests:\n')


# Create log file if it doesn't exist.
if not os.path.exists(LOG_FILE_NAME):
    with open(LOG_FILE_NAME, 'w') as f:
        f.write('Last Run')

# Reset the test log.
if args.new:
    with open(LOG_FILE_NAME, 'w') as f:
        f.write('Last Run')
# -----------------------------------------------------------------------------

# Get Passed Test Information
# -----------------------------------------------------------------------------

# Get passed tests.
test_log = pd.read_csv(LOG_FILE_NAME)
passed_tests = test_log.index.tolist()

# Get modification date times for tests.
test_modified = []
for test in passed_tests:
    modification_time = datetime.fromtimestamp(os.path.getmtime(test))
    modification_time = modification_time.strftime("%d/%m/%Y %H:%M:%S")
    test_modified.append(modification_time)
test_log['Test Modified'] = test_modified

# Get modification date times for modules.
module_modified = []
for test in passed_tests:
    module = test.replace('test_', '')
    module = module.replace('-', '/')
    module = SRC_DIRECTORY+f'/{module}'
    modification_time = datetime.fromtimestamp(os.path.getmtime(module))
    modification_time = modification_time.strftime("%d/%m/%Y %H:%M:%S")
    module_modified.append(modification_time)
test_log['Module Modified'] = module_modified

# Get Test Module Dependencies
test_dependencies = {}
imports = re.compile(r"import ([\w]+)")
for test in passed_tests:
    with open(test, 'r') as f:
        data = f.read()
        data = imports.findall(data)
        test_dependencies[test] = data
# -----------------------------------------------------------------------------

# Re-add Modified Files to tests
# -----------------------------------------------------------------------------
if args.add_modified:
    last_run = np.array([
        datetime.strptime(x, '%d/%m/%Y %H:%M:%S')
        for x in test_log['Last Run']])
    test_modified = np.array([
        datetime.strptime(x, '%d/%m/%Y %H:%M:%S')
        for x in test_log['Test Modified']])
    module_modified = np.array([
        datetime.strptime(x, '%d/%m/%Y %H:%M:%S')
        for x in test_log['Module Modified']])
    run_after_modification = (
        (last_run > test_modified) & (last_run > module_modified))
    re_added_tests = test_log[np.invert(run_after_modification)]
    test_log = test_log[run_after_modification]
    passed_tests = test_log.index.tolist()

# Output to the console
if not test_log.empty:
    print(test_log)
else:
    print('      None')

if "re_added_tests" in locals():
    if not re_added_tests.empty:
        print('\n'+dashes)
        print('  Re-added Tests:\n')
        print(re_added_tests)

print('\n'+dashes)
# -----------------------------------------------------------------------------

#


# Run Test Files
# -----------------------------------------------------------------------------
pattern = re.compile(r"test_[\w\-]+\.py")
test_files = [
    file for file in os.listdir('./') if pattern.match(file) is not None]
test_files = [file for file in test_files if file not in passed_tests]
if args.remove is not None:
    test_files = [file for file in test_files if file not in args.remove]
test_files.sort()

# Output for the console
print('  Remaining Tests:\n')
for test in test_files:
    print(f'      {test}')
print('\n'+equals+'\n')

run_strings = [f'python {file}' for file in test_files]
newly_passed_tests = []
for test in run_strings:
    print(f'TEST: {test}', '\n')
    output = subprocess.run(test, capture_output=True, shell=True, text=True)
    if output.returncode == 0:
        csv_row_string = test.replace('python ', '')+f',{run_undertaken}'
        csv_row_string = csv_row_string.split(',')
        newly_passed_tests.append(csv_row_string)
        continue
    print(output.stderr)
    print('\n'*3)
# -----------------------------------------------------------------------------

# Save Passed Tests
# -----------------------------------------------------------------------------
test_log = test_log.iloc[:, :1]
for test in newly_passed_tests:
    test_log.loc[-1] = [test[1]]
    index = test_log.index.tolist()
    index[-1] = test[0]
    test_log.index = index

# Console output
pd.DataFrame(columns=test_log.columns).to_csv(LOG_FILE_NAME, index=False)
if test_log.empty:
    pass
test_log.to_csv(LOG_FILE_NAME, header=None, mode='a')
print('\n'+equals)
print('  Newly Passed:\n')
if len(newly_passed_tests) > 0:
    for test in newly_passed_tests:
        print(f'      {test[0]}')
else:
    print('      None')
print('\n'+equals)
print('  Failed:\n')
test_files = [x for x in test_files if x not in test_log.index.tolist()]
for test in test_files:
    print(f"      {test}")
print('\n')
# -----------------------------------------------------------------------------
