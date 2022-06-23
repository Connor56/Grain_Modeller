'''
Name:
    Output_Parser
Description:
    Reads output files, for example log.lammps, and gulp output, parses the
    output for useful data and places it into a pandas DataFrame.
'''

import numpy as np
import re
import pandas as pd
from warnings import warn


class SimLog():

    def __init__(self, name, raw_data, runs=None):
        '''
        Initialise the parsed log data class, stores LAMMPS log data for future
        analysis.
        '''
        self.name = name
        self.raw_data = raw_data
        self.runs = runs

    def __repr__(self):
        return (f"SimLog('{self.name}', "
               + f"{self.raw_data}', "
               + f"runs={self.runs})")


class Run():

    def __init__(self, commands, data, compute_time):
        '''
        Holds the thermo output resulting from LAMMPS commands that run
        simulations.
        '''
        self.commands = commands
        self.data = data
        self.compute_time = compute_time

    def __eq__(self, o):
        try:
            variable_1 = self.commands == o.commands
            variable_2 = np.all(self.data == o.data)
            variable_3 = self.compute_time == o.compute_time
            if variable_1 and variable_2 and variable_3:
                return True
            else:
                return False
        except AttributeError:
            return False

    def __repr__(self):
        return f"Run({self.commands}, \n{self.data}, {self.compute_time})"


def lammps_log(file, input_header):
    '''
    Parses a LAMMPS log file splitting the data up by input name and run type,
    input_header is used to find the input names. Further splits the log into
    the different runs which make up each complete simulation procedure. These
    are turned into run objects and added to the simulation log as a list.
    Complete SimLog objects are returned to the user for analysis.
    '''
    simulation_logs = split_lammps_log(file, input_header)
    simulation_logs = get_lammps_runs(simulation_logs)
    return simulation_logs


def split_lammps_log(file, input_header):
    '''
    Splits log file by input name, providing a raw string of information for
    each input stored in a parsed log class.
    '''
    if input_header[-1] == ':':
        raise ValueError(
            "Last character of input_header is a colon, if this isn't "
            + "deliberate it will break the code. Consider changing this.")
    with open(file, 'r') as f:
        input_data = f.read()
    search_string = (input_header.encode('unicode-escape').decode()
                     + r":\s+(\w+)\n")
    pattern = re.compile(search_string)
    matches = pattern.finditer(input_data)
    splits = [match.span()[0] for match in matches]
    rolled_splits = np.roll(np.array(splits), -1).tolist()
    rolled_splits[-1] = len(input_data)
    splits = list(zip(splits, rolled_splits))
    simulation_logs = []
    for split in splits:
        raw_data = input_data[split[0]: split[1]]
        matches = list(pattern.finditer(raw_data))
        data_index = matches[0].span()[1]
        name = matches[0].group(1)
        raw_data = raw_data[data_index:]
        if raw_data[-1] == '\n':
            raw_data = raw_data[:-1]
        log = SimLog(name, raw_data)
        simulation_logs.append(log)
    return simulation_logs


def get_lammps_runs(simulation_logs):
    '''
    Extract run data from raw output data of each simulation log.
    '''
    for log in simulation_logs:
        pattern = re.compile(r"(unfix.+|fix.+|minimize.+)")
        commands = list(pattern.finditer(log.raw_data))
        pattern = re.compile(r"Step.*")
        headers = list(pattern.finditer(log.raw_data))
        pattern = re.compile(r"Loop time.+")
        loop_times = list(pattern.finditer(log.raw_data))
        runs = []
        for output in list(zip(headers, loop_times)):
            run_commands = relevant_lammps_commands(commands, output)
            output_data, compute_time = format_lammps_output(
                output, log.raw_data)
            runs.append(Run(run_commands, output_data, compute_time))
        log.runs = runs
    return simulation_logs


def relevant_lammps_commands(commands, output):
    '''
    Finds the commands which occur before the run output and places them all in
    a list.
    '''
    run_commands = [command.group() for command in commands
                    if command.span()[1] < output[0].span()[0]]
    return run_commands


def format_lammps_output(output, raw_data):
    '''
    Selects the run data output from the raw data string, formats it into a
    Pandas DataFrame. Returns that along with the run loop time.
    '''
    run_output = raw_data[output[0].span()[1]+1: output[1].span()[0]-1]
    headers = raw_data[output[0].span()[0]: output[0].span()[1]]
    headers = headers.split(' ')
    if '' in headers:
        headers.remove('')
    pattern = re.compile(r"([-\d\.]+)[\s\t]*"*len(headers))
    output_data = list(pattern.finditer(run_output))
    output_data = [list(map(float, match.groups())) for match in output_data]
    output_data = np.array(output_data)
    output_data = {headers[n]: output_data[:, n] for n in range(len(headers))}
    output_data = pd.DataFrame(output_data)
    compute_time = raw_data[output[1].span()[0]: output[1].span()[1]]
    compute_time = float(compute_time.split(' ')[3])
    return (output_data, compute_time)
