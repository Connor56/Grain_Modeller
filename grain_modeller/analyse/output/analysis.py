'''
Name:
    Analysis
Description:
    Analyses the output of computational experiments, e.g. the variable output
    from a LAMMPS or GULP output file, or the recorded atomic positions of a
    grain from an xyz dump file. Utilises pyplot to produce graphs to make
    analysis of simulation logs easier.
'''

import numpy as np
import re
import matplotlib.pyplot as plt


def single_line_plot(simulation_logs, x_column, y_column, string_pattern=None,
        run=0, data_range=':', marker_type=None, title=None, show=False,
        save=True, legend=True, style=None):
    '''
    Produces matplotlib plots from simulation log data. Plotted data is defined
    by the given x and y columns. Logs can be chosen based on their name, using
    the string pattern variable. The desired run can be chosen along with the
    data range of interest. Marker Type and Title are used in pyplot. All
    simulation log data is collated on the same plot, choose if this plot is
    shown, and if it's saved. Choose a style.
    '''
    simulation_logs = select_logs(simulation_logs, string_pattern)
    if style is not None: plt.style.use(style)
    fig, ax = plt.subplots()
    for log in simulation_logs:
        x_data = get_log_data(log, run, x_column, data_range)
        y_data = get_log_data(log, run, y_column, data_range)
        ax.plot(x_data, y_data, label=log.name, marker=marker_type)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    if legend: ax.legend()
    if title is not None: ax.set_title(title)
    if show: plt.show()
    if save and title is not None: fig.savefig(title)
    if save and title is None: fig.savefig('No Title')


def multiple_line_plots(simulation_logs, x_column, y_column,
        string_pattern=None, run=0, data_range=':', marker_type=None,
        show=False, save=True, legend=True, style=None):
    '''
    Produces matplotlib plots from simulation log data. Plotted data is defined
    by the given x and y columns. Logs can be chosen based on their name, using
    the string pattern variable. The desired run can be chosen along with the
    data range of interest. Marker Type and Title are used in pyplot. All
    simulation logs have their own plot, choose if these plots are shown,
    and if they are saved. Choose a style.
    '''
    simulation_logs = select_logs(simulation_logs, string_pattern)
    if style is not None: plt.style.use(style)
    for log in simulation_logs:
        fig, ax = plt.subplots()
        x_data = get_log_data(log, run, x_column, data_range)
        y_data = get_log_data(log, run, y_column, data_range)
        ax.plot(x_data, y_data, label=log.name, marker=marker_type)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        if legend: ax.legend()
        if show: plt.show()
        if save: fig.savefig(log.name)


def select_logs(simulation_logs, string_pattern):
    '''
    Matches simulation log names to the given string pattern, any simulation
    log whose name doesn't match the string pattern is discarded.
    '''
    if string_pattern is None:
        return simulation_logs
    pattern = re.compile(string_pattern)
    simulation_logs = [log for log in simulation_logs
                       if len(list(pattern.finditer(log.name))) > 0]
    return simulation_logs


def get_log_data(simulation_log, run, column, data_range):
    '''
    Selects information from a simulation log, using the run, column, and data
    range to decide which information to select and return. Run is the ordered
    integer value of log run instance you're interested in. Column is the
    name of the column you want to get data from, and the data range is the
    range of the data you you want to get from that column. Data ranges can be
    given in string form for complex indexing: e.g. '0:-2'.
    '''
    if abs(run) > len(simulation_log.runs):
        raise ValueError(f"Run choice {run} is out of range.")
    if column not in simulation_log.runs[run].data.columns.values:
        raise ValueError(f"Column '{column}' not in run data.")
    log_data = simulation_log.runs[run].data[column]
    if isinstance(data_range, str):
        begin, end = data_range.split(':')
        begin = 0 if begin == '' else int(begin)
        if end == '':
            log_data = log_data.iloc[begin:].values
        else:
            end = int(end)
            log_data = log_data.iloc[begin:end].values
    else:
        try:
            log_data = log_data.iloc[data_range].values
        except AttributeError:
            log_data = log_data.iloc[data_range]
    return log_data
