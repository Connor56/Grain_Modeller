import unittest
import os
import sys
import pandas as pd
import numpy as np
import re
import copy


class TestOutputParser(unittest.TestCase):

    def test_SimLog(self):
        '''
        Is it initialised correctly?
        '''
        parsed_log = output_parser.SimLog('test', 'this is raw data')
        self.assertTrue(parsed_log.name == 'test')
        self.assertTrue(parsed_log.raw_data == 'this is raw data')

    def test_Run(self):
        '''
        Is it initialised correctly?
        '''
        commands = ['run a test']
        data = ['This', 'is', 'some', 'data']
        compute_time = 5.
        test_run = output_parser.Run(commands, data, compute_time)
        self.assertTrue(test_run.commands == ['run a test'])
        self.assertTrue(test_run.data == ['This', 'is', 'some', 'data'])
        self.assertTrue(test_run.compute_time == 5.)
        test_run_2 = output_parser.Run(
            copy.deepcopy(commands), copy.deepcopy(data),
            copy.deepcopy(compute_time))
        self.assertTrue(test_run == test_run_2)
        test_run.commands = ['different_name']
        self.assertFalse(test_run == test_run_2)
        self.assertFalse(test_run == 5)

    def test_lammps_log(self):
        '''
        Does it read data files correctly and get the data from them?
        '''
        test_data = ('input_header: test_1\nfix This is a test\nStep\n 1\n 2'
                     + '\nLoop time of 5.0\n')
        with open('test_1.log', 'w') as f:
            f.write(test_data)
        test_data = output_parser.lammps_log('test_1.log', 'input_header')
        self.assertTrue(test_data[0].name == 'test_1')
        expected_raw = ('fix This is a test\nStep\n 1\n 2'
                        + '\nLoop time of 5.0')
        self.assertTrue(test_data[0].raw_data == expected_raw)
        expected_runs = [output_parser.Run(
            ['fix This is a test'], pd.DataFrame({'Step': [1.0, 2.0]}),
            5.0)]
        self.assertTrue(test_data[0].runs == expected_runs)
        os.remove('test_1.log')

    def test_split_lammps_log(self):
        '''
        Does it correctly separate the different inputs log data?
        '''
        test_data = ('input_header: test_1\nhi\nthis is a test\ninput_header:'
                     + '   test_2\nthis is the end of the data')
        expected_names = ['test_1', 'test_2']
        expected_raw = ['hi\nthis is a test', 'this is the end of the data']
        with open('test.log', 'w') as file:
            file.write(test_data)
        parsed_logs = output_parser.split_lammps_log(
            'test.log', 'input_header')
        parsed_names = [log.name for log in parsed_logs]
        parsed_raw = [log.raw_data for log in parsed_logs]
        self.assertTrue(parsed_names == expected_names)
        self.assertTrue(parsed_raw == expected_raw)
        self.assertRaises(
            ValueError, output_parser.split_lammps_log, 'test.log', 'test:')
        os.remove('test.log')

    def test_get_lammps_runs(self):
        '''
        Does the function correctly get run output from a raw output string?
        '''
        output_file = 'test_files/test_output_parser_log_file.lammps'
        with open(output_file, 'r') as f:
            output_string = f.read()
        name = 'test'
        simulation_log = [output_parser.SimLog(name, output_string)]
        simulation_log = output_parser.get_lammps_runs(simulation_log)
        expected_run_data = pd.DataFrame(
            {'Step': [0.0, 1000.0], 'Temp': [0.0, 1.1448199],
             'E_pair': [-5146.8078, -5184.8572], 'E_mol': [0., 0.],
             'TotEng': [-5146.8078, -5184.7094],
             'Press': [6744.3059, 0.94514147],
             'v_num_grain_atoms': [1000., 1000.],
             'c_grain_energy': [-5146.8078, -5184.8572],
             'v_energy_per_atom': [-5.1468078, -5.1848572]})
        self.assertTrue(np.all(
            simulation_log[0].runs[0].data[:2] == expected_run_data))
        expected_run_data = pd.DataFrame(
            {'Step': [5000.0, 6000.0], 'Temp': [0.88322027, 1.0556786],
             'E_pair': [-5184.8605, -5184.8882], 'E_mol': [0., 0.],
             'TotEng': [-5184.7465, -5184.7519],
             'Press': [0.92392844, 1.0164741],
             'v_num_grain_atoms': [1000., 1000.],
             'c_grain_energy': [-5184.8605, -5184.8882],
             'v_energy_per_atom': [-5.1848605, -5.1848882]})
        self.assertTrue(np.all(
            simulation_log[0].runs[1].data[:2] == expected_run_data))
        expected_run_data = pd.DataFrame(
            {'Step': [10000.0, 11000.0], 'Temp': [1.0520877, 1.0520877],
             'E_pair': [-5184.8775, -5185.0072], 'E_mol': [0., 0.],
             'TotEng': [-5184.7417, -5184.8713],
             'Press': [0.94515543, 1.6891786],
             'v_num_grain_atoms': [1000., 1000.],
             'c_grain_energy': [-5184.8775, -5185.0072],
             'v_energy_per_atom': [-5.1848775, -5.1850072]})
        self.assertTrue(np.all(
            simulation_log[0].runs[2].data[:2] == expected_run_data))

    def test_relevant_lammps_commands(self):
        '''
        Does it find all the relevant commands preceding the run which affect
        the output?
        '''
        output_file = 'test_files/test_output_parser_log_file.lammps'
        with open(output_file, 'r') as f:
            output_string = f.read()
        pattern = re.compile(r"(unfix.+|fix.+|minimize.+)")
        commands = list(pattern.finditer(output_string))
        pattern = re.compile(r"Step.+")
        headers = list(pattern.finditer(output_string))
        pattern = re.compile(r"Loop time.+")
        loop_times = list(pattern.finditer(output_string))
        output = list(zip(headers, loop_times))
        run_commands = output_parser.relevant_lammps_commands(
            commands, output[0])
        expected_commands = [
            'fix run all npt temp 1.0 1.0 $(100.0*dt) aniso 1.0 1.0 '
            + '$(1000.0*dt)', 'fix run all npt temp 1.0 1.0 '
            + '0.10000000000000000555 aniso 1.0 1.0 $(1000.0*dt)',
            'fix run all npt temp 1.0 1.0 0.10000000000000000555 aniso 1.0 '
            + '1.0 1']
        self.assertTrue(run_commands == expected_commands)
        run_commands = output_parser.relevant_lammps_commands(
            commands, output[2])
        expected_commands += ['minimize 0.0 0.0 10000 100000']
        self.assertTrue(run_commands == expected_commands)

    def test_format_lammps_output(self):
        '''
        Does output get correctly formatted into a pandas dataframe, and
        return a loop time?
        '''
        output_file = 'test_files/test_output_parser_log_file.lammps'
        with open(output_file, 'r') as f:
            output_string = f.read()
        pattern = re.compile(r"(unfix.+|fix.+|minimize.+)")
        pattern = re.compile(r"Step.+")
        headers = list(pattern.finditer(output_string))
        pattern = re.compile(r"Loop time.+")
        loop_times = list(pattern.finditer(output_string))
        output = list(zip(headers, loop_times))
        output_data, loop_time = output_parser.format_lammps_output(
            output[0], output_string)
        expected_output_data = pd.DataFrame(
            {'Step': [0.0, 1000.0], 'Temp': [0.0, 1.1448199],
             'E_pair': [-5146.8078, -5184.8572], 'E_mol': [0., 0.],
             'TotEng': [-5146.8078, -5184.7094],
             'Press': [6744.3059, 0.94514147],
             'v_num_grain_atoms': [1000., 1000.],
             'c_grain_energy': [-5146.8078, -5184.8572],
             'v_energy_per_atom': [-5.1468078, -5.1848572]})
        self.assertTrue(np.all(output_data[:2] == expected_output_data))
        self.assertTrue(loop_time == 7.70749)


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from analyse.output import output_parser
    unittest.main()
