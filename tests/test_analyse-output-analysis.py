import unittest
from collections import namedtuple
import os
import sys
import pandas as pd


class TestAnalysis(unittest.TestCase):

    def test_single_line_plot(self):
        '''
        Does a dataplot get created correctly?
        '''
        TestLog = namedtuple('TestLog', ['name', 'runs'])
        TestRun = namedtuple('TestRun', ['data'])
        test_data = pd.DataFrame({'Steps': [1, 2, 3, 5],
                                  'Value': [1.2, 3.5, 4.5, 2]})
        test_log = TestLog('Test', [TestRun(test_data)])
        analysis.single_line_plot([test_log], 'Steps', 'Value')
        self.assertTrue('No Title.png' in os.listdir())
        os.remove('No Title.png')
        title = 'This is a test'
        analysis.single_line_plot([test_log], 'Steps', 'Value', title=title)
        self.assertTrue('This is a test.png' in os.listdir())
        os.remove('This is a test.png')
        test_data_2 = pd.DataFrame({'Steps': [1, 2, 3, 5],
                                    'Value': [0.2, 3.9, 1.5, 4]})
        test_log = [TestLog('Test1', [TestRun(test_data)]),
                    TestLog('Test2', [TestRun(test_data_2)])]
        analysis.single_line_plot(test_log, 'Steps', 'Value')
        self.assertTrue('No Title.png' in os.listdir())
        os.remove('No Title.png')

    def test_multiple_line_plots(self):
        '''
        Does a dataplot get created correctly?
        '''
        TestLog = namedtuple('TestLog', ['name', 'runs'])
        TestRun = namedtuple('TestRun', ['data'])
        test_data = pd.DataFrame({'Steps': [1, 2, 3, 5],
                                  'Value': [1.2, 3.5, 4.5, 2]})
        test_data_2 = pd.DataFrame({'Steps': [1, 2, 3, 5],
                                    'Value': [0.2, 3.9, 1.5, 4]})
        test_log = [TestLog('Test1', [TestRun(test_data)]),
                    TestLog('Test2', [TestRun(test_data_2)])]
        analysis.multiple_line_plots(test_log, 'Steps', 'Value')
        self.assertTrue('Test1.png' in os.listdir())
        self.assertTrue('Test2.png' in os.listdir())
        os.remove('Test1.png')
        os.remove('Test2.png')

    def test_select_logs(self):
        '''
        Are the correct logs selected from the log list?
        '''
        TestLog = namedtuple('TestLog', ['name'])
        test_names = [TestLog('this is a test'),
                      TestLog('not a test'),
                      TestLog("this isn't a test"),
                      TestLog('this is blah blah test')]
        string_pattern = r"this is .+ test"
        new_names = analysis.select_logs(test_names, string_pattern)
        expected_names = [TestLog('this is a test'),
                          TestLog('this is blah blah test')]
        self.assertTrue(new_names == expected_names)
        new_names = analysis.select_logs(test_names, None)
        expected_names = [TestLog('this is a test'),
                          TestLog('not a test'),
                          TestLog("this isn't a test"),
                          TestLog('this is blah blah test')]
        self.assertTrue(new_names == expected_names)

    def test_get_log_data(self):
        '''
        Does the function return the correct data from the pandas dataframe?
        '''
        TestLog = namedtuple('TestLog', ['name', 'runs'])
        TestRun = namedtuple('TestRun', ['data'])
        test_data = pd.DataFrame({'Steps': [1, 2, 3, 5],
                                  'Value': [1.2, 3.5, 4.5, 2]})
        test_log = TestLog('Test', [TestRun(test_data)])
        retrieved_data = analysis.get_log_data(test_log, 0, 'Steps', ':')
        expected_data = [1, 2, 3, 5]
        self.assertTrue(retrieved_data.tolist() == expected_data)
        retrieved_data = analysis.get_log_data(test_log, 0, 'Value', ':-2')
        expected_data = [1.2, 3.5]
        self.assertTrue(retrieved_data.tolist() == expected_data)
        retrieved_data = analysis.get_log_data(test_log, 0, 'Value', -1)
        expected_data = 2
        self.assertTrue(retrieved_data == expected_data)
        self.assertRaisesRegex(
            ValueError, "Run choice 4 is out of range.",
            analysis.get_log_data, test_log, 4, 'Wrong', ':-2')
        self.assertRaisesRegex(
            ValueError, "Column 'Wrong' not in run data.",
            analysis.get_log_data, test_log, 0, 'Wrong', ':-2')


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from analyse.output import analysis
    unittest.main()
