import datetime
import os
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from configmanager.configmanager import ConfigManager
from grid2bench.EpisodeDataExtractor import EpisodeDataExtractor

conf_path = os.path.abspath("../conf.ini")
conf = ConfigManager(benchmark_name='Tests', path=conf_path)

data_path = os.path.abspath(os.path.join('..', conf.get_option('data_path')))
agent_name = conf.get_option_tolist('agents_names')[1]

agent_path = os.path.join(data_path, agent_name)
episode_name = conf.get_option_tolist('episodes_names')[0]


class TestEpisodeDataExtractor(unittest.TestCase):

  def setUp(self):
    self.agent_path = agent_path
    self.agent_name = agent_name
    self.episode_name = episode_name

    self.episode_data = EpisodeDataExtractor(self.agent_path, self.episode_name)

  def test_get_observation_by_timestamp(self):
    observation = self.episode_data.get_observation_by_timestamp(
      datetime.datetime(year=2019, month=1, day=6, hour=0, minute=20))

    self.assertEqual(observation, self.episode_data.observations[-2])

  def test_get_action_by_timestamp(self):
    action = self.episode_data.get_action_by_timestamp(
      datetime.datetime(year=2019, month=1, day=6, hour=0, minute=15))
    self.assertEqual(action, self.episode_data.actions[3])

  def test_get_computation_time_by_timestamp(self):
    computation_time = self.episode_data.get_computation_time_by_timestamp(
      datetime.datetime(year=2019, month=1, day=6, hour=0, minute=20))
    self.assertEqual(computation_time, self.episode_data.computation_times[4])

  def test_get_timestep_by_datetime(self):
    timestep = self.episode_data.get_timestep_by_datetime(
      datetime.datetime(year=2019, month=1, day=6, hour=0, minute=0))
    self.assertEqual(timestep, 0)

  def test_compute_actions_freq_by_timestamp(self):
    list_actions_freq = self.episode_data.actions_freq_by_timestamp()[
      'NB action']
    self.assertListEqual(list_actions_freq.to_list(), [1, 4, 1, 1])

  def test_compute_actions_freq_by_type(self):
    nb_switch_line = self.episode_data.actions_freq_by_type()[
      'NB line switched']
    nb_topological_changes = self.episode_data.actions_freq_by_type()[
      'NB topological change']
    nb_redispatch_changes = self.episode_data.actions_freq_by_type()[
      'NB redispatching']
    nb_storage_changes = self.episode_data.actions_freq_by_type()[
      'NB storage changes']
    nb_curtailment_changes = self.episode_data.actions_freq_by_type()[
      'NB curtailment']

    self.assertListEqual(nb_switch_line.to_list(), [0, 0, 0, 1])
    self.assertListEqual(nb_topological_changes.to_list(), [0, 4, 1, 0])
    self.assertListEqual(nb_redispatch_changes.to_list(), [1, 0, 0, 0])
    self.assertListEqual(nb_storage_changes.to_list(), [0, 0, 0, 0])
    self.assertListEqual(nb_curtailment_changes.to_list(), [0, 0, 0, 0])

  def test_compute_actions_freq_by_station(self):
    list_impacted_substations = self.episode_data.impacted_subs_by_timestamp()
    impacted_sub_stations = [list_impacted_substations[i]['subs_impacted'] for i
                             in range(len(list_impacted_substations))]
    self.assertListEqual(impacted_sub_stations, [{'sub_13'}, {'sub_12'}])

  def test_compute_overloaded_lines_by_timestamp(self):
    list_overloaded_lines = self.episode_data.overloaded_lines_by_timestamp()
    overloaded_lines = [list_overloaded_lines[i]['Overloaded lines'] for i in
                        range(len(list_overloaded_lines))]
    self.assertListEqual(overloaded_lines, [{5, 14}, {5, 14}, {8, 16}])

  def test_compute_disconnected_lines_by_timestamp(self):
    list_disconnected_lines = self.episode_data.disconnected_lines_by_timestamp()
    disconnected_lines = [list_disconnected_lines[i]['Disconnected lines'] for i
                          in range(len(list_disconnected_lines))]
    self.assertListEqual(disconnected_lines, [{5, 14, 15}])

  def test_compute_action_sequences_length(self):
    actions_sequence_length = \
      self.episode_data.compute_action_sequences_length()['Sequence length']

    self.assertListEqual(actions_sequence_length.to_list(), [1, 2, 3, 4])

  def test_create_topology_df(self):
    c, df1 = self.episode_data.create_topology_df()
    self.assertEqual(c, 9)
    df2 = pd.DataFrame({'t_step': [1, 2, 4], 'time_stamp': [
      datetime.datetime.strptime('2019-01-06 00:05:00', '%Y-%m-%d %H:%M:%S'),
      datetime.datetime.strptime('2019-01-06 00:10:00', '%Y-%m-%d %H:%M:%S'),
      datetime.datetime.strptime('2019-01-06 00:20:00', '%Y-%m-%d %H:%M:%S')],
                        'action_id': [1, 2, 4], 'susbtation': [5, 4, 4],
                        'episode': ['0', '0', '0']})
    assert_frame_equal(df1, df2, check_dtype=False)

  def test_create_injection_df(self):
    c, df1 = self.episode_data.create_injection_df()
    self.assertEqual(c, 0)
    df2 = pd.DataFrame(
      columns=['t_step', 'time_stamp', 'action_id', 'count', 'impacted'])

    assert_frame_equal(df1, df2, check_dtype=False)

  def test_create_dispatch_df(self):
    c, df1 = self.episode_data.create_dispatch_df()
    self.assertEqual(c, 1)
    df2 = pd.DataFrame({'t_step': [0], 'time_stamp': [
      datetime.datetime.strptime('2019-01-06 00:00:00', '%Y-%m-%d %H:%M:%S')],
      'action_id': [0], 'generator_id': [0], 'generator_name': ['gen_1_0'],
      'amount': [-3.75]})
    assert_frame_equal(df1, df2, check_dtype=False)

  def test_create_force_line_df(self):
    c, df1 = self.episode_data.create_force_line_df()
    self.assertEqual(c, 0)
    df2 = pd.DataFrame(
      columns=['t_step', 'time_stamp', 'action_id', 'type', 'powerline'])
    assert_frame_equal(df1, df2, check_dtype=False)

  def test_create_curtailment_df(self):
    c, df1 = self.episode_data.create_curtailment_df()
    self.assertEqual(c, 0)
    df2 = pd.DataFrame(columns=['t_step', 'time_stamp', 'action_id', 'limit'])
    assert_frame_equal(df1, df2, check_dtype=False)
  def test_create_storage_df(self):
    c, df1 = self.episode_data.create_storage_df()
    self.assertEqual(c, 0)
    df2 = pd.DataFrame(columns=['t_step', 'time_stamp', 'action_id',
                                'capacities'])
    assert_frame_equal(df1, df2, check_dtype=False)

