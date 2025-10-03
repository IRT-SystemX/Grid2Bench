# Copyright (C) 2022,
# IRT-SystemX (https://www.irt-systemx.fr), RTE (https://www.rte-france.com)
# See authors in pyporject.toml
# SPDX-License-Identifier: MPL-2.0
"""This module is used to transform data already extracted using the
:class:`EpisodeDataExtractor` class, and to prepare it for visualization.

The module is also used to store several episode information for later use,
such as metrics evaluation

  Typical usage example of the module:

  .. code-block:: python

    from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

    agent_log_path = '../data/input/Expert_Agent'
    episode_names = ['dec16_1', 'dec16_2']

    # loading episodes data
    expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

    # sequence actions for all loaded episodes
    df1 = expert_agent.actions_freq_by_type_several_episodes()
    print('Frequency of actions by type for all episodes \\n',df1)

    # sequence actions for only selected episodes
    episodes_names = ['dec16_1']
    df2 = expert_agent.actions_freq_by_type_several_episodes(
          episodes_names=['dec16_1'])
    print('Frequency of actions by type for dec16_1 episode \\n', df2)

"""
import itertools
import os
from collections import Counter
from datetime import datetime
from typing import List, Optional, Dict

import ipywidgets as widgets
import numpy as np
import pandas as pd
import qgrid
from IPython.display import display
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from tqdm import tqdm
import plotly.express as px

from grid2bench.EpisodeDataExtractor import EpisodeDataExtractor


class EpisodesDataTransformer:
  """This class is used to load multiple episode data at once and then
  prepare it for analysis based to specific metrics.

  Attributes:

    - agent_path: agent's log file path
    - agent_name: agent's name, as the same name as the agent folder
    - episodes_names: list of episode names with the same name as the episode
      data folder
    - episodes_data: list of :class:`EpisodeDataExtractor`
    - n_lines: number lines in the power grid
    - n_episode: number of episodes loaded in episodes_data

  """

  def __init__(self, agent_path: str, episodes_names: list):
    """Init and loading data for the class.

    :param agent_path: agent log file path
    :type agent_path: str
    :param episodes_names: list of episode names to load, it must be the same
     name as the episode folders
    :type episodes_names: list of str
    """
    self.agent_path = agent_path
    self.episodes_names = episodes_names
    self.agent_name = os.path.basename(agent_path)

    self.episodes_data = self.load_all_episodes()
    self.n_lines = self.episodes_data[0]._n_lines()
    self.n_episode = len(self.episodes_names)

  def load_all_episodes(self) -> List[EpisodeDataExtractor]:
    """Load episodes data from the episodes_names list.

    :return: list of EpisodeDataExtractor
    :rtype: list of EpisodeDataExtractor
    """
    if not self.episodes_names:
      self.episodes_names = [name for name in os.listdir(self.agent_path) if
                             os.path.isdir(os.path.join(self.agent_path, name))]

    return [EpisodeDataExtractor(self.agent_path, episode_name) for episode_name
            in tqdm(self.episodes_names)]

  def get_action_by_timestamp(self, timestamp_str: str) -> BaseAction:
    """Get the grid2op action object whose timestamp is date_time.

    :param timestamp_str: the datetime sought '%Y-%m-%d %H:%M:%S'
    :type timestamp_str: str
    :return: Action if exists otherwise raise Error
    :rtype: grid2op.BaseAction object

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      str_date_time = '2012-12-04 07:10:00'
      act = expert_agent.get_action_by_timestamp(str_date_time)
      print(act)

    """
    for episode in self.episodes_data:
      timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
      if timestamp in episode.timestamps:
        return episode.get_action_by_timestamp(timestamp)

  def get_observation_by_timestamp(self, episode_name: str,
                                   timestamp_str: str) -> BaseObservation:
    """Get the grid2op Observation object whose timestamp is date_time.

    :param timestamp_str: the datetime sought '%Y-%m-%d %H:%M:%S'
    :type timestamp_str: str
    :return: Observation if exists otherwise raise Error
    :rtype: grid2op.BaseObservation object

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      str_date_time = '2012-12-04 07:10:00'
      obs = expert_agent.get_observation_by_timestamp(str_date_time)
      print(obs)

    """
    idx = self.episodes_names.index(episode_name)
    episode = self.episodes_data[idx]
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

    return episode.get_observation_by_timestamp(timestamp)




  def actions_freq_by_type_several_episodes(self, episodes_names: Optional[
    List] = None) -> pd.DataFrame:
    """Returns the number of unit actions by type for all episodes in
    episodes_names

    If episodes_names=None, then returns the results of all loaded episodes.

    Unit actions including switched lines, topological impacts, redispatching,
    storage and curtailment.

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :return: Actions frequency of : NB line switched, NB topological change,
             NB redispatching, NB storage changes, NB curtailment
    :rtype: DataFrame

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # actions frequency by type for all loaded episodes
      df1 = expert_agent.actions_freq_by_type_several_episodes()
      print('Frequency of actions by type for all episodes \\n',df1)

      # actions frequency for only selected episodes
      episodes_names = ['dec16_1']
      df2 = expert_agent.actions_freq_by_type_several_episodes(
                episodes_names=['dec16_1'])
      print('Frequency of actions by type for dec16_1 episode \\n', df2)

    """
    if not episodes_names: episodes_names = self.episodes_names

    df = pd.DataFrame()
    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        df = pd.concat([df, episode_data.actions_freq_by_type()], axis=0)

    df_sum = pd.DataFrame({'Frequency': df.sum(axis=0)})

    return df_sum

  def actions_freq_by_station_several_episodes(self, episodes_names: Optional[
    List] = None) -> pd.DataFrame:
    """Returns the impacted substations at each timestamp for all episodes in
    episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :return: impacted substations by agent actions
    :rtype: a list of dictionaries {Timestamp, Sub impacted}

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # all loaded episodes
      df1 = expert_agent.actions_freq_by_station_several_episodes()
      print('Frequency of actions by station for all episodes \\n',df1)

      # selected episodes
      episodes_names = ['dec16_1']
      df2 = expert_agent.actions_freq_by_station_several_episodes(
                episodes_names=['dec16_1'])
      print('Frequency of actions by station for dec16_1 episode \\n', df2)

    """

    if not episodes_names: episodes_names = self.episodes_names

    actions_freq_by_station = []
    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        actions_freq_by_station.extend(
          episode_data.impacted_subs_by_timestamp())

    impacted_stations_flatten = []

    for item in actions_freq_by_station:
      impacted_stations_flatten.extend(list(item['subs_impacted']))

    x = list(Counter(impacted_stations_flatten).keys())
    y = list(Counter(impacted_stations_flatten).values())
    df = pd.DataFrame(data=np.array([x, y]).transpose(),
                      columns=['Substation', 'Frequency'])
    df = df.astype({'Frequency': int}, errors='raise')

    return df

  def computation_times_several_episodes(self, episodes_names: Optional[
    List] = None) -> pd.DataFrame:
    """Action execution times at each timestamp for all episodes in
    episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :return: a dataframe of execution times (s) at each timestamp
    :rtype: DataFrame

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # all loaded episodes
      df1 = expert_agent.computation_times_several_episodes()
      print('Actions execution times for all episodes \\n',df1)

      # filter selected episodes, if episodes_names == default value then
      # return results from all loaded episodes
      episodes_names = ['dec16_1']
      df2 = expert_agent.computation_times_several_episodes(
                episodes_names=['dec16_1'])
      print('Actions execution times for dec16_1 episode \\n', df2)

    """

    executions_time = pd.DataFrame(columns=['Timestamp', 'Execution time'])

    if not episodes_names: episodes_names = self.episodes_names

    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        df = pd.DataFrame(np.column_stack((episode_data.timestamps,
                                           episode_data.computation_times[
                                           :episode_data.n_action])),
                          columns=executions_time.columns)

        executions_time = pd.concat([executions_time, df], ignore_index=True)

    executions_time = executions_time.sort_values(by='Timestamp',
                                                  ascending=True)

    return executions_time

  def distance_from_initial_topology(self, episodes_names: Optional[
    List] = None) -> pd.DataFrame:
    """Number of changes compared to initial topology at each timestamp
    for all episodes in episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :return: distance from initial topology at each timestamp
    :rtype: DataFrame

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # filter selected episodes, if episodes_names == default value then
      # return results from all loaded episodes
      episodes_names = ['dec16_1']
      df = expert_agent.distance_from_initial_topology(
                episodes_names=['dec16_1'])
      print('Distance from initial topology for dec16_1 episode \\n', df)

    """

    if not episodes_names: episodes_names = self.episodes_names

    x = []
    y = []

    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        for j in range(episode_data.n_action):
          act = episode_data.actions[j]
          obs = episode_data.observations[j]

          line_statuses = episode_data.observations[j].line_status
          # True == sub has something on bus 2, False == everything on bus 1
          # Distance for subs == subs_on_bus2.sum()
          subs_on_bus_2 = np.repeat(False, episode_data.observations[j].n_sub)
          # objs_on_bus_2 will store the id of objects connected to bus 2
          objs_on_bus_2 = {id: [] for id in
                           range(episode_data.observations[j].n_sub)}
          distance, _, _, _ = episode_data.get_distance_from_obs(act,
                                                                 line_statuses,
                                                                 subs_on_bus_2,
                                                                 objs_on_bus_2,
                                                                 obs)

          x.append(episode_data.timestamps[j])
          y.append(distance)

    df_distance = pd.DataFrame(data=np.array([x, y]).transpose(),
                               columns=['Timestamp', 'Distance'])

    df_distance = df_distance.sort_values(by='Timestamp', ascending=True)
    df_distance = df_distance.astype({'Distance': int}, errors='raise')

    return df_distance

  def get_sequence_actions(self, episodes_names: Optional[
    List] = None) -> pd.DataFrame.groupby:
    """Return sequence actions as grouped dataframe for all episodes in
    episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    A sequence of actions is defined by taking several actions in a
    non-disconnected sequence of time steps. A non-disconnected sequence means
    that there is no "do nothing" in the sequence.

    Examples:

    - Sequence 1 : Switch bus action, Topological changes, Do nothing
      --> Sequence of length 2
    - Sequence 2 : Switch bus action, Do nothing, Topological changes
      --> not a sequence
    - Sequence 3 : Switch bus action, Topological changes, Do nothing, Switch
      bus action, Topological changes, Topological changes, Do nothing -->
      two sequences, the first of length 2 and the second of length 3

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :return: a DataFrame time series with columns: Timestamp, Sequence
              length, NB action, Impacted subs, Impacted lines
    :rtype: DataFrame

    Example of usage :

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # filter selected episodes
      episodes_names = ['dec16_1']
      df = expert_agent.get_sequence_actions(
                episodes_names=['dec16_1'])
      print('First sequence of actions in dec16_1 episode \\n', df.first())

    """
    i = 0
    incremental_id = 0

    if not episodes_names: episodes_names = self.episodes_names

    df_sequence_length = pd.DataFrame()
    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        df = episode_data.compute_action_sequences_length()
        df_sequence_length = pd.concat([df_sequence_length, df], axis=0,
                                       ignore_index=True)

    df_sequence_length.insert(0, 'Sequence ID',
                              -1 * np.ones(df_sequence_length.shape[0], int))

    while i + 1 < df_sequence_length.shape[0]:

      if df_sequence_length.loc[i + 1, 'Sequence length'] == 1:
        df_sequence_length.iloc[i, 0] = incremental_id
        i = i + 1

      else:
        j = 2
        while (i + j) < df_sequence_length.shape[0] and df_sequence_length.loc[
          i + j, 'Sequence length'] != 1: j = j + 1

        df_sequence_length.loc[i:i + j, 'Sequence ID'] = incremental_id
        i = i + j

      incremental_id = incremental_id + 1

    df_sequence_length.loc[
      df_sequence_length['Sequence ID'] == -1, 'Sequence ID'] = incremental_id

    grp = df_sequence_length.groupby('Sequence ID')

    return grp

  def display_sequence_actions(self, episodes_names: Optional[List] = None,
                               min_length: int = 0,
                               max_length: int = 400) -> pd.DataFrame:
    """Return sequence of actions as a flattened list for all episodes in
    episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    A sequence of actions is defined by taking several actions in a
    non-disconnected sequence of time steps. A non-disconnected sequence means
    that there is no "do nothing" in the sequence.

    Examples:

    - Sequence 1 : Switch bus action, Topological changes, Do nothing
      --> Sequence of length 2
    - Sequence 2 : Switch bus action, Do nothing, Topological changes
      --> not a sequence
    - Sequence 3 : Switch bus action, Topological changes, Do nothing, Switch
      bus action, Topological changes, Topological changes, Do nothing -->
      two sequences, the first of length 2 and the second of length 3

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :param min_length: minimum length of sequence
    :type min_length: int
    :param max_length: maximum length of sequence
    :type max_length: int
    :return: a DataFrame with columns: Timestamp, Sequence start, Sequence end,
     Sequence length, NB actions, NB unitary actions, Impacted Subs,
     Impacted lines
    :rtype: DataFrame

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # filter selected episodes
      episodes_names = ['dec16_1']
      df = expert_agent.display_sequence_actions(
        episodes_names=['dec16_1'],
        min_length= 0,
        max_length=15)
      print('Sequence actions in dec16_1 episode \\n', df)

    """

    grp = self.get_sequence_actions(episodes_names=episodes_names)

    list_dict = []
    columns = ['Timestamp', 'Sequence start', 'Sequence end', 'Sequence length',
               'NB actions', 'NB unitary actions', 'Impacted Subs',
               'Impacted lines']

    for _, group in grp:

      if min_length <= group.shape[0] < max_length:
        start_timestamp = group.iloc[0, 1]
        end_timestamp = group.iloc[group.shape[0] - 1, 1]
        sequence_length = group.shape[0]

        # count total number of impacted subs
        nb_impacted_subs = len(
          [item for sublist in group['Impacted subs'] for item in sublist])
        # count total number of impacted lines
        nb_impacted_lines = len(
          [item for sublist in group['Impacted lines'] for item in sublist])
        # number of total actions
        nb_actions = nb_impacted_subs + nb_impacted_lines

        nb_unitary_actions = group['NB action'].sum()

        impacted_subs = group['Impacted subs'].tolist()
        impacted_lines = group['Impacted lines'].tolist()

        dict_row = {'Timestamp': group['Timestamp'].tolist(),
                    'Sequence start': start_timestamp,
                    'Sequence end': end_timestamp,
                    'Sequence length': sequence_length,
                    'NB actions': nb_actions,
                    'NB unitary actions': nb_unitary_actions,
                    'Impacted Subs': impacted_subs,
                    'Impacted lines': impacted_lines}
        list_dict.append(dict_row)

    return pd.DataFrame(list_dict, columns=columns)

  def display_sequence_actions_filter(self,
                                      episodes_names: Optional[List] = None,
                                      min_length: int = 0,
                                      max_length: int = 400):
    """Display sequence actions as separate dataframes for all episodes in
    episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    A sequence of actions is defined by taking several actions in a
    non-disconnected sequence of time steps. A non-disconnected sequence means
    that there is no "do nothing" in the sequence.

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :param min_length: minimum length of sequence
    :type min_length: int
    :param max_length: maximum length of sequence
    :type max_length: int
    :return: a DataFrame time series with columns: Timestamp, Sequence
              length, NB unitary actions, Impacted subs, Impacted lines
    :rtype: DataFrame

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # filter selected episodes
      episodes_names = ['dec16_1']
      expert_agent.display_sequence_actions_filter(
        episodes_names=['dec16_1'],
        min_length=1,
        max_length=15)

    """

    grp = self.get_sequence_actions(episodes_names)

    for _, group in grp:
      if min_length <= group.shape[0] < max_length:
        df = group.drop(['Sequence ID', 'Sequence length'], axis=1)
        df = df.rename(columns={'NB action': 'NB unitary actions'})
        display(df)

  def action_sequences_to_dict(self, episodes_names: Optional[List] = None,
                               min_length: int = 0, max_length: int = 400) -> \
      List[Dict]:
    """Helper function to transform sequence of actions df to dict.

    Transform the data structure and prepare it for plotting with plotly's
    gantt chart

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :param min_length: minimum length of sequence
    :type min_length: int
    :param max_length: maximum length of sequence
    :type max_length: int
    :return: a dictionary list of : Type, Start, Finish, Actions,
    Actions_percent.
    :rtype: dictionary list
    """

    df = self.display_sequence_actions(episodes_names, min_length, max_length)
    max_length = df['Sequence length'].max()
    dict_list = []

    for i in range(df.shape[0]):
      dict_list.append(
        dict(Type=self.agent_name, Start=str(df.loc[i, 'Sequence start']),
             Finish=str(df.loc[i, 'Sequence end']),
             Actions=df.loc[i, 'Sequence length'], Actions_percent=(df.loc[
                                                                      i, 'NB unitary actions'] / max_length) * 100, ))

    return dict_list

  def overloaded_lines_freq_several_episodes(self, episodes_names: Optional[
    List] = None) -> pd.DataFrame:
    """Overloaded lines for all episodes in episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :return: frequency of overloaded lines
    :rtype: Dataframe

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # filter selected episodes
      episodes_names = ['dec16_1']
      df = expert_agent.overloaded_lines_freq_several_episodes(
        episodes_names=['dec16_1'])
      print(df)

    """

    if not episodes_names: episodes_names = self.episodes_names

    overloaded_lines = []

    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        overloaded_lines_for_episode_i = episode_data.overloaded_lines_by_timestamp()

        overloaded_lines_flatten = [list(item) for dict in
                                    overloaded_lines_for_episode_i for key, item
                                    in dict.items() if
                                    key == 'Overloaded lines']

        overloaded_lines = overloaded_lines + overloaded_lines_flatten

    overloaded_lines_flatten = list(itertools.chain(*overloaded_lines))

    data = [overloaded_lines_flatten.count(x) for x in
            range(self.episodes_data[0]._n_lines())]

    df = pd.DataFrame(np.array(
      [self.episodes_data[0]._name_of_lines(range(len(data))),
       data]).transpose(), columns=['Line', 'Overloaded'])

    df = df.astype({'Line': str, 'Overloaded': int}, errors='raise')
    df = df.loc[df['Overloaded'] != 0]

    return df

  def disconnected_lines_freq_several_episodes(self, episodes_names: Optional[
    List] = None) -> pd.DataFrame:
    """Disconnected lines for all episodes in episodes_names.

    If episodes_names=None, then returns the results of all loaded episodes.

    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :return: frequency of disconnected lines
    :rtype: Dataframe

    Example of usage:

    .. code-block:: python

      from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer

      agent_log_path = '../data/input/Expert_Agent'
      episode_names = ['dec16_1', 'dec16_2']

      # loading episodes data
      expert_agent = EpisodesDataTransformer(agent_log_path, episode_names)

      # filter selected episodes
      episodes_names = ['dec16_1']
      df = expert_agent.disconnected_lines_freq_several_episodes(
        episodes_names=['dec16_1'])
      print(df)
    """

    if not episodes_names: episodes_names = self.episodes_names

    disconnected_lines = []

    for episode_data in self.episodes_data:

      if episode_data.episode_name in episodes_names:
        disconnected_lines_for_episode_i = episode_data.disconnected_lines_by_timestamp()
        disconnected_lines_flatten = [list(item) for dict in
                                      disconnected_lines_for_episode_i for
                                      key, item in dict.items() if
                                      key == 'Disconnected lines']

        disconnected_lines = disconnected_lines + disconnected_lines_flatten

    disconnected_lines_flatten = list(itertools.chain(*disconnected_lines))

    data = [disconnected_lines_flatten.count(x) for x in
            range(self.episodes_data[0]._n_lines())]

    df = pd.DataFrame(np.array(
      [self.episodes_data[0]._name_of_lines(range(len(data))),
       data]).transpose(), columns=['Line', 'Disconnected'])

    df = df.astype({'Line': str, 'Disconnected': int}, errors='raise')
    df = df.loc[df['Disconnected'] != 0]

    return df

  # TODO docstings --> Fereshteh

  def display_detailed_action_type(self, episodes_names: List):

    data_display = display(display_id="data_display")
    output_display = display(display_id="output_display")
    grid = qgrid.QGridWidget(df=pd.DataFrame())

    w = widgets.Dropdown(
      options=['Select', 'Tolopology', 'Force_line', 'Redispatching',
               'Injection', 'Curtailment', 'Storage'], value='Select',
      description='Table', )

    def on_change(change):
      if change['type'] == 'change' and change['name'] == 'value':

        result = pd.DataFrame()
        c = 0
        for episode_data in self.episodes_data:
          if (not len(
              episodes_names)) or episode_data.episode_name in episodes_names:
            functions = {'Tolopology': episode_data.create_topology_df,
                         'Force_line': episode_data.create_force_line_df,
                         'Redispatching': episode_data.create_dispatch_df,
                         'Injection': episode_data.create_injection_df,
                         'Curtailment': episode_data.create_curtailment_df,
                         'Storage': episode_data.create_storage_df}
            r = functions[change['new']]()
            r[1]['episode_name'] = episode_data.episode_name
            result = pd.concat([result, r[1]])
            c += r[0]
        output_display.update(
          "total Number of " + change['new'] + " changes:" + str(c))
        grid.df = result

    w.observe(on_change)
    #         ouptup_display = display(display_id="ouptup_display")

    display(w)
    output_display.display('')

    data_display.display(grid)

  def get_actions_by_substation_by_id(self):
    act_episodes = []
    print('Calculating actions id')
    for i, episode_data in tqdm(enumerate(self.episodes_data)):
      act_episodes.extend(episode_data.actions.objects)
      if i == 0:
        _, df = episode_data.create_topology_df()
      else:
        _, df2 = episode_data.create_topology_df()
        df = pd.concat([df, df2], ignore_index=True)

    def f(x):
      episode_name = x['episode']
      episode_idx = self.episodes_names.index(episode_name)
      timestep = x['t_step']
      action = self.episodes_data[episode_idx].actions[timestep]
      return act_episodes.index(action)

    df['action_id'] = df.apply(f, axis=1)

    df['nb_action'] = np.ones(df.shape[0])
    df['susbtation'] = df['susbtation'].apply(lambda x: f'sub_{x}')
    df['action_id'] = df['action_id'].apply(lambda x: f'act_{x}')

    return df

  def get_action_by_id(self, action_id):
    act_episodes = []
    for i, episode_data in enumerate(self.episodes_data):
      act_episodes.extend(episode_data.actions.objects)
    return act_episodes[action_id]

  @staticmethod
  def plot_actions_by_station_by_id(df, title: Optional[
    str] = 'Frequency of actions by substation', **fig_kwargs):

    fig = px.sunburst(df, path=['susbtation', 'action_id'], values='nb_action',
                      title=title, )
    fig.update_layout(**fig_kwargs)
    return fig
