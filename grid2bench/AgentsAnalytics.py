# Copyright (C) 2022,
# IRT-SystemX (https://www.irt-systemx.fr), RTE (https://www.rte-france.com)
# See authors in pyporject.toml
# SPDX-License-Identifier: MPL-2.0
"""This module allows to benchmark several agents in several scenarios
through predefined KPIs (see KPI section).

The module uses the `plotly package <https://plotly.com/>`_ to visualize
these KPIs in the form of graphs.

The modules load data in the forms :class:`EpisodesDataTransformer` and
:class:`EpisodesDataExtractor` . You can access these objects like this:

  .. code-block:: python

    from grid2bench.AgentsAnalytics import AgentsAnalytics

    input_data_path = os.path.abspath('../data/input')
    agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
    episodes_names = ['dec16_1', 'dec16_2']

    # loading data
    agents = AgentsAnalytics(
      data_path=input_data_path,
      agents_names=agents_names,
      episodes_names=episodes_names
    )

    # get indexes
    expert_agent_idx = agents.agents_names.index('Expert_Agent')
    dec16_2_idx = agents.episodes_names.index('dec16_2')

    # access expert agent all episodes results --> EpisodesDataTransformer class
    episodes = agents.agents_data[expert_agent_idx]
    print('Loaded episodes for {} are : {}'.format(
      episodes.agent_name,
      episodes.episodes_names))

    # access expert agent dec16_2 episode results --> EpisodeDataExtractor class
    dec16_2 = agents.agents_data[expert_agent_idx].episodes_data[dec16_2_idx]
    print('Episode name: {}'.format(dec16_2.episode_name))

"""
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import grid2op
from grid2op.PlotGrid import PlotMatplot

from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer


class AgentsAnalytics:
  """This class is used to benchmark several agents in several scenarios.

  It is also used to store agents' episode results via the agents_data attribute

  Attributes:

    - data_path: data folder path, parent directory path for agent log files
    - agents_names: list of agents names with the same name as the agent's
      data folders
    - episodes_names: list of episode names with the same name as the episode
      data folders
    - agents_data: list of :class:`EpisodesDataTransformer`

  """

  def __init__(self, data_path: str, agents_names: Optional[List] = None,
               episodes_names: Optional[List] = None):
    """Init and loading data for the class.

    :param data_path: parent directory path for agent log files
    :param agents_names: a list of each agent repository name,
                         if empty (not recommended) it will load all
                         repositories in the data path
    :param episodes_names: list of episode names with the same name as the
                           episode data folder
    """
    self.data_path = data_path
    self.agents_names = agents_names
    self.episodes_names = episodes_names

    self.agents_data = self.load_agents_results()

  def load_agents_results(self) -> List[EpisodesDataTransformer]:
    """Load agents' episodes data from the episodes_names list.

    Each element of the list contains agent's episodes data as
    :class:`EpisodesDataTransformer`

    :return: list of agents' episodes
    :rtype: list of :class:`EpisodesDataTransformer`
    """
    if not self.agents_names:
      self.agents_names = [name for name in os.listdir(self.data_path) if
                           os.path.isdir(os.path.join(self.data_path, name))]

    if not self.episodes_names:
      self.episodes_names = [name for name in os.listdir(
        os.path.join(self.data_path, self.agents_names[0])) if os.path.isdir(
        os.path.join(self.data_path, self.agents_names[0], name))]

    return [EpisodesDataTransformer(
      agent_path=os.path.join(self.data_path, agent_name),
      episodes_names=self.episodes_names) for agent_name in self.agents_names]

  @staticmethod
  def plot_actions_freq_by_station(
      agents_results: List[EpisodesDataTransformer],
      episodes_names: Optional[List] = None,
      title: Optional[str] = 'Frequency of actions by station', **fig_kwargs):
    """A bar chart representing the number of actions impacting each station
    and for each agent.

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.

    :param agents_results: list of agents episodes log, each item is a list
           of episode logs for an agent.
    :type agents_results: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Frequency of actions by station'
    :type title: str
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: Barchart of frequency of impacted stations
    :rtype: Plotly figure

    Example of usage:

    .. code-block:: python

      import os
      from grid2bench.AgentsAnalytics import AgentsAnalytics

      input_data_path = os.path.abspath('../data/input')
      agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
      episodes_names = ['dec16_1', 'dec16_2']

      # loading data
      agents = AgentsAnalytics(
        data_path=input_data_path,
        agents_names=agents_names,
        episodes_names=episodes_names
      )

      agents_logs = agents.agents_data
      fig = AgentsAnalytics.plot_actions_freq_by_station(agents_logs)

      fig.show()
    """

    agent_names = []

    # for the first agent
    agent_names.append(agents_results[0].agent_name)
    df = agents_results[0].actions_freq_by_station_several_episodes(
      episodes_names)
    df = df.rename(columns={'Frequency': agents_results[0].agent_name})

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)

      df2 = agent.actions_freq_by_station_several_episodes(episodes_names)
      df2 = df2.rename(columns={'Frequency': agent.agent_name})

      df = df.join(df2.set_index('Substation'), on='Substation')

    newnames = {}
    y_list = []

    for i in range(len(agent_names)):
      newnames[f'wide_variable_{i}'] = agent_names[i]
      y_list.append(df[agent_names[i]].to_list())

    fig = px.bar(x=df['Substation'].to_list(), y=y_list, text_auto='.2s',
                 labels={'x': 'Station', 'value': 'Frequency'}, barmode='group',
                 title=title)

    fig.for_each_trace(
      lambda t: t.update(name=newnames[t.name], legendgroup=newnames[t.name],
                         hovertemplate=t.hovertemplate.replace(t.name, newnames[
                           t.name])))

    fig.update_traces(textfont_size=12, textangle=0, textposition='outside',
                      cliponaxis=False)

    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_actions_freq_by_type(agents_results: List[EpisodesDataTransformer],
                                episodes_names: Optional[List] = None,
                                title: Optional[str] = 'Frequency of actions '
                                                       'by type',
                                row: Optional[int] = 1, col: Optional[int] = 2,
                                **fig_kwargs):
    """A pie chart representing the frequency of unit actions by type.

    Unit actions can be of type : switched lines, topological impacts,
    redispatching, storage and curtailment.

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.


    :param agents_results: list of agents episodes log, each item is a list
           of episode logs for an agent.
    :type agents_results: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Frequency of actions by type'
    :type title: str
    :param row: number of rows in plotly subplot
    :type row: int
    :param col: number of columns in plotly subplot
    :type col: int
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: A pie chart of frequency of unit actions by type
    :rtype: plotly figure

    Example of usage:

    .. code-block:: python

      import os
      from grid2bench.AgentsAnalytics import AgentsAnalytics

      input_data_path = os.path.abspath('../data/input')
      agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
      episodes_names = ['dec16_1', 'dec16_2']

      # loading data
      agents = AgentsAnalytics(
        data_path=input_data_path,
        agents_names=agents_names,
        episodes_names=episodes_names
      )

      agents_logs = agents.agents_data
      fig = AgentsAnalytics.plot_actions_freq_by_type(agents_logs)

      fig.show()
    """

    agent_names = []
    for agent in agents_results:
      agent_names.append(agent.agent_name)

    list_i = []
    for i in range(row):
      list_j = []
      for j in range(col):
        list_j.append({'type': 'domain'})
      list_i.append(list_j)

    fig = make_subplots(row, col, specs=list_i, subplot_titles=agent_names)

    for i in range(row):
      for j in range(col):
        data = agents_results[i + j].actions_freq_by_type_several_episodes(
          episodes_names)
        fig.add_trace(go.Pie(labels=data.index, values=data['Frequency'],
                             name=agent_names[i + j]), i + 1, j + 1)

    fig.update_traces(textposition='inside')
    fig.update_layout(title_text=title, uniformtext_minsize=12,
                      uniformtext_mode='hide', **fig_kwargs)
    return fig

  @staticmethod
  def plot_actions_freq_by_station_pie_chart(
      agents_results: List[EpisodesDataTransformer],
      episodes_names: Optional[List] = None,
      title: str = 'Frequency of actions by station', row: int = 1,
      col: int = 2, **fig_kwargs):

    """A Pie chart representing the number of actions impacting each station
    and for each agent. Similar to plot_actions_freq_by_station, but returns
    a pie chart instead

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.

    :param agents_results: list of agents episodes log, each item is a list
           of episode logs for an agent.
    :type agents_results: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Frequency of actions by station'
    :type title: str
    :param row: number of rows in plotly subplot
    :type row: int
    :param col: number of columns in plotly subplot
    :type col: int
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: Barchart of frequency of impacted stations
    :rtype: Plotly figure

    Example of usage:

    .. code-block:: python

      import os
      from grid2bench.AgentsAnalytics import AgentsAnalytics

      input_data_path = os.path.abspath('../data/input')
      agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
      episodes_names = ['dec16_1', 'dec16_2']

      # loading data
      agents = AgentsAnalytics(
        data_path=input_data_path,
        agents_names=agents_names,
        episodes_names=episodes_names
      )

      agents_logs = agents.agents_data
      fig = AgentsAnalytics.plot_actions_freq_by_station_pie_chart(agents_logs)

      fig.show()
    """

    agent_names = []
    for agent in agents_results:
      agent_names.append(agent.agent_name)

    list_i = []
    for i in range(row):
      list_j = []
      for j in range(col):
        list_j.append({'type': 'domain'})
      list_i.append(list_j)

    fig = make_subplots(row, col, specs=list_i, subplot_titles=agent_names)

    for i in range(row):
      for j in range(col):
        data = agents_results[i + j].actions_freq_by_station_several_episodes(
          episodes_names)
        fig.add_trace(
          go.Pie(labels=data['Substation'], values=data['Frequency'],
                 name=agent_names[i + j]), i + 1, j + 1)

    fig.update_traces(textposition='inside')
    fig.update_layout(title_text=title, uniformtext_minsize=12,
                      uniformtext_mode='hide', **fig_kwargs)
    return fig

  @staticmethod
  def plot_lines_impact(agents_results: List[EpisodesDataTransformer],
                        episodes_names: Optional[List] = None,
                        title: str = 'Overloaded Lines by station',
                        overloaded: bool = True, **fig_kwargs):
    """A bar chart representing the frequency of actions impacts on each power
    lines and for each agent.

    Two possible type of impacts : Line overflow impact, default value or
    fig_type ='disconnected'. And line overflow impact, can be obtained by
    giving fig_type ='disconnected'.

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.

    :param agents_results: list of agents episodes log, each item is a list
                           of episode logs for an agent.
    :type agents_results: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
                           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Overloaded Lines by station'
    :type title: str
    :param overloaded: diagram type: frequency of overloaded lines or
                     frequency of disconnected lines. Default = overloaded,
                     else disconnected
    :type fig_type: bool
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: Barchart of frequency of impacted lines
    :rtype: plotly figure

    Example of usage:

    .. code-block:: python

      import os
      from grid2bench.AgentsAnalytics import AgentsAnalytics

      input_data_path = os.path.abspath('../data/input')
      agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
      episodes_names = ['dec16_1', 'dec16_2']

      # loading data
      agents = AgentsAnalytics(
        data_path=input_data_path,
        agents_names=agents_names,
        episodes_names=episodes_names
      )

      agents_logs = agents.agents_data

      # overflowed lines
      fig_overflowed_lines = AgentsAnalytics.plot_lines_impact(agents_logs)
      fig_overflowed_lines.show()

      # disconnected lines
      fig_disconnected_lines = AgentsAnalytics.plot_lines_impact(
      agents_logs,
      overloaded= False)
      fig_disconnected_lines.show()

    """
    agent_names = []

    # for the first agent
    agent_names.append(agents_results[0].agent_name)

    if overloaded:
      df = agents_results[0].overloaded_lines_freq_several_episodes(
        episodes_names)
      df = df.rename(columns={'Overloaded': agents_results[0].agent_name})
    else:
      df = agents_results[0].disconnected_lines_freq_several_episodes(
        episodes_names)
      df = df.rename(columns={'Disconnected': agents_results[0].agent_name})
      title = 'Disconnected Lines by station'

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)
      if overloaded:
        df2 = agent.overloaded_lines_freq_several_episodes(episodes_names)
        df2 = df2.rename(columns={'Overloaded': agent.agent_name})
      else:
        df2 = agent.disconnected_lines_freq_several_episodes(episodes_names)
        df2 = df2.rename(columns={'Disconnected': agent.agent_name})

      df = df.join(df2.set_index('Line'), on='Line')

    newnames = {}
    y_list = []
    for i in range(len(agent_names)):
      newnames[f'wide_variable_{i}'] = agent_names[i]
      y_list.append(df[agent_names[i]].to_list())

    fig = px.bar(df, x='Line', y=y_list, text_auto='.2s',
                 labels={'x': 'Line', 'value': 'Frequency'}, barmode='group',
                 title=title, log_y=True, )

    fig.for_each_trace(
      lambda t: t.update(name=newnames[t.name], legendgroup=newnames[t.name],
                         hovertemplate=t.hovertemplate.replace(t.name, newnames[
                           t.name])))

    fig.update_traces(textfont_size=12, textangle=0, textposition='outside',
                      cliponaxis=False)

    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_computation_times(agents_results: List[EpisodesDataTransformer],
                             episodes_names: Optional[List] = None,
                             title: str = 'Action Execution Time',
                             **fig_kwargs):
    """Time series line graph representing actions execution time at each
    timestamp and for each agent.

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.

    :param agents_results: list of agents episodes log, each item is a list
                           of episode logs for an agent.
    :type agents_results: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
                           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Action Execution Time'
    :type title: str
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: Time series line graph of actions execution times
    :rtype: plotly figure

    Example of usage:

    .. code-block:: python

      import os
      from grid2bench.AgentsAnalytics import AgentsAnalytics

      input_data_path = os.path.abspath('../data/input')
      agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
      episodes_names = ['dec16_1', 'dec16_2']

      # loading data
      agents = AgentsAnalytics(
        data_path=input_data_path,
        agents_names=agents_names,
        episodes_names=episodes_names
      )

      agents_logs = agents.agents_data

      fig = AgentsAnalytics.plot_computation_times(
            agents_logs,
            episodes_names=['dec16_1'])
      fig.show()

    """
    agent_names = []

    # for the first agent
    agent_names.append(agents_results[0].agent_name)
    df = agents_results[0].computation_times_several_episodes(episodes_names)
    df = df.rename(columns={'Execution time': agents_results[0].agent_name})

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)
      df2 = agent.computation_times_several_episodes(episodes_names)
      df2 = df2.rename(columns={'Execution time': agent.agent_name})
      df = df.join(df2.set_index('Timestamp'), on='Timestamp')

    # Create traces
    fig = go.Figure()

    for agent_name in agent_names:
      fig.add_trace(
        go.Scatter(x=df['Timestamp'].tolist(), y=df[agent_name].tolist(),
                   mode='lines+markers', name=agent_name))

    fig.update_layout(xaxis={'rangeslider': {'visible': True}}, title=title,
                      xaxis_title='Timestamp', yaxis_title='Execution Time (s)')
    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_distance_from_initial_topology(
      agents_results: List[EpisodesDataTransformer],
      episodes_names: Optional[List] = None,
      title: str = 'Distance from initial topology', **fig_kwargs):
    """line chart representing the number of changes compared to the initial
    topology at each timestamp and for each agent.

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.

    :param agents_results: list of agents episodes log, each item is a list
                           of episode logs for an agent.
    :type agents_results: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
                           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Distance from initial topology'
    :type title: str
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: Time series line graph of actions execution times
    :rtype: plotly figure

    Example of usage:

    .. code-block:: python

      import os
      from grid2bench.AgentsAnalytics import AgentsAnalytics

      input_data_path = os.path.abspath('../data/input')
      agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
      episodes_names = ['dec16_1', 'dec16_2']

      # loading data
      agents = AgentsAnalytics(
        data_path=input_data_path,
        agents_names=agents_names,
        episodes_names=episodes_names
      )

      agents_logs = agents.agents_data

      fig = AgentsAnalytics.plot_distance_from_initial_topology(
            agents_logs,
            episodes_names=['dec16_1'])
      fig.show()

    """

    # for the first agent
    agent_names = [agents_results[0].agent_name]
    df = agents_results[0].distance_from_initial_topology(episodes_names)
    df = df.rename(columns={'Distance': agents_results[0].agent_name})

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)
      df2 = agent.distance_from_initial_topology(episodes_names)
      df2 = df2.rename(columns={'Distance': agent.agent_name})

      df = df.join(df2.set_index('Timestamp'), on='Timestamp')

    # Create traces
    fig = go.Figure()

    for agent_name in agent_names:
      fig.add_trace(
        go.Scatter(x=df['Timestamp'].tolist(), y=df[agent_name].tolist(),
                   mode='lines+markers', line_shape='hvh', name=agent_name))

    fig.update_layout(xaxis={'rangeslider': {'visible': True}}, title=title,
                      xaxis_title='Timestamp', yaxis_title='Distance')
    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_actions_sequence_length(agents_data: List[EpisodesDataTransformer],
                                   episodes_names: Optional[List] = None,
                                   title: str = 'Sequence length of actions',
                                   min_length: int = 0, max_length: int = 400,
                                   **fig_kwargs):
    """Gantt diagram representing sequence actions for each agent.

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

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.

    :param agents_data: list of agents episodes log, each item is a list
                           of episode logs for an agent.
    :type agents_data: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
                           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Distance from initial topology'
    :type title: str
    :param min_length: filter only sequence actions greater than
    :type min_length: int
    :param max_length: filter only asequence actions less than
    :type max_length: int
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: gantt chart of actions sequence
    :rtype: plotly figure

    Example of usage:

      .. code-block:: python

        import os
        from grid2bench.AgentsAnalytics import AgentsAnalytics

        input_data_path = os.path.abspath('../data/input')
        agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
        episodes_names = ['dec16_1', 'dec16_2']

        # loading data
        agents = AgentsAnalytics(
          data_path=input_data_path,
          agents_names=agents_names,
          episodes_names=episodes_names
        )

        agents_logs = agents.agents_data

        fig = AgentsAnalytics.plot_actions_sequence_length(
              agents_logs,
              episodes_names=['dec16_1'])
        fig.show()

    """
    plot_data = []
    for agent in agents_data:
      plot_data.extend(
        agent.action_sequences_to_dict(episodes_names, min_length, max_length))

    fig = px.timeline(plot_data, x_start='Start', x_end='Finish', y='Type',
                      color='Actions',
                      labels={'Type': 'Agent ', 'Actions': 'Action Sequence'},
                      title=title, color_continuous_scale=['green', 'red'], )
    fig.update_layout(xaxis={'rangeslider': {'visible': True}})
    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  # cumulative reward extraction
  def cumulative_reward(agent_data: EpisodesDataTransformer,
                        episodes_names: List):
    """Helper function to transform cumulative rewards and accomplished
    time-steps into a data frame.

    Extract cumulative rewards and completed time steps, then turn them into a
    time-series data frame.

    :param agent_data: agent's episodes log
    :type agent_data: :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
                           for all loaded episodes.
    :type episodes_names: list of str
    :return: dataframe of cumulative reward and accomplished time-steps
    :rtype: pandas dataframe

    Example of usage:

      .. code-block:: python

        import os
        from grid2bench.AgentsAnalytics import AgentsAnalytics

        input_data_path = os.path.abspath('../data/input')
        agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
        episodes_names = ['dec16_1', 'dec16_2']

        # loading data
        agents = AgentsAnalytics(
          data_path=input_data_path,
          agents_names=agents_names,
          episodes_names=episodes_names
        )

        agents_logs = agents.agents_data

        # get expert agent index
        expert_agent_idx = agents.agents_names.index('Expert_Agent')
        expert_agent = agents_logs[expert_agent_idx]

        df = AgentsAnalytics.cumulative_reward(
          expert_agent,
          episodes_names=episodes_names)
        df
    """
    episode_names = []
    cum_rewards = []
    nb_time_steps = []

    for episode in agent_data.episodes_data:
      if episode.episode_name in episodes_names:
        episode_names.append(episode.episode_name)
        cum_rewards.append(episode.cum_reward / 100)
        nb_time_steps.append(episode.nb_timestep_played)

    df = pd.DataFrame(
      data=np.array([episode_names, nb_time_steps, cum_rewards]).transpose(),
      columns=['Episode', 'Played timesteps', 'Cumulative reward'], )

    return df.astype(
      {'Episode': 'str', 'Played timesteps': int, 'Cumulative reward': float})

  @staticmethod
  def plot_cumulative_reward(agents_data: List[EpisodesDataTransformer],
                             episodes_names: Optional[List] = None,
                             CumReward: Optional[bool] = True, title: Optional[
        str] = 'Cumulative reward per episode', **fig_kwargs):
    """A bar graph representing the cumulative reward/accomplished time-steps
    per episodes for each agent.

    If fig_type is default, plot cumulative rewards. If
    fig_type='Acctimesteps'  then plot completed timesteps instead

    You can filter the agents to display by giving the list "agents_results"
    only to the desired agents.

    Similarly, you can filter the episodes by giving the "episode_names" list
    only to the episodes you want to display. If episodes_names=None,
    then  returns the results of all loaded episodes.

    :param agents_data: list of agents episodes log, each item is a list
                           of episode logs for an agent.
    :type agents_data: list of :class:`EpisodeDataTransformer`
    :param episodes_names: filter specific episodes. If none, returns results
                           for all loaded episodes.
    :type episodes_names: list of str
    :param title: bar chart title, bar chart title, default value =
                  'Overloaded Lines by station'
    :type title: str
    :param CumReward: diagram type: cumulative reward per episode or
                     accomplished time-steps per episode.
                     Default = cumulative reward, else accomplished time-steps
    :type CumReward: bool
    :param fig_kwargs: keyword arguments from the plotly library. Example:
                       height= 700. For more arguments vist the plotly
                       documentation https://plotly.com/python/
    :type fig_kwargs: **kwargs
    :return: Barchart of cumulative rewards/accomplished time-steps
    :rtype: plotly figure
    """
    if not episodes_names: episodes_names = agents_data[0].episodes_names

    new_names = {}
    y_list = []

    for i, agent in enumerate(agents_data):
      if CumReward:
        df = AgentsAnalytics.cumulative_reward(agent, episodes_names)
        y = df['Cumulative reward'].tolist()
      else:
        df = AgentsAnalytics.cumulative_reward(agent, episodes_names)
        y = df['Played timesteps'].tolist()

      agent_episodes = df['Episode'].tolist()
      new_names[f'wide_variable_{i}'] = agent.agent_name
      y_list.append(y)

    if CumReward:
      x_title = '$\\frac{Cumulative reward}{100}$'
    else:
      title = x_title = 'Accomplished time steps'

    fig = px.bar(x=agent_episodes, y=y_list, text_auto='.2s',
                 labels={'x': 'Scenario', 'value': x_title}, barmode='group',
                 title=title)

    fig.for_each_trace(
      lambda t: t.update(name=new_names[t.name], legendgroup=new_names[t.name],
                         hovertemplate=t.hovertemplate.replace(t.name,
                                                               new_names[
                                                                 t.name])))

    # Set y-axes titles
    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_actions_freq_by_substation_by_id(
      agents_results: List[EpisodesDataTransformer],
      episodes_names: Optional[List] = None,
      title: Optional[str] = 'Frequency of actions by type',
      row: Optional[int] = 1, col: Optional[int] = 2, **fig_kwargs):

    agent_names = []
    for agent in agents_results:
      agent_names.append(agent.agent_name)

    list_i = []
    for i in range(row):
      list_j = []
      for j in range(col):
        list_j.append({'type': 'domain'})
      list_i.append(list_j)

    fig = make_subplots(row, col, specs=list_i, subplot_titles=agent_names)

    for i in range(row):
      for j in range(col):
        data = agents_results[i + j].get_actions_by_substation_by_id()
        fig.add_trace(go.Pie(labels=data.index, values=data['Frequency'],
                             name=agent_names[i + j]), i + 1, j + 1)

    fig.update_traces(textposition='inside')
    fig.update_layout(title_text=title, uniformtext_minsize=12,
                      uniformtext_mode='hide', **fig_kwargs)
    return fig

  @staticmethod
  def visualize_grid_state(observation_space, agent: EpisodesDataTransformer,
                           episode_name: str, timestamp_str: str, **kwargs):

    plot_helper = PlotMatplot(observation_space)

    # get the observation
    obs = agent.get_observation_by_timestamp(episode_name, timestamp_str)

    return plot_helper.plot_obs(obs)
