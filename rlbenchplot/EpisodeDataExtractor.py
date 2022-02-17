import os
import numpy as np
import pandas as pd
from grid2op.Episode import EpisodeData
from datetime import timedelta


class EpisodeDataExtractor:
    """Data for episode analysis extracted from grid2op's EpisodeData class


    Attributes:
        agent_path : a string indicating the agent's path
        agent_name : a string indicating the name of the agent
        episode_name : a string indicating the name of the agent's episode

        observations: a list of observation objects in the episode
        actions : a list of action objects in the episode

        n_observation : number of observations in the episode
        computation_times : a list of the execution time of each action in the episode

    """

    def __init__(self, agent_path, episode_name):
        """Loading data by using EpisodeData class"""

        self.agent_path = os.path.abspath(agent_path)
        self.agent_name = os.path.basename(agent_path)
        self.episode_name = episode_name

        try:
            self.observations = EpisodeData.from_disk(agent_path=self.agent_path, name=self.episode_name).observations
            self.actions = EpisodeData.from_disk(agent_path=self.agent_path, name=self.episode_name).actions
            self.computation_times = EpisodeData.from_disk(agent_path=self.agent_path, name=self.episode_name).times
        except FileNotFoundError:
            print("Wrong episode name or agent path")

        self.n_observation = len(self.observations)
        self.n_action = len(self.actions)
        self.timestamps = [self.observations[i].get_time_stamp() for i in range(self.n_action)]

    def get_observation_by_timestamp(self, datetime):
        """

        :param datetime:
        :return:
        """
        return self.observations[self.timestamps.index(datetime)]

    def get_action_by_timestamp(self, datetime):
        """

        :param datetime:
        :return:
        """
        return self.actions[self.timestamps.index(datetime)]

    def get_computation_time_by_timestamp(self, datetime):
        """

        :param datetime:
        :return:
        """
        return self.computation_times[self.timestamps.index(datetime)]

    def get_timestep_by_datetime(self, datetime):
        """

        :param datetime:
        :return:
        """
        return self.timestamps.index(datetime)

    def compute_actions_freq_by_timestamp(self):
        """

        :return:
        """
        action_freq = []

        for i in range(self.n_action - 1):
            action_impact = self.actions[i].impact_on_objects()

            if action_impact["has_impact"]:
                nb_actions = (
                    # number of lines switched
                        action_impact["switch_line"]["count"]
                        +
                        # number of topological changes
                        len(action_impact["topology"]["bus_switch"])
                        +
                        len(action_impact["topology"]["assigned_bus"])
                        +
                        len(action_impact["topology"]["disconnect_bus"])
                        +
                        # number of redispatch changes
                        len(action_impact["redispatch"]["generators"])
                        +
                        # number of storage changes
                        len(action_impact["storage"]["capacities"])
                        +
                        # number of curtailment changes
                        len(action_impact["curtailment"]["limit"]))

                action_freq.append({
                    "Timestamp": self.timestamps[i],
                    "NB action": nb_actions
                })

        return pd.DataFrame(action_freq)

    def compute_actions_freq_by_type(self):
        """

        :return:
        """
        timestamps = []
        nb_switch_line = []
        nb_topological_changes = []
        nb_redispatch_changes = []
        nb_storage_changes = []
        nb_curtailment_changes = []

        for i in range(self.n_action - 1):
            action_impact = self.actions[i].impact_on_objects()

            if action_impact["has_impact"]:
                timestamps.append(self.timestamps[i])
                nb_switch_line.append(action_impact["switch_line"]["count"])
                nb_topological_changes.append(
                    len(action_impact["topology"]["bus_switch"])
                    +
                    len(action_impact["topology"]["assigned_bus"])
                    +
                    len(action_impact["topology"]["disconnect_bus"])
                )
                nb_redispatch_changes.append(len(action_impact["redispatch"]["generators"]))
                nb_storage_changes.append(len(action_impact["storage"]["capacities"]))
                nb_curtailment_changes.append(len(action_impact["curtailment"]["limit"]))

        dict_actions_freq = {
            "Timestamp": timestamps,
            "NB line switched": nb_switch_line,
            "NB topological change": nb_topological_changes,
            "NB redispatching": nb_redispatch_changes,
            "NB storage changes": nb_storage_changes,
            "NB curtailment": nb_curtailment_changes
        }

        return pd.DataFrame(dict_actions_freq)

    def compute_actions_freq_by_station(self):
        """

        :return:
        """
        subs_impacted_by_timestamp = []

        for action, i in zip(self.actions, range(self.n_action - 1)):
            subs_impacted = action.get_topological_impact()[1]
            list_subs_impacted = action.name_sub[np.where(subs_impacted == True)]
            if len(list_subs_impacted) != 0:
                subs_impacted_by_timestamp.append({
                    "Timestamp": self.timestamps[i],
                    "subs_impacted": set(list_subs_impacted)
                }
                )

        return subs_impacted_by_timestamp

    def compute_overloaded_lines_by_timestamp(self):
        """

        :return:
        """

        overloaded_lines = []

        for observation, i in zip(self.observations, range(self.n_action)):
            lines_id = np.where(observation.timestep_overflow != 0)[0]
            if len(lines_id) != 0:
                overloaded_lines.append(
                    {
                        "Timestamp": self.timestamps[i],
                        "Overloaded lines": set(lines_id)
                    }
                )

        return overloaded_lines

    def compute_disconnected_lines_by_timestamp(self):
        """

        :return:
        """

        disconnected_lines = []

        for observation, i in zip(self.observations, range(self.n_action)):
            lines_id = np.where(observation.line_status != True)[0]
            if len(lines_id) != 0:
                disconnected_lines.append(
                    {
                        "Timestamp": self.timestamps[i],
                        "Disconnected lines": set(lines_id)
                    }
                )

        return disconnected_lines

    # TODO : optimize it
    def compute_action_sequences_length(self):
        """

        :return:
        """
        action_sequences_length = self.compute_actions_freq_by_type().copy()
        columns = [x for x in action_sequences_length.columns if x not in ["Timestamp"]]
        df = action_sequences_length[columns]
        df[df > 0] = 1
        action_sequences_length.loc[:, columns] = df

        for column in columns:
            for i in range(1, len(action_sequences_length)):
                if action_sequences_length.loc[i, column] != 0:
                    previous_timestamp = action_sequences_length.loc[i, "Timestamp"] - timedelta(minutes=5)
                    if previous_timestamp in list(action_sequences_length["Timestamp"]):
                        if action_sequences_length.loc[i - 1, column] != 0:
                            action_sequences_length.loc[i, column] = action_sequences_length.loc[i - 1, column] + 1

        return action_sequences_length

    def _n_lines(self):
        """

        :return:
        """
        return self.observations[0].n_line

    def _name_of_lines(self, lines_id):
        """

        :return:
        """
        return self.actions[0].name_line[lines_id]

    #Functions reused from Grid2Vis
    def get_distance_from_obs(
            self, act, line_statuses, subs_on_bus_2, objs_on_bus_2, obs
    ):

        impact_on_objs = act.impact_on_objects()

        # lines reconnections/disconnections
        line_statuses[
            impact_on_objs["force_line"]["disconnections"]["powerlines"]
        ] = False
        line_statuses[
            impact_on_objs["force_line"]["reconnections"]["powerlines"]
        ] = True
        line_statuses[impact_on_objs["switch_line"]["powerlines"]] = np.invert(
            line_statuses[impact_on_objs["switch_line"]["powerlines"]]
        )

        topo_vect_dict = {
            "load": obs.load_pos_topo_vect,
            "generator": obs.gen_pos_topo_vect,
            "line (extremity)": obs.line_ex_pos_topo_vect,
            "line (origin)": obs.line_or_pos_topo_vect,
        }

        # Bus manipulation
        if impact_on_objs["topology"]["changed"]:
            for modif_type in ["bus_switch", "assigned_bus"]:

                for elem in impact_on_objs["topology"][modif_type]:
                    objs_on_bus_2 = self.update_objs_on_bus(
                        objs_on_bus_2, elem, topo_vect_dict, kind=modif_type
                    )

            for elem in impact_on_objs["topology"]["disconnect_bus"]:
                # Disconnected bus counts as one for the distance
                subs_on_bus_2[elem["substation"]] = True

        subs_on_bus_2 = [
            True if objs_on_2 else False for _, objs_on_2 in objs_on_bus_2.items()
        ]

        distance = len(line_statuses) - line_statuses.sum() + sum(subs_on_bus_2)
        return distance, line_statuses, subs_on_bus_2, objs_on_bus_2

    def update_objs_on_bus(self, objs_on_bus_2, elem, topo_vect_dict, kind):
        for object_type, pos_topo_vect in topo_vect_dict.items():
            if elem["object_type"] == object_type and elem["bus"]:
                if kind == "bus_switch":
                    objs_on_bus_2 = self.update_objs_on_bus_switch(
                        objs_on_bus_2, elem, pos_topo_vect
                    )
                else:
                    objs_on_bus_2 = self.update_objs_on_bus_assign(
                        objs_on_bus_2, elem, pos_topo_vect
                    )
                break
        return objs_on_bus_2

    @staticmethod
    def update_objs_on_bus_switch(objs_on_bus_2, elem, pos_topo_vect):
        if pos_topo_vect[elem["object_id"]] in objs_on_bus_2[elem["substation"]]:
            # elem was on bus 2, remove it from objs_on_bus_2
            objs_on_bus_2[elem["substation"]] = [
                x
                for x in objs_on_bus_2[elem["substation"]]
                if x != pos_topo_vect[elem["object_id"]]
            ]
        else:
            objs_on_bus_2[elem["substation"]].append(pos_topo_vect[elem["object_id"]])
        return objs_on_bus_2

    @staticmethod
    def update_objs_on_bus_assign(objs_on_bus_2, elem, pos_topo_vect):
        if (
                pos_topo_vect[elem["object_id"]] in objs_on_bus_2[elem["substation"]]
                and elem["bus"] == 1
        ):
            # elem was on bus 2, remove it from objs_on_bus_2
            objs_on_bus_2[elem["substation"]] = [
                x
                for x in objs_on_bus_2[elem["substation"]]
                if x != pos_topo_vect[elem["object_id"]]
            ]
        elif (
                pos_topo_vect[elem["object_id"]] not in objs_on_bus_2[elem["substation"]]
                and elem["bus"] == 2
        ):
            objs_on_bus_2[elem["substation"]].append(pos_topo_vect[elem["object_id"]])
        return objs_on_bus_2