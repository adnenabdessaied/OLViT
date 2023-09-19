#! /usr/bin/env python
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

Reads SIMMC 2.1 dataset, creates train, devtest, dev formats for ambiguous candidates.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import json
import os
import random


SPLITS = ["train", "dev", "devtest", "teststd"]


def get_image_name(scene_ids, turn_ind):
    """Given scene ids and turn index, get the image name.
    """
    sorted_scene_ids = sorted(
        ((int(key), val) for key, val in scene_ids.items()),
        key=lambda x: x[0],
        reverse=True
    )
    # NOTE: Hardcoded to only two scenes.
    if turn_ind >= sorted_scene_ids[0][0]:
        scene_label = sorted_scene_ids[0][1]
    else:
        scene_label = sorted_scene_ids[1][1]
    image_label = scene_label
    if "m_" in scene_label:
        image_label = image_label.replace("m_", "")
    return f"{image_label}.png", scene_label


def get_object_mapping(scene_label, args):
    """Get the object mapping for a given scene.
    """
    scene_json_path = os.path.join(
        args["scene_json_folder"], f"{scene_label}_scene.json"
    )
    with open(scene_json_path, "r") as file_id:
        scene_objects = json.load(file_id)["scenes"][0]["objects"]
    object_map = [ii["index"] for ii in scene_objects]
    return object_map


def dictionary_to_string(dictionary):
    result = ""
    for k, v in dictionary.items():
        result += k + ":"
        result += str(v) + " "
    return result


def get_all_answers(dialogs):
    all_answers = []
    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        for turn_ind, turn_datum in enumerate(dialog_datum["dialogue"]):
            all_answers.append(turn_datum["system_transcript"])
    return all_answers


def main(args):
    for split in SPLITS:
        read_path = args[f"simmc_{split}_json"]
        print(f"Reading: {read_path}")
        with open(read_path, "r") as file_id:
            dialogs = json.load(file_id)


        # load the metadata files
        with open(args["furniture_prefab_metadata"], "r") as file:
            furniture_metadata = json.load(file)

        with open(args["fashion_prefab_metadata"], "r") as file:
            fashion_metadata = json.load(file)

        # get all answer fromm all dialogues to sample answer candidates from for each dialogue iteration
        all_answers = get_all_answers(dialogs)
        

        # Reformat into simple strings with positive and negative labels.
        # (dialog string, label)
        ambiguous_candidates_data = []
        for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
            turns = []
            q_turns = []
            a_turns = []

            for turn_ind, turn_datum in enumerate(dialog_datum["dialogue"]):
                query = [turn_datum["transcript"]]
                answer = [turn_datum["system_transcript"]]
                answer_candidates = []

                # sample random answers from the list of all answers as answer candidates
                # sample n_answer_candidates - 1 wrong answer candidates from the list of all answers
                for _ in range(int(args["n_answer_candidates"]) - 1):
                    random_idx = random.randint(0, len(all_answers) - 1)
                    answer_candidates.append([all_answers[random_idx]])
                answer_candidates.insert(0, answer)
                #random.shuffle(answer_candidates)

                #annotations = turn_datum["transcript_annotated"]
                #if annotations.get("disambiguation_label", False):
                #label = annotations["disambiguation_candidates"]
                image_name, scene_id = get_image_name(
                    dialog_datum["scene_ids"], turn_ind
                )

                # load the scene files and get all the prefab pahts to get the object descriptions for each scene
                prefab_paths = []
                scene_path = os.path.join(args["scene_json_folder"], f"{scene_id}_scene.json")
                with open(scene_path, "r") as scene_file:
                    scene_data = json.load(scene_file)
                for scene in scene_data["scenes"]:
                    for object in scene["objects"]:
                        prefab_paths.append(object["prefab_path"])
                
                # get the metadata for all objects of the scene (prefab_paths)
                object_metadata = []
                for prefab_path in prefab_paths:
                    if scene_id[:11] in ["cloth_store", "m_cloth_sto"]:
                        object_dict = fashion_metadata[prefab_path]
                    elif scene_id[:7] == "wayfair":
                        object_dict = furniture_metadata[prefab_path]
                    object_str = dictionary_to_string(object_dict)
                    object_metadata.append([object_str])


                # If dialog contains multiple scenes, map it accordingly.
                #object_map = get_object_mapping(scene_label, args)
                new_datum = {
                    "query": query,
                    "answer": answer,
                    "answer_candidates": answer_candidates,
                    "q_turns": copy.deepcopy(q_turns),
                    "a_turns": copy.deepcopy(a_turns),
                    "turns": copy.deepcopy(turns),
                    "object_metadata": object_metadata,
                    "dialog_id": dialog_datum["dialogue_idx"],
                    "turn_id": turn_ind,
                    #"input_text": copy.deepcopy(history),
                    #"ambiguous_candidates": label,
                    "image_name": image_name,
                    #"object_map": object_map,
                }

                ambiguous_candidates_data.append(new_datum)                

                turns.append([turn_datum["transcript"] + turn_datum["system_transcript"]])
                q_turns.append(query)
                a_turns.append(answer)


                # Ignore if system_transcript is not found (last round teststd).
                # if turn_datum.get("system_transcript", None):
                #    history.append(turn_datum["system_transcript"])

        print(f"# instances [{split}]: {len(ambiguous_candidates_data)}")
        save_path = os.path.join(
            args["ambiguous_candidates_save_path"],
            f"simmc2.1_ambiguous_candidates_dstc11_{split}.json"
        )
        print(f"Saving: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(
                {
                    "source_path": read_path,
                    "split": split,
                    "data": ambiguous_candidates_data,
                },
                file_id
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simmc_train_json", default=None, help="Path to SIMMC 2.1 train"
    )
    parser.add_argument(
        "--simmc_dev_json", default=None, help="Path to SIMMC 2.1 dev"
    )
    parser.add_argument(
        "--simmc_devtest_json", default=None, help="Path to SIMMC 2.1 devtest"
    )
    parser.add_argument(
        "--simmc_teststd_json", default=None, help="Path to SIMMC 2.1 teststd (public)"
    )
    parser.add_argument(
        "--scene_json_folder", default=None, help="Path to SIMMC scene jsons"
    )
    parser.add_argument(
        "--ambiguous_candidates_save_path",
        required=True,
        help="Path to save SIMMC disambiguate JSONs",
    )
    parser.add_argument(
        "--fashion_prefab_metadata", required=True,
        help="Path to the file with all metadata for fashion objects"
    )
    parser.add_argument(
        "--furniture_prefab_metadata", required=True,
        help="Path to the file with all metadata for fashion objects"
    )
    parser.add_argument(
        "--n_answer_candidates", required=True,
        help="number of answer candidates for the ranking task"
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
