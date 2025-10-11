import json
import os

import coredumpy
import requests

coredumpy.patch_except(directory='./dumps')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 "
                  "Mobile Safari/537.36",
    "Content-Type": "text/html",
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'game-alias': 'ba'
}


def _download_json(url):
    for i in range(4):
        response = requests.get(
            url=url,
            headers=HEADERS,
        )
        if response.status_code != 200:
            print(f"Failed to fetch data from {url}, error code: {response.status_code},retry {i + 1}/3")
            continue
        return response.json()
    raise ValueError(f"Failed to fetch data from {url}")


def update_activity_schaledb():
    events_json = _download_json("https://schaledb.com/data/en/events.min.json")
    en_localization_json = _download_json("https://schaledb.com/data/en/localization.min.json")
    cn_localization_json = _download_json("https://schaledb.com/data/zh/localization.min.json")
    jp_localization_json = _download_json("https://schaledb.com/data/jp/localization.min.json")

    armor_type_translator = {1: "burst", 2: "pierce", 3: "mystic", 4: "shock"}

    # remove permanent events
    event_ids = []

    # try:
    #     event_ids.remove(807) # miku
    #     event_ids.remove(831) # railgun
    # except ValueError:
    #     pass
    #万一呢

    for event_data in events_json["Events"]:
        if "Permanent" in event_data and event_data["Permanent"].get("EventOpenCn", False):
            continue
        event_ids.append(event_data["Id"])

    for event_id in event_ids:
        # mkdir for each event
        if not os.path.exists(f"../src/activities/{event_id}"):
            os.mkdir(f"../src/activities/{event_id}")
        event_stage_list = []
        for stage_id, stage_data in events_json["Stages"].items():
            if str(stage_data["Id"]).startswith(str(event_id)):
                continue
            difficulty = stage_data["Difficulty"]
            if difficulty != 1:
                # Currently we only save task data of normal difficulty
                # Known difficulties:
                # 1: Normal
                # 2: Challenge
                continue
            stage_index = stage_data["Stage"]
            stage_name = stage_data["Name"]

            # stage_data["ArmorType"] is a list and indicates any possible armor type of enemy
            # we only keep the first armor type for simplicity
            armor_type = armor_type_translator[stage_data["ArmorTypes"][0]]

            # stage_data["EntryCost"](list):each item(list) [0] is the cost type, [1] is the cost amount
            # here type is always AP so we only keep the amount
            entry_cost = stage_data["EntryCost"][0][1]
            event_stage_list.append({
                "stage_index": stage_index,
                "stage_name": stage_name,
                "entry_cost": entry_cost,
                "armor_type": armor_type
            })
        # sort stages by stage_index
        event_stage_list.sort(key=lambda x: x["stage_index"])
        with open(f"../src/activities/{event_id}/info.json", "w", encoding="utf-8") as f:
            event_info = {
                "event_id": event_id,
                "event_name_en": en_localization_json["EventName"][str(event_id)],
                "event_name_cn": cn_localization_json["EventName"][str(event_id)],
                "event_name_jp": jp_localization_json["EventName"][str(event_id)],
                "stages": event_stage_list
            }
            json.dump(event_info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    update_activity_schaledb()
