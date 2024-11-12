from dateutil.relativedelta import relativedelta
from typing import List, Union, Tuple, Callable
from datetime import datetime, timedelta
from torch import Tensor
import pandas as pd
import torch
import os


def get_data_paths() -> List[Callable[[str], str]]:
    return [
        lambda date: f"./data/SOHO/CELIAS_PM/{date}.csv",
        lambda date: f"./data/SOHO/CELIAS_SEM/{date}.csv",
        lambda date: f"./data/WIND/MAG/{date}.csv",
        lambda date: f"./data/WIND/TDP_PM/{date}.csv",
        lambda date: f"./data/ACE/SWEPAM/{date}.csv",
        lambda date: f"./data/ACE/EPAM/{date}.csv",
        lambda date: f"./data/ACE/SIS/{date}.csv",
        lambda date: f"./data/ACE/MAG/{date}.csv",
        lambda date: f"./data/DSCOVR/L2/faraday/{date}.csv",
        lambda date: f"./data/DSCOVR/L2/magnetometer/{date}.csv",
    ]


def get_data(paths: List[str]) -> Tensor:
    df_list: List[pd.DataFrame] = []
    for path in paths:
        df_list.append(pd.read_csv(path))
    return torch.from_numpy(pd.concat(df_list, axis=1).values).unsqueeze(0)


def timedelta_to_freq(timedelta_obj) -> str:
    total_seconds = timedelta_obj.total_seconds()

    if total_seconds % 1 != 0:
        raise ValueError("Timedelta must represent a whole number of seconds")

    days = total_seconds // (24 * 3600)
    hours = (total_seconds % (24 * 3600)) // 3600
    minutes = ((total_seconds % (24 * 3600)) % 3600) // 60
    seconds = ((total_seconds % (24 * 3600)) % 3600) % 60

    freq_str = ""

    if days > 0:
        freq_str += str(int(days)) + "day"
    if hours > 0:
        freq_str += str(int(hours)) + "hour"
    if minutes > 0:
        freq_str += str(int(minutes)) + "min"
    if seconds > 0:
        freq_str += str(int(seconds)) + "sec"

    return freq_str


def datetime_interval(
    init: datetime,
    last: datetime,
    step_size: Union[relativedelta, timedelta],
    output_format: str = "%Y%m%d",
) -> List[str]:
    current_date = init
    date_list = []
    while current_date <= last:
        date_list.append(current_date.strftime(output_format))
        current_date += step_size
    return date_list


main_condition: Callable[[datetime, datetime], bool] = (
    lambda date, scrap_date: timedelta(days=-4) < scrap_date - date < timedelta(days=4)
)


def date_union(
    first_date: datetime, second_scrap_date: Tuple[datetime, datetime]
) -> Tuple[datetime, datetime]:
    if first_date < second_scrap_date[0]:
        return first_date - timedelta(days=4), second_scrap_date[1]
    elif first_date < second_scrap_date[1]:
        return second_scrap_date
    else:
        return second_scrap_date[0], first_date + timedelta(days=2)


def general_dates(name: str) -> List[Tuple[datetime, datetime]]:
    path: str = f"./{name}.txt"
    assert os.path.exists(path), "Not valid model_type or name, path not found"

    with open(path, "r") as file:
        dates = list(
            map(lambda x: datetime.strptime(x.split()[1], "%Y/%m/%d"), file.readlines())
        )

    scrap_date_list: List[Tuple[datetime, datetime]] = [
        (datetime(1990, 10, 10), datetime(1990, 10, 11))
    ]

    for date in dates:
        flag: bool = True
        i = 0
        while i < len(scrap_date_list):
            scrap_date = scrap_date_list[i]
            if main_condition(date, scrap_date[0]) or main_condition(
                date, scrap_date[1]
            ):
                scrap_date_list[i] = date_union(date, scrap_date)
                flag = False
                break
            i += 1

        if flag:
            scrap_date_list.append((date - timedelta(days=4), date + timedelta(days=2)))

    return scrap_date_list[1:]


def merge_scrap_date_lists(
    first: List[Tuple[datetime, datetime]], second: List[Tuple[datetime, datetime]]
) -> List[Tuple[datetime, datetime]]:
    a: List[Tuple[datetime, datetime]] = sorted(first + second, key=lambda x: x[0])
    out_scrap_date_list: List[Tuple[datetime, datetime]] = []
    i: int = 0
    while i < len(a):
        start, end = a[i]
        while i + 1 < len(a) and a[i + 1][0] <= end:
            end = max(end, a[i + 1][1])
            i += 1
        out_scrap_date_list.append((start, end))
        i += 1

    return out_scrap_date_list


def create_dates() -> List[Tuple[datetime, datetime]]:
    return merge_scrap_date_lists(
        general_dates("pre_2016"),
        general_dates("post_2016"),
    )
