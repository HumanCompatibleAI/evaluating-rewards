"""Shard tests to support parallelism across multiple machines."""

from typing import Iterable, List, Sequence, TypeVar
import hashlib

Item = TypeVar("Item")


def positive_int(x) -> int:
    x = int(x)
    if x < 0:
        raise ValueError(f"Argument {x} must be positive")
    return x


def pytest_addoption(parser):
    group = parser.getgroup("shard")
    group.addoption(
        "--shard-id",
        dest="shard_id",
        type=positive_int,
        default=0,
        help="Number of this shard.",
    )
    group.addoption(
        "--num-shards",
        dest="num_shards",
        type=positive_int,
        default=1,
        help="Total number of shards.",
    )


def get_verbosity(config) -> bool:
    return config.option.verbose


def pytest_report_collectionfinish(config, startdir, items: Iterable[Item]):
    verbosity = get_verbosity(config)
    if verbosity == 0:
        return "running {} items due to CircleCI parallelism".format(len(items))
    elif verbosity > 0:
        return "running {} items due to CircleCI parallelism: {}".format(
            len(items), ", ".join([item.nodeid for item in items])
        )


def hash(x: str) -> int:
    return int(hashlib.md5(x.encode()).hexdigest(), 16)


def filter_items_by_shard(items: Iterable[Item], shard_id: int, num_shards: int) -> Sequence[Item]:
    shards = [hash(item.nodeid) % num_shards for item in items]

    new_items = []
    for shard, item in zip(shards, items):
        if shard == shard_id:
            new_items.append(item)
    return new_items


def pytest_collection_modifyitems(session, config, items: List[Item]):
    shard_id = config.getoption("shard_id")
    shard_total = config.getoption("num_shards")
    if shard_id >= shard_total:
        raise ValueError("shard_num = f{shard_num} must be less than shard_total = f{shard_total}")

    items[:] = filter_items_by_shard(items, shard_id, shard_total)
