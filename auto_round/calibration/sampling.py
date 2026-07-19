# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0


def uniform_call_indices(total_calls: int, max_calls: int) -> tuple[int, ...]:
    """Return deterministic call indices uniformly spanning a sequence."""
    if type(total_calls) is not int or total_calls < 1:
        raise ValueError(f"`total_calls` must be a positive integer, got {total_calls!r}")
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError(f"`max_calls` must be a positive integer, got {max_calls!r}")
    if total_calls <= max_calls:
        return tuple(range(total_calls))
    if max_calls == 1:
        return (total_calls // 2,)
    return tuple(index * (total_calls - 1) // (max_calls - 1) for index in range(max_calls))
