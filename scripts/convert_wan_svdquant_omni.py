# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert existing Nunchaku Wan MXFP4 onefiles or pipelines into canonical Omni format."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Nunchaku Wan onefile, component, or complete pipeline")
    parser.add_argument("--output", required=True, help="New output directory; existing paths are rejected")
    args = parser.parse_args(argv)
    from auto_round.export.svdquant_omni import convert_wan_nunchaku_to_omni

    print(convert_wan_nunchaku_to_omni(args.source, args.output))


if __name__ == "__main__":
    main()
