# Copyright 2020 The TimeEvolvingMPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module to define global configuration for the time_evovling_mpo package.
"""

# Numpy datatype
NP_DTYPE = "complex128"

# Seperator string for __str__ functions
SEPERATOR = "----------------------------------------------\n"

# The default backend for tensor network calculations
BACKEND = 'tensor-network'

# Dict of all backends and their default configuration
BACKEND_CONFIG = {
    'tensor-network': {},
    }

# 'silent', 'simple' or 'bar' as a default to show the progress of computations
PROGRESS_TYPE = 'bar'
