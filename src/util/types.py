import numpy as np
from typing import TypeAlias

# Using TypeAlias for compatibility with older Python 3.10+ if needed,
# but the user requested 'type' syntax if they are on 3.12+.
# However, for maximum safety and tool support, I'll use simple aliases if possible
# or the PEP 695 syntax if I'm sure.
# The user specifically asked for:
# type value_array = np.ndarray
# type bit_array = np.ndarray

type value_array = np.ndarray
type bit_array = np.ndarray
