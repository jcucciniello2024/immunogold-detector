import os
from typing import List


ALLOWED_DATA_ROOTS: List[str] = [
    "/Users/aniksahai/Desktop/Max Planck Project/project/data/Max Planck Data",
    "/mnt/beegfs/home/asahai2024/max-planck-project/project/data/Max Planck Data",
]


def enforce_allowed_data_root(data_root: str) -> str:
    """Allow only the current approved Max Planck dataset roots."""
    root_real = os.path.realpath(os.path.abspath(data_root))
    allowed_real = [os.path.realpath(os.path.abspath(p)) for p in ALLOWED_DATA_ROOTS]

    for allowed in allowed_real:
        if root_real == allowed or root_real.startswith(allowed + os.sep):
            return root_real

    msg = (
        "Data root is blocked for now.\n"
        f"Got: {data_root}\n"
        "Allowed roots:\n- " + "\n- ".join(ALLOWED_DATA_ROOTS)
    )
    raise ValueError(msg)

