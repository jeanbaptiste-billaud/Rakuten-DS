import os
import sys
from src.utils.common_utils import get_project_root

def test_get_project_root():
    """
    Test basic behavior of get_project_root.
    It should return an absolute path ending with the project folder name (Rakuten-DS).
    """
    root = get_project_root()
    assert os.path.isabs(root)
    assert os.path.basename(root) == "Rakuten-DS" or os.path.basename(root) == "workspaces" # in some envs it might be workspace root
