import os

os.environ['ROS_PYTHON_CHECK_FIELDS'] = '1'
os.environ['LINE_PROFILE'] = '0'

import threading

def preload_matplotlib():
    # preload costly imports that are not used immediately
    from networkx.algorithms.shortest_paths.generic import shortest_path
    from networkx.drawing.nx_pydot import read_dot

# Start preloading in the background
threading.Thread(target=preload_matplotlib, daemon=True).start()