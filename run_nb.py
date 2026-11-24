import nbformat, time, sys
from nbclient import NotebookClient
t=time.time()
nb=nbformat.read("Coal_Proximate_to_Ultimate_Model.ipynb", as_version=4)
NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata":{"path":"."}}).execute()
nbformat.write(nb,"Coal_Proximate_to_Ultimate_Model.ipynb"); print("executed in", round(time.time()-t), "s")
