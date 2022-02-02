import papermill
import sys

# pipeline of notebooks
notebooks = [
  './notebooks/preprocessing',
  './notebooks/RandomForestCV',
  './notebooks/mlp',
  './notebooks/process_results',
]

for nbname in notebooks:
  print('Updating', nbname)
  papermill.execute_notebook(f'{nbname}.ipynb', f'{nbname}.ipynb', stdout_file=sys.stdout, stderr_file=sys.stderr)