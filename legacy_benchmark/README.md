# Benchmark

### Usage
```python=
python run_benchmark.py -s [strategy_name] -t [task_name] -n [exp_name] --config benchmark/config.yaml
```
The results will be logged into ```results/benchmark/[task_name]/[exp_name]```. If ```exp_name``` is not specfied, set to ```strategy_name``` by default.

All strategies and tasks are defined in ```load.py```.

For developers, change ```--config``` if you want to adjust hyperparameters.