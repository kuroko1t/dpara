multiprocess GPU running for dnn model.
Manage multiple GPUs to search for hyper-parameters on GPUs with different performance.

```python
from model import dnn_model_run
from devq.core import Dpara

dpara = Dpara()

batch_size_list = [32, 64, 128]

for i, input_param  in enumerate(batch_size_list):
    dpara.loop_para(dnn_model_run, (input_param), i)
```

# LICENCE
MIT
