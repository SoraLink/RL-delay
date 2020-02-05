# rllab.misc.logger 
* logger is to log the information in to <code>../data/NAME_SCOPE/TIME_STAMP</code>. <code>../data</code> must exists.
## Usage
### step
1. <code>import rllab.misc.logger as logger</code> in your alporithm first;
1. Set name scope:  assign a scope name to <code>logger.task_name_scope</code>, generally <code>logger.task_name_scope = self.env.name</code> in <code>YourAlgos.\_\_init\_\_()</code>
1. Run init: <code>logger.init()</code> if start a new experiment. To resume an experiment:
`
info = logger.init(restore=True, dir_name=YOUR_CHECKPOINT_TIME_STAMP, session=YOUR_TF_SESSION)
`. Return value is the information pickled by checkpoint.

1. Use methods in your algorithm.

## Method
* init(restore = False, dir_name = None, session = None, itr = 'latest'): dir_name (string) can be 'latest', logger will load the latest experiment. itr (int or 'latest') the iteration you want to load, works when checkpointing by 'gap' or 'all'.
* log(s, with_prefix=True, with_timestamp=True, color=None): works like print, but the content will be logged into log file.
* save_itr_params(itr, session, other_info): itr (int) is the number of iteration; session (tf.session) TensorFlow session; other_info (a dict is encouraged) all varibles outside TensorFlow. This function is used to save checkpoint. Checkpoint mode (_snapshot_mode) can be 'last' or 'all' or 'gap', 'last' is default.
* dump_tabular(*args, **kwargs): this function is to insert data (tabular) into csv file. All argument will be assigned to log() function.
* record_tabular(key, val): key (string): name of the column; val (whatever) the data. Record the data during traing, and dump_tabular() will save all records at the end of each epoch.
* push_prefix(prefix): prefix (string): a prefix for each log.
* pop_prefix(): pop out the last prefix.
* _snapshot_mode: default='gap', can be 'last': override the previous checkpoint; 'all': save each checkpoints.
* _snapshot_gap: default=100, save the checkpoint for each 100 steps, only effective when <code>_snapshot_mode='gap'</code>