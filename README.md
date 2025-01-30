Short README because AI can't generate this for me (it still can't do a lot actually)

To run the data pipeline:
```bash
python src/data_0_pipeline.py
```

To run the model pipeline:
```bash
python src/model_0_pipeline.py
```

This will also generate all the plots and save them to the `results/plots/` directory.

Don't try using the other files, they're mainly to help me not break stuff.

Config.py is your main starting point. You change parameters here to try things out and run the pipeline.
Config_features.py allows you to enable/disable features.

This data is poor, we really have very little to work with.

Frankly, the current results prove that a random set of trades with an effective stoploss will perform better than any of these models. -look at the val prediction score, >> it's negative! <<

Data is split into train/val/test. train and val are used for training the model, test is used for for simulation.

We also have a problem when trying to fit data, which is that it is biased towards predicting 0 (as that is the most common outcome). So I suggest we either figure out a custom loss function or revert back to my original method of running simulations and using that data.

Todo (if someone wants to do it):
- [ ] Add a custom loss function
- [ ] Implement feature window 
- [ ] Normalize feature set
- [ ] Create list of test data date ranges
- [ ] Swap target data for a 3d probability distribution function. 
^The even wiser version would be a Reinforcement Learning model trained off actual profit/loss data but the implementation of that could take months.
