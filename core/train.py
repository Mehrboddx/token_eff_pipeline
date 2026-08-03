from dataset import load_math_dataset, filter_by_level, train_test

dataset = load_math_dataset()

dataset = filter_by_level(dataset, ["Level 1", "Level 2"])

train_ds, eval_ds = train_test(dataset)

