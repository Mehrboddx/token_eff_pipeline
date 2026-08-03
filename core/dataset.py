from datasets import load_dataset


def load_math_dataset(name="qwedsacf/competition_math", split="train"):
    """Load the MATH-style dataset (problem, level, type, solution)."""
    dataset = load_dataset(name, split=split)
    return dataset


def filter_by_level(dataset, levels):
    """Keep only specific difficulty levels, e.g. levels=['Level 1', 'Level 2']."""
    return dataset.filter(lambda x: x["level"] in levels)


def filter_by_type(dataset, types):
    """Keep only specific problem types, e.g. types=['Algebra', 'Geometry']."""
    return dataset.filter(lambda x: x["type"] in types)


def train_test(dataset, test_size=0.1, seed=42):
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


if __name__ == "__main__":
    ds = load_math_dataset()
    print(ds)
    print(ds[0])