import os
import random
import numpy as np
import tensorflow as tf

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
def load_jsonl(path):
    """
    Load a jsonl file into a pandas dataframe
    """
    df = pd.read_json(path, lines=True, )
    return df


def save_pickle(obj: object, file_path: str) -> None:
    """
    Save an object to a pickle file.

    Args:
        obj: The object to save.
        file_path: The path to the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> object:
    """
    Load an object from a pickle file.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The loaded object.
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_from_txt_file(path):
    """Load the raw text from a file.

    Args:
        path (str): The path to the text file.

    Returns:
        str: The raw text from the file.
    """
    with open(path, 'r', encoding="utf-8") as f:
        return f.read()


def save_to_txt_file(text, path):
    """Load the raw text from a file.

    Args:
        text (str): Text to save
        path (str): The path to the text file.

    Returns:
        None; File is saved
    """
    with open(path, "w") as f:
        f.write(text)


def tf_set_memory_growth():
    print(f"\n... SETTING MEMORY GROWTH STARTING ...\n")
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except ValueError:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    print(f"\n... SETTING MEMORY GROWTH COMPLETED ...\n")


def tf_xla_jit():
    print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")
    print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
    # enable XLA optmizations (10% speedup when using @tf.function calls)
    tf.config.optimizer.set_jit(True)
    print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")


def flatten_l_o_l(nested_list):
    """Flatten a list of lists.

    Args:
        nested_list (list): A list of lists to be flattened.

    Returns:
        list: A flattened list.
    """

    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """Print a line of a given length with a specified symbol.

    Args:
        symbol (str, optional): The symbol to use for the line. Defaults to "-".
        line_len (int, optional): The length of the line to print. Defaults to 110.
        newline_before (bool, optional): Whether to print a newline character before the line. Defaults to False.
        newline_after (bool, optional): Whether to print a newline character after the line. Defaults to False.

    Returns:
        None; Prints a line
    """

    if newline_before: print()
    print(symbol * line_len)
    if newline_after: print()