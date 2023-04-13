from typing import List, Callable, Union
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value: Union[str, bytes], is_list: bool = False) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte.
    
    Args:
        value (Union[str, bytes]): A string or byte value to convert.
        is_list (bool, optional): Whether the input value is a list. Defaults to False.
    
    Returns:
        tf.train.Feature: A bytes_list feature.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    
    if not is_list:
        value = [value]
    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value: float, is_list: bool = False) -> tf.train.Feature:
    """Returns a float_list from a float / double.
    
    Args:
        value (float): A float value to convert.
        is_list (bool, optional): Whether the input value is a list. Defaults to False.
    
    Returns:
        tf.train.Feature: A float_list feature.
    """
    if not is_list:
        value = [value]
        
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: Union[bool, int], is_list: bool = False) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint.
    
    Args:
        value (Union[bool, int]): A bool or integer value to convert.
        is_list (bool, optional): Whether the input value is a list. Defaults to False.
    
    Returns:
        tf.train.Feature: An int64_list feature.
    """
    if not is_list:
        value = [value]
        
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_raw(token_ids: List[int]) -> bytes:
    """Creates a tf.Example message ready to be written to a file from N features.

    Args:
        token_ids (list of ints): A list of integers representing the tokens for each string
    
    Returns:
        bytes: A tf.Example Message ready to be written to file.
    """
    # Create a dictionary mapping the feature name to the 
    # tf.Example-compatible data type.
    feature = {
        "token_content": _int64_feature(token_ids, is_list=True),
    }
    
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(ds, n_ex, output_suffix, version_str, n_ex_per_rec=10_000, serialize_fn=serialize_raw, out_dir="./tfrecords", verbose=True):
    """Write the dataset into TFRecords format.
    
    Args:
        ds: The dataset to be written.
        n_ex (int): The number of examples in the dataset.
        output_suffix (str): The output suffix for the TFRecords files.
        version_str (str): The version string for the dataset.
        n_ex_per_rec (int, optional): The number of examples per TFRecord file. Defaults to 10_000.
        serialize_fn (Callable, optional): The serialization function for the dataset. Defaults to serialize_raw.
        out_dir (str, optional): The output directory for the TFRecords files. Defaults to "./tfrecords".
        verbose (bool, optional): Whether to display progress information. Defaults to True.
    
    Returns:
        None; TFRecords are created in the specified folder
    """
    n_recs = int(np.ceil(n_ex/n_ex_per_rec))
    rec_range = range(n_recs)
    
    # Make dataset generator iterable
    ds = iter(ds)
        
    # Dataset directory
    if not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.join(out_dir, f"{output_suffix.strip('_')}_{version_str}")
    if not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)
        
    # Add progress bar if requested
    if verbose:
        rec_range = tqdm(rec_range, desc="Writing TFRecords", total=n_recs)
    
    # Create tfrecords
    for i in rec_range:
        print(f"\n... Writing TFRecord {i+1} of {n_recs} ({n_ex_per_rec} per TFRecord)...\n")
        tfrec_path = os.path.join(out_dir, f"{(i+1):02}_{n_recs:02}.tfrec")
        
        # This makes the tfrecord
        with tf.io.TFRecordWriter(tfrec_path) as writer:
            for ex in tqdm(range(n_ex_per_rec), total=n_ex_per_rec):
                try:
                    example = serialize_fn(next(ds))
                    writer.write(example)
                except:
                    break