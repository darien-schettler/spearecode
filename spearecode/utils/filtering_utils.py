from typing import List, Union, Callable, Tuple, Optional
import pandas as pd
import os


def get_n_chars(text: str) -> int:
    """ Calculate the number of characters in the given text.

    Args:
        text (str): Input text.

    Returns:
        int: Number of characters in the text.
    """
    return len(text)


def get_n_lines(text: str) -> int:
    """ Calculate the number of lines in the given text.

    Args:
        text (str): Input text.

    Returns:
        int: Number of lines in the text.
    """
    return len(text.split("\n"))


def get_n_tokens(tokens: List[int]) -> int:
    """ Calculate the number of tokens in the given list.

    Args:
        tokens (List[int]): List of integers representing tokens.

    Returns:
        int: Number of tokens in the list.
    """
    return len(tokens)


def tokenize(text: str, encoder: Callable) -> List[int]:
    """ Tokenize the given text using the provided encoder function.

    Args:
        text (str): Input text.
        encoder (Callable): Function to tokenize the text.

    Returns:
        List[int]: List of integers representing tokenized string.
    """
    return encoder(text)


def check_chunks(n_tokens: int, min_chunk_size: int = 128, max_chunk_size: int = 2048) -> bool:
    """ Check if the number of tokens falls within the specified chunk size range.

    Args:
        n_tokens (int): Number of tokens.
        min_chunk_size (int, optional): Minimum chunk size. Defaults to 128.
        max_chunk_size (int, optional): Maximum chunk size. Defaults to 2048.

    Returns:
        bool: True if the number of tokens falls within the specified range, False otherwise.
    """
    return min_chunk_size <= n_tokens < max_chunk_size


def get_metadata_df(df, drop_col_strings: Tuple[str] = ('content',), additional_drop_strs: Optional[List[str]] = None) -> pd.DataFrame:
    """ Get a metadata dataframe with specified columns removed.

    Args:
        df (pd.DataFrame): Input dataframe.
        drop_col_strings (Tuple[str], optional): 
            – Tuple of strings to remove columns that contain them. Defaults to ('content',).
        additional_drop_strs (Optional[List[str]], optional): 
            – Additional strings to remove columns that contain them. Defaults to None.

    Returns:
        pd.DataFrame: New dataframe with specified columns removed.
    """
    if additional_drop_strs is not None:
        drop_col_strings = list(additional_drop_strs) + additional_drop_strs
    cols_to_drop = [_c for _c in df.columns if any(_x in _c for _x in drop_col_strings)]
    return df.copy().drop(columns=cols_to_drop)


def pad_truncate_centered(tokenized_str: List[int], fixed_length: int = 384, pad_value: int = 0) -> List[int]:
    """
    Pad or truncate the tokenized strings such that they have a 
    fixed length and are centered around the middle of the string.

    Args:
        tokenized_str (list): List of integers representing tokenized string.
        fixed_length (int, optional): The desired fixed length for the output list. Defaults to 384.
        pad_value (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        list: Padded or truncated list of integers with the specified fixed length.
    """
    n_tokens = len(tokenized_str)

    if n_tokens < fixed_length:
        n_pad = fixed_length - n_tokens
        n_left_pad = n_pad // 2
        n_right_pad = n_pad - n_left_pad
        return [pad_value] * n_left_pad + tokenized_str + [pad_value] * n_right_pad
    else:
        n_remove = n_tokens - fixed_length
        n_left_remove = n_remove // 2
        n_right_remove = n_remove - n_left_remove
        return tokenized_str[n_left_remove:n_tokens - n_right_remove]

    
def drop_str_from_col_names(df: pd.DataFrame, s_to_drop: str) -> pd.DataFrame:
    """ Remove a specified string from the column names of a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        s_to_drop (str): String to remove from column names.

    Returns:
        pd.DataFrame: New dataframe with updated column names.
    """
    df.columns = [_c.replace(s_to_drop, "").replace("__", "_").strip("_") for _c in df.columns]
    return df


def save_ds_version(df: pd.DataFrame, 
                    output_suffix: str, 
                    meta_dir: str, 
                    ds_dir: str, 
                    version_str: str = "v1", 
                    meta_df: Optional[pd.DataFrame] = None) -> None:
    """ Save the dataset version as CSV files.

    Args:
        df (pd.DataFrame): Input dataframe.
        output_suffix (str): Output file suffix.
        meta_dir (str): Directory path to save metadata CSV.
        ds_dir (str): Directory path to save dataset CSV.
        version_str (str, optional): Dataset version string. Defaults to "v1".
        meta_df (Optional[pd.DataFrame], optional): 
            – Metadata dataframe. If not provided, it is generated from the input dataframe. Defaults to None.
            
    Returns:
        None; Saves dataframes to disk (content and metadata)
    """
    ds_csv_path = os.path.join(ds_dir, version_str)
    df.to_csv(ds_csv_path + f"_{output_suffix.strip('_')}.csv", index=False)

    if meta_df is None:
        meta_df = get_metadata_df(df)

    ds_meta_path = os.path.join(meta_dir, version_str)
    meta_df.to_csv(ds_meta_path + f"_{output_suffix.strip('_')}.csv", index=False)
