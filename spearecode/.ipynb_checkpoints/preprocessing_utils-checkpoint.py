import random
import argparse


def split_and_strip(raw_text, delimiter='\n\n\n'):
    """ Split a string into sections and strip whitespace from each section.

    Args:
        raw_text (str): The string to be processed.
        delimiter (str): The delimiter to use when splitting the string.

    Returns:
        list: A list of the processed sections.
    """
    return [x.strip('\n').strip() for x in raw_text.split(delimiter)]


def remove_chars(text, chars_to_remove):
    """ Remove characters from a string.

    Args:
        text (str): The string to be processed.
        chars_to_remove (list): A list of characters to remove.

    Returns:
        str: The processed string.
    """
    for char in chars_to_remove:
        text = text.replace(char, ' ')
    return text


def replace_chars(text, chars_to_replace):
    """ Replace characters in a string with other characters.

    Args:
        text (str): The string to be processed.
        chars_to_replace (dict): A dictionary of characters to replace as keys and the replacement characters as values.

    Returns:
        str: The processed string.
    """
    for src, dst in chars_to_replace.items():
        text = text.replace(src, dst)
    return text


def preprocess_shakespeare(raw_text, remove_warning_text=True, remove_rare_chars=True,
                           remove_preamble=True, preamble_end=7, remove_bookends=True):
    """ Preprocess the raw text of Shakespeare's works.

    Args:
        raw_text (str): The raw text of Shakespeare's works.
        remove_warning_text (bool, optional): Whether to remove the warning text at the beginning of the text.
        remove_rare_chars (bool, optional): Whether to remove characters that are not in the desired character set.
        remove_preamble (bool, optional): Whether to remove the preamble at the beginning of the text.
        preamble_end (int, optional): The index of the last section of the preamble.
        remove_bookends (bool, optional): Whether to remove the text that is often found at the end
            and the beginning of Shakespeare plays/sonnets/etc.

    Returns:
        str: The processed text.
    """
    raw_text_sections = split_and_strip(raw_text)

    if remove_preamble:
        _ = raw_text_sections[:preamble_end]

    if remove_warning_text:
        warning_text = """<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM
    ...
        DISTRIBUTED OR USED
    COMMERCIALLY.  PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY
    SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>"""

        raw_ss_text = '\n\n'.join(raw_text_sections[preamble_end:]).replace(warning_text, '')
    else:
        raw_ss_text = '\n\n'.join(raw_text_sections[preamble_end:])

    # Remove the text after "THE END" and split on "by William Shakespeare"
    if remove_bookends: 
        ss_text = ''.join(
            [x.rsplit("THE END", 1)[0] for x in raw_ss_text.split("by William Shakespeare") if "THE END" in x]
        )
    else:
        ss_text = ''.join(raw_ss_text)

    # Remove/replace characters that are not in the desired character set
    if remove_rare_chars:
        chars_to_remove = ["`", "}", "_", "<", "_"]
        chars_to_replace = {"[": "(", "]": ")"}
        ss_text = remove_chars(ss_text, chars_to_remove)
        ss_text = replace_chars(ss_text, chars_to_replace)

    return ss_text


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
        

def shuffle_text(text, delimiter='\n\n'):
    """ Shuffle the sections of a string.

    Args:
        text (str): The string to be processed.
        delimiter (str): The delimiter to use when splitting the string.

    Returns:
        str: The processed string.
    """
    text_sections = text.split(delimiter)
    random.shuffle(text_sections)
    return delimiter.join(text_sections)
    

def print_check_speare(ss_text):
    ################################################################################
    #                                 --- RECAP ---
    ################################################################################
    # About 5x the size of the Karpathy dataset
    #     - Note that we have removed about 129,766 characters
    #           --> 110,000 characters from the warning above
    #               >>> len(raw_text)-len(raw_text.replace(_text_to_remove, ''))
    #           --> ~10,000 characters from the gutenberg preamble
    #               >>> len(project_gutenberg_preable_text)
    #           --> ~3,000 characters from the `by` and `THE END` splitting
    #           --> ~6,000 characters from something... I'm not sure
    #                - My guess is that it has to do with the newline characters
    #                  and because I split/replace/join using them it looks like we
    #                  have ~4000-5000 characters less than the raw text just with
    #                  the loss of the \n character between raw_text --> raw_ss_text
    ################################################################################

    ################################################################################
    #                             --- TEXT STATS ---
    ################################################################################
    print("\n... DATASET INFO:"
          f"\n\tNUMBER OF CHARS --> {len(ss_text):,}"
          f"\n\tNUMBER OF LINES --> {len(ss_text.splitlines()):,}")

    # Print a sample of text
    print("\n\n... FIRST 1000 CHARACTERS:\n")
    print(ss_text[:1000])

    # Print a sample of text
    print("\n\n... LAST 1000 CHARACTERS:\n")
    print(ss_text[-1000:])

    __idx = random.sample(range(1000, 100_000, 1000), 1)[0]
    print("\n\n... RANDOM 1000 CHARACTERS:\n")
    print(" ".join(ss_text[__idx:(__idx + 1000)].split(" ")[1:-1]))
    ################################################################################