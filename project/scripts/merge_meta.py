import pickle
import sys
from pathlib import Path


def load(f_p):
    """Load a pickle file."""
    with open(f_p, "rb") as f:
        return pickle.load(f)


def merge_pickles(file_paths):
    """Merge dictionaries from multiple pickle files."""
    merged_dict = {}
    for path in file_paths:
        # Load each pickle file and merge
        current_dict = load(path)
        merged_dict.update(current_dict)
    return merged_dict


def modify_and_validate_data(data):
    """Modify keys in the data and validate."""
    meta_sentence_key = "Metaphoric Sentence"
    meta_sentence_key_plural = "Metaphoric Sentences"
    meta_sentence_key_1 = "Metaphoric Sentence 1"

    new_data = {}
    for key, val in data.items():
        new_val = {k.replace("The ", ""): v for k, v in val.items()}
        if meta_sentence_key_1 in val:
            new_val[meta_sentence_key] = val[meta_sentence_key_1]
        new_data[key] = new_val

    # Validation
    for val in new_data.values():
        if meta_sentence_key not in val and meta_sentence_key_plural not in val:
            raise AssertionError("Validation failed.")
    return new_data


def main():
    # Check if enough arguments are provided
    if len(sys.argv) < 3:
        print(
            "Usage: python script.py <output_file_path> <path_to_pickle1> <path_to_pickle2> ..."
        )
        sys.exit(1)

    # The last argument is the output file path
    output_file_path = sys.argv[1]
    # The rest of the arguments are the input file paths
    file_paths = sys.argv[2:]

    merged_dict = merge_pickles(file_paths)
    final_data = modify_and_validate_data(merged_dict)

    if not Path(output_file_path).parent.exists():
        Path(output_file_path).parent.mkdir(parents=True)

    # Save the final merged and modified dictionary
    with open(output_file_path, "wb") as f:
        pickle.dump(final_data, f)

    print(f"Merged pickle file saved as '{output_file_path}'.")


if __name__ == "__main__":
    main()
