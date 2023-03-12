"""
This file builds graphs from our logs
"""
import itertools
import re
import matplotlib.pyplot as plt

def extract_datapoint_from_line(lines, line_marker, prefix):
    """
    Takes in a set of lines belonging to an epoch, searches for
    a single line containing the desired datapoint, and extracts the
    value. This function assumes that the datapoint is a single floating-
    point number.

    Args:
        lines               A set of lines that we're filtering
        line_marker         A string that only lines containing the datapoint
                            contain (used to filter)
        prefix              The substring right before the datapoint (used to
                            mark where within the line the datapoint is)
    """
    lines_with_datapoint = list(filter(lambda line: line_marker in line, lines))
    error_str = f"For some reason we found {len(lines_with_datapoint)} lines that contain \"{line_marker}\":\n"
    error_str += "\n---\n".join(lines_with_datapoint)
    assert len(lines_with_datapoint) == 1, error_str
    [line_with_datapoint] = lines_with_datapoint
    substr_containing_datapoint = re.search(f"(?<={prefix})[0-9\.?]*", line_with_datapoint)
    datapoint = float(substr_containing_datapoint.group())
    return datapoint

def parse_data_from_lines_in_epoch(lines_in_epoch):
    # Extract epoch number
    epoch_num = extract_datapoint_from_line(lines_in_epoch, "Epoch", "Epoch ")

    # Extract loss
    loss = extract_datapoint_from_line(lines_in_epoch, "Loss", "Loss - ")

    # Extract clean accuracy
    clean_accuracy = extract_datapoint_from_line(lines_in_epoch, "Clean accuracy", "accuracy: ")

    # Extract robust accuracy
    robust_accuracy = extract_datapoint_from_line(lines_in_epoch, "Robust accuracy", "accuracy ")

    return {
        "epoch_num": epoch_num, "loss": loss, "clean_accuracy": clean_accuracy,
        "robust_accuracy": robust_accuracy}

def extract_data_from_log(logfile, starting_string, ending_string):
    """
    Extracts the datapoints from the provided logfile
    """
    # List of datapoints
    output = []

    with open(logfile) as f:
        # Seek to starting string
        for line in f:
            if starting_string in line:
                break

        # Go through file line by line until we hit the ending string
        for line in f:
            if ending_string in line:
                break

            # Accumulate all log lines belonging to the current epoch
            if "Epoch" in line:             # Marks the start of an epoch
                lines_in_epoch = [line]
                for line_in_epoch in itertools.takewhile(lambda x: "Finished" not in x, f):
                    lines_in_epoch += [line_in_epoch]

                extracted_data = parse_data_from_lines_in_epoch(lines_in_epoch)
                output += [extracted_data]

    return output

def create_plot_from_extracted_data(extracted_data):
    epochs = [datapoint['epoch_num'] for datapoint in extracted_data]
    loss = [datapoint['loss'] for datapoint in extracted_data]
    clean_accuracy = [datapoint['clean_accuracy'] for datapoint in extracted_data]
    robust_accuracy = [datapoint['robust_accuracy'] for datapoint in extracted_data]

    # If this assertion fails, our reported data will be corrupted
    assert len(epochs) == len(loss) == len(clean_accuracy) == len(robust_accuracy), "Something catastrophic has happened"

    # Create the figures
    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, layout='constrained')

    # Plot the data
    loss_axis.plot(epochs, loss, label='Training loss')
    accuracy_axis.plot(epochs, clean_accuracy, label='clean accuracy (val)')
    accuracy_axis.plot(epochs, robust_accuracy, label='robust accuracy (val)')

    # Add x-labels
    loss_axis.set_xlabel('Epochs')
    accuracy_axis.set_xlabel('Epochs')

    # Accuracy is a percentage, so it should be between 0 and 1
    accuracy_axis.set_ylim([0,1])

    # Add y-labels
    loss_axis.set_ylabel('Training loss')  # Add a y-label to the axes.
    accuracy_axis.set_ylabel('Accuracy (on val set)')  # Add a y-label to the axes.

    # Add titles
    loss_axis.set_title("Training loss vs epoch")  # Add a title to the axes.
    accuracy_axis.set_title("Accuracy vs epoch")  # Add a title to the axes.

    # Add legends
    loss_axis.legend()
    accuracy_axis.legend()

    plt.show()

# 0 eps
pgd10_0eps_weight_decay_5_extracted_data = extract_data_from_log("outputs_from_128_batch_models.txt", "PGD10 0eps 5e-4 weight_decay", "PGD10 unieps 5e-4 weight_decay")
create_plot_from_extracted_data(pgd10_0eps_weight_decay_5_extracted_data)
