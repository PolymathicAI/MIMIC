import os
import pty
import sys

from lore import logger


def download_file(name, link, dest_file):
    """
    Download a file using wget with resume support via a pseudo-terminal.

    This function downloads a file from the specified URL to the given destination path.
    It utilizes wget in a child process, ensuring that progress bars and other TTY-specific
    features are displayed correctly. The parent process reads and forwards the output from
    the pseudo-terminal, allowing real-time feedback to the user.

    Parameters:
        name (str): A descriptive identifier for the file being downloaded.
        link (str): The URL of the file to download.
        dest_file (str): The full path (including filename) where the downloaded file will be saved.

    Returns:
        int: The exit status of the wget command. An exit status of 0 indicates success,
             while any non-zero value indicates that the download encountered an error.

    Notes:
        - The function forks a pseudo-terminal to trick wget into enabling its progress bar.
        - Any errors during the reading of the pseudo-terminal output are logged,
          and the final exit status of the wget process is used to determine success.
    """

    # Construct wget command with resume (-c) and output file option (-O)
    command = ["wget", "-c", "-O", dest_file, link]

    # Fork a pseudo-terminal so that wget sees a TTY and displays its progress bar properly.
    pid, fd = pty.fork()
    if pid == 0:
        # Child process: change directory and execute wget.
        os.execvp("wget", command)
    else:
        # Parent process: continuously read and write output.
        try:
            while True:
                try:
                    output = os.read(fd, 1024)
                except OSError:
                    break
                if not output:
                    break
                # Write to sys.stdout to preserve the carriage return behavior.
                sys.stdout.write(output.decode())
                sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error reading from process: {e}")

        # Wait for the child process to finish and capture its exit status.
        _, exit_status = os.waitpid(pid, 0)
        if exit_status != 0:
            logger.error(f"Download of {name} failed with exit status {exit_status}")
        else:
            logger.info(f"Download of {name} completed successfully")
        return exit_status


def download_folder(name, link, dest_folder):
    """
    Download a folder (and its subdirectories) using rsync with overall progress information via a pseudo-terminal.

    This function downloads a folder from the specified rsync URL to the given destination directory.
    It forks a pseudo-terminal so that rsync displays its progress information correctly.
    The output is streamed to sys.stdout in real time.

    Parameters:
        name (str): A descriptive identifier for the folder being downloaded.
        link (str): The rsync URL of the folder to download, e.g.,
                    "rsync://hgdownload.cse.ucsc.edu/goldenPath/mm10/phyloP60way/"
        dest_folder (str): The destination directory where the folder and its contents will be saved.

    Returns:
        int: The exit status of the rsync command. A 0 exit status indicates success.
    """

    # Ensure destination directory exists.
    os.makedirs(dest_folder, exist_ok=True)

    # Construct rsync command:
    # - -a: Archive mode to preserve structure and file properties.
    # - -z: Enable compression during transfer.
    # - --info=progress2: Show overall progress (less verbose than -v).
    command = ["rsync", "-azh", "--info=progress2", link, dest_folder]

    # Fork a pseudo-terminal so that rsync detects a TTY and shows progress.
    pid, fd = pty.fork()
    if pid == 0:
        # In the child process, execute the rsync command.
        os.execvp("rsync", command)
    else:
        # In the parent process, continuously read from the pseudo-terminal and print output.
        try:
            while True:
                try:
                    output = os.read(fd, 1024)
                except OSError:
                    break
                if not output:
                    break
                sys.stdout.write(output.decode())
                sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error reading from process: {e}")

        # Wait for rsync process to complete and capture its exit status.
        _, exit_status = os.waitpid(pid, 0)
        if exit_status != 0:
            logger.error(f"Download of {name} failed with exit status {exit_status}")
        else:
            logger.info(f"Download of {name} completed successfully")
        return exit_status
