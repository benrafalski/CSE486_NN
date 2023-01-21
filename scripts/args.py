from argparse import ArgumentParser

def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utilty that will spawn up "
        "parties for MPC scripts on AWS"
    )

    parser.add_argument(
        "-t",
        "--test",
        type=int,
        default=3000,
        help="Testing size used for training",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of rounds used for training",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=10,
        help="Batch size used for training",
    )

    parser.add_argument(
        "-l",
        "--learning",
        type=int,
        default=5,
        help="Learning rate used for training",
    )

    return parser.parse_args()