import logging
import mlflow
import json

from git_utils import get_git_revision_short_hash, get_git_url


def main():
    """Report git information
    """
    info = dict(
        name="mwe-executor",
        git_url = get_git_url(),
        git_hash = get_git_revision_short_hash(),
        type="info"
    )

    logging.info(info)

    with open('output.json', 'w') as output:
        json.dump(info, output)

    mlflow.log_artifact('output.json')

if __name__ == '__main__':
    main()