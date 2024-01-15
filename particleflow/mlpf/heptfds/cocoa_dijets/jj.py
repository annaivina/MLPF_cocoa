from pathlib import Path

import tensorflow as tf
from utils_edm import (
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
    generate_examples,
    split_sample,
)

import tensorflow_datasets as tfds

_DESCRIPTION = """
COCOA EDM4HEP dataset with jj with calorimeter clusters.
  - X: reconstructed tracks and calorimeter clusters, variable number N per event
  - ygen: stable generator particles, zero-padded to N per event
  - ycand: baseline particle flow particles, zero-padded to N per event
"""

_CITATION = """
Pata, Joosep, Wulff, Eric, Duarte, Javier, Mokhtar, Farouk, Zhang, Mengke, Girone, Maria, & Southwick, David. (2023).
Simulated datasets for detector and particle flow reconstruction: CLIC detector (1.1) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.8260741
"""


class CocoaEdmjjClustersPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.7.0")
    RELEASE_NOTES = {
        "0.9.0": "Small stats",
        "1.0.0": "Initial release",
        "1.1.0": "Remove track referencepoint feature",
        "1.2.0": "Keep all interacting genparticles",
        "1.5.0": "Regenerate with ARRAY_RECORD",
        "1.7.0": "this version is for the cocoa dijet samples"
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For the raw input files in ROOT EDM4HEP format, please see the citation above.

    The processed tensorflow_dataset can also be downloaded from:
    FIXME
    """

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(CocoaEdmjjClustersPf, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(
                        shape=(
                            None,
                            max(len(X_FEATURES_TRK), len(X_FEATURES_CL)),
                        ),
                        dtype=tf.float32,
                    ),
                    "ygen": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "file_id": tfds.features.Tensor(shape=(), dtype=tf.int64),  # New field for file_id
                    "event_id": tfds.features.Tensor(shape=(), dtype=tf.int64),  # New field for event_id
                    
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                x_features_track=X_FEATURES_TRK,
                x_features_cluster=X_FEATURES_CL,
                y_features=Y_FEATURES,
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.manual_dir
        return split_sample(Path(path / ""))

    def _generate_examples(self, files):
        return generate_examples(files)
