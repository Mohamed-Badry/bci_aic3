from bci_aic3.data import BCIDataset
from bci_aic3.models.eegnet import EEGNet
from bci_aic3.util import read_json_to_dict
from bci_aic3.paths import (
    MODELS_DIR,
    CONFIG_DIR,
    RAW_DATA_DIR,
    LABEL_MAPPING_PATH,
    REVERSE_LABEL_MAPPING_PATH,
    PROJECT_ROOT,
)


def make_inference():
    pass


def load_models(ssvep_path, ssvep_class, mi_path, mi_class):
    ssvep_model = load_ssvep(ssvep_path, ssvep_class)
    mi_model = load_mi(mi_path, mi_class)

    return ssvep_model, mi_model


def load_mi(weights_path, model_class):
    pass


def load_ssvep(weights_path, model_class):
    pass


def predict_ssvep(ssvep_model, test_ssvep):
    pass


def predict_mi(mi_model, test_mi):
    pass


def main():
    label_mapping = read_json_to_dict(LABEL_MAPPING_PATH)

    test_mi = BCIDataset(
        csv_file="test.csv",
        base_path=RAW_DATA_DIR,
        task_type="MI",
        label_mapping=label_mapping,
    )

    test_ssvep = BCIDataset(
        csv_file="test.csv",
        base_path=RAW_DATA_DIR,
        task_type="MI",
        label_mapping=label_mapping,
    )


if __name__ == "__main__":
    main()
