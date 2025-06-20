from bci_aic3.data import BCIDataset
from bci_aic3.models.eegnet import EEGNet
from bci_aic3.paths import (
    CONFIG_DIR,
    LABEL_MAPPING_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REVERSE_LABEL_MAPPING_PATH,
)
from bci_aic3.util import load_model, read_json_to_dict


def make_inference():
    pass


def load_models(ssvep_config, mi_config):
    ssvep_model = load_model(
        model_path=ssvep_config["path"],
        model_class=ssvep_config["class"],
        model_kwargs=ssvep_config["kwargs"],
        device=ssvep_config["device"],
        optim=ssvep_config["optim"],
        learning_rate=ssvep_config["lr"],
    )

    mi_model = load_model(
        model_path=mi_config["path"],
        model_class=mi_config["class"],
        model_kwargs=mi_config["kwargs"],
        device=mi_config["device"],
        optim=mi_config["optim"],
        learning_rate=mi_config["lr"],
    )

    return ssvep_model, mi_model


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
        split="test",
        label_mapping=label_mapping,
    )

    test_ssvep = BCIDataset(
        csv_file="test.csv",
        base_path=RAW_DATA_DIR,
        task_type="MI",
        split="test",
        label_mapping=label_mapping,
    )


if __name__ == "__main__":
    main()
