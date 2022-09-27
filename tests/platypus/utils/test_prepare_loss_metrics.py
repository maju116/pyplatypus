from pyplatypus.utils.prepare_loss_metrics import prepare_loss_function, prepare_metrics, prepare_loss_and_metrics


def test_prepare_loss_function(mocker, monkeypatch):
    mocker.patch("pyplatypus.utils.prepare_loss_metrics.convert_to_snake_case", return_value="loss_function_name")
    monkeypatch.setattr("pyplatypus.utils.prepare_loss_metrics.segmentation_loss.loss_function_name", "loss_function", raising=False)
    assert prepare_loss_function(loss="loss_function_name", n_class=2) == "loss_function"

def test_prepare_metrics(mocker):
    mocker.patch("pyplatypus.utils.prepare_loss_metrics.prepare_loss_function", return_value="duplicated_metric1")
    assert set(prepare_metrics(
        metrics=["duplicated_metric1", "duplicated_metric1"], n_class=2
        )) == set(["categorical_crossentropy", "duplicated_metric1"])

def test_prepare_loss_and_metrics(mocker):
    script_path = "pyplatypus.utils.prepare_loss_metrics"
    mocker.patch(script_path + ".prepare_loss_function", return_value="loss_function")
    mocker.patch(script_path + ".prepare_metrics", return_value=["metric"])
    assert prepare_loss_and_metrics("loss", ["metric"], n_class=2) == ("loss_function", ["metric"])
