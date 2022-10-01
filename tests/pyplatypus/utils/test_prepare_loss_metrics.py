from pyplatypus.utils.prepare_loss_metrics import(
    prepare_loss_function, prepare_metrics, prepare_loss_and_metrics, prepare_optimizer, prepare_callback, prepare_callbacks_list
    )


class mocked_optimizer_spec:

    name = "optimizer_name"

    def dict():
        return {"name": "optimizer_name"}
    
class mocked_topt_optimizer:
    
    def __init__(self, name: str):
        self.name = name



class mocked_callback_spec:

    name = "CallbackName"

    def dict():
        return {"name": "callback_name"}


class mocked_tcb_callback:
    
    def __init__(self, input_dict: dict):
        self.name = input_dict.get("name")
    

def test_prepare_loss_function(mocker, monkeypatch):
    mocker.patch("pyplatypus.utils.prepare_loss_metrics.convert_to_snake_case", return_value="loss_function_name")
    monkeypatch.setattr("pyplatypus.utils.prepare_loss_metrics.SegmentationLoss.loss_function_name", "loss_function", raising=False)
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


def test_prepare_optimizer(monkeypatch):
    monkeypatch.setattr("pyplatypus.utils.prepare_loss_metrics.TOPT.optimizer_name", mocked_topt_optimizer, raising=False)
    input_spec = mocked_optimizer_spec
    optimizer = prepare_optimizer(optimizer=input_spec)
    assert optimizer.name == "optimizer_name"


def test_prepare_callback(monkeypatch):
    monkeypatch.setattr("pyplatypus.utils.prepare_loss_metrics.TCB.CallbackNameExtension", mocked_tcb_callback, raising=False)
    input_spec = mocked_callback_spec
    callback = prepare_callback(callback=input_spec)
    assert callback.name == "callback_name"


def test_prepare_callbacks_list(mocker):
    mocker.patch("pyplatypus.utils.prepare_loss_metrics.prepare_callback", return_value="callback")
    assert prepare_callbacks_list(["callback_spec", "callback_spec"]) == ["callback", "callback"]

