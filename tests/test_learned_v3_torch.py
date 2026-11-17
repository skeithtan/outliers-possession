"""The numpy network of engine/learned.py against torch (model.Net) for every feature version.

Needs torch (skipped without it): run with the training interpreter. The weights are random,
the input is a random matrix of the version's width. The same bounds as tests/test_learned.py:
1e-9 in float64, 1e-5 against the float32 path of the spike.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine import learned  # noqa: E402
from engine import learned_features as lf  # noqa: E402
from tools.learned_spike import export, model, train  # noqa: E402


@pytest.mark.parametrize("version", lf.VERSIONS)
def test_numpy_inference_matches_torch(version, tmp_path):
    torch.manual_seed(7)
    nets_t = [model.net_for_version(version, hidden=16) for _ in range(2)]
    with torch.no_grad():
        for p in nets_t[1].parameters():
            p.mul_(2.0)
    for net in nets_t:
        net.eval()
        # the width gives the version, except stage50-v2s/-v2t (the width of stage50-v2)
        assert net.n_in == lf.n_features(version)
        assert net.feature_version == lf.version_of_width(net.n_in)
    path = os.path.join(str(tmp_path), "w.npz")
    export.write(path, [n.state_dict() for n in nets_t], version)
    nets = learned.load_weights(path)
    assert learned.model_version(nets) == version
    x = np.random.RandomState(3).normal(0, 1, (60, lf.n_features(version))).astype(np.float32)
    copies = learned.symmetric_copies(x)
    total = np.zeros((60, 3))
    for net_t, net in zip(nets_t, nets):
        net64 = model.net_for_state(net_t.state_dict()).double()
        net64.load_state_dict({k: v.double() for k, v in net_t.state_dict().items()})
        net64.eval()
        with torch.no_grad():
            logits_t = net64(torch.from_numpy(copies)).numpy()
        assert np.abs(learned.forward(net, copies) - logits_t).max() < 1e-9
        p_np = learned.predict_symmetric(net, x)
        assert np.abs(p_np - train.predict_p5(net_t, x)).max() < 1e-9
        assert np.abs(p_np - model.predict_symmetric(net_t, x)).max() < 1e-5
        total += p_np
    assert np.abs(learned.predict_ensemble(nets, x) - total / 2).max() < 1e-12
    # Symmetry of the ensemble for this version's tables.
    p = learned.predict_ensemble(nets, x)
    assert np.abs(learned.predict_ensemble(nets, lf.transform(x, True)) - p[:, [1, 0, 2]]).max() \
        < 1e-12


def test_train_builds_a_net_of_the_configured_width(tmp_path):
    old = train.FEATURE_VERSION
    try:
        train.configure_features(lf.VERSION_V3)
        assert train.n_inputs() == 55 and train.count_params() > 0
        net = model.Net(n_in=train.n_inputs())
        path = os.path.join(str(tmp_path), "net.pt")
        torch.save(net.state_dict(), path)
        assert train.load_net(path).n_in == 55
    finally:
        train.configure_features(old)
