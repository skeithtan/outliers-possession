"""The learned stage: a numpy port of the network of tools/learned_spike and its two decoders.

LEARNED_STAGE_PROTOCOL.md Sections 1, 2 and 6 define the stage. Config.LEARNED_MODE selects it:

- "off": possession.run() writes the c22 labels and this module loads nothing.
- "model" (candidate M): the labels of the learned ensemble alone. A three-state Viterbi over
  H, A and D decodes each graded block (decode_model()). With Config.LEARNED_STRUCTURE the
  frames before each half's kickoff of games.py are forced to D in that Viterbi, the rule of
  c22 and c23 (pre_kickoff_mask()). With Config.LEARNED_PRIOR_BONUS > 0 the log-probability of
  the rule engine's own class (the labels of LEARNED_MODE "off") gets that bonus on each H, A
  or D frame before the Viterbi (apply_prior()); 0 decodes exactly as before.
- With Config.LEARNED_KICKOFF_TEAM (mode "model", on by default since round 8) the first
  LEARNED_KICKOFF_FLOOR_S of each half from its kickoff frame are forced to the team that takes
  the kickoff: in half 1 the team of the first alive frame of the decode without this rule, in
  half 2 the other team (Law 8). After the floor the Viterbi decides, so the forced team lasts
  until the decoder's first change of team or dead ball (kickoff_teams(), kickoff_team_mask()).
- "hybrid" (candidate H): the alive and dead frames of c22 stay. On each alive run of c22, a
  two-state Viterbi over the H and A probabilities of the ensemble gives the team
  (decode_hybrid()).
- Config.LEARNED_PF "on" (round 6) adds the per-frame member of engine/learned_pf.py in either
  mode: the frame probabilities given to the decoder are LEARNED_PF_WEIGHT x p_pf +
  (1 - LEARNED_PF_WEIGHT) x p_network, rounded again to PROB_DECIMALS (blend_pf(),
  candidate_frame_probabilities()); with a member whose weights file says swap_mode
  "tracking", p_network takes its swapped copies from the swapped tracking, paired, so the whole
  output is exactly team-swap symmetric (predict_ensemble_paired()). "off" (the default)
  computes nothing more and changes no byte.

The network is model.Net of tools/learned_spike: a linear layer with GELU, one residual
temporal convolution (kernel 5, padding 2) with GELU, a two-layer bidirectional GRU and a
linear head with three classes (H, A, D). The input is the 5 fps feature matrix of
engine.learned_features, in the feature version that the weights file records (its width is
the input width of the network). The port computes in float64:

- GELU is the exact form x / 2 (1 + erf(x / sqrt 2)), as torch.nn.GELU computes it. numpy has
  no erf, so _erf() applies math.erf to each element. math.erf is the erf of the C library,
  correct to about one unit in the last place (1e-16 relative).
- The GRU follows the equations of torch.nn.GRU, gate order r, z, n:
  r = sigmoid(W_ir x + b_ir + W_hr h + b_hr), z = sigmoid(W_iz x + b_iz + W_hz h + b_hz),
  n = tanh(W_in x + b_in + r (W_hn h + b_hn)), h' = (1 - z) n + z h. The output of a layer is
  the forward states, then the reverse states. Dropout acts in training only, so inference has
  none.
- The class probabilities are the softmax of the head. The four symmetric copies of the input
  give four probability rows per bin. A swapped copy predicts with H and A exchanged, so the
  port exchanges them back. An ensemble of N models gives 4 N rows per bin. Their mean is the
  probability of the ensemble at 5 fps.
- The port rounds the 5 fps probabilities once to PROB_DECIMALS decimals. A difference in the
  last bits of a matrix product then changes a rounded value only when the value lies that
  close to a rounding boundary. Linear
  interpolation between bin centres then gives a probability for every 25 fps frame, and the
  decoders read these.

The weights file is one .npz that tools/learned_spike/export.py writes from the torch state
dictionaries. Keys: `n_models` (int), `feature_version` (str) and, for model k,
`m{k}/<name>` with the state dictionary names of model.Net (`inp.0.weight`, `conv.weight`,
`gru.weight_ih_l0`, `gru.weight_hh_l0_reverse`, `head.bias` and the others), as float64.

Python 3.9, numpy only. No randomness, no clock, no network.
"""

import math
import os

import numpy as np

from engine import data
from engine import learned_features as lf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODES = ("off", "model", "hybrid")
LABELS = ("H", "A", "D", "X")   # code i is LABELS[i], as in evaluation/possession_metrics.py
H, A, D, X = 0, 1, 2, 3
N_CLASSES = 3
# The four symmetric copies (swap, mirror), in the order of tools/learned_spike/model.py.
SYMMETRIES = ((False, False), (False, True), (True, False), (True, True))
GRU_LAYERS = 2
KERNEL = 5
# The 5 fps ensemble probabilities are rounded to this many decimals before the upsampling
# (LEARNED_STAGE_PROTOCOL.md 2, after the advisor review of 9 November).
PROB_DECIMALS = 4
# The decoders of the spike clip each probability to this floor before the log, so that no log
# is -inf (tools/learned_spike/model.py decode(), hybrid.py). Part of the registered recipe.
PROB_FLOOR = 1e-4
# The hybrid decoder divides the H and A probabilities by their sum, which is at least this
# floor (tools/learned_spike/hybrid.py).
TEAM_SUM_FLOOR = 1e-9
# The two-state Viterbi of the hybrid decoder is the three-state Viterbi with the D state shut:
# log-probability -BLOCKED and switch cost BLOCKED (tools/learned_spike/hybrid.py).
BLOCKED = 1e9


def prior_bonus(cfg):
    """float cfg.LEARNED_PRIOR_BONUS (0.0 when absent). Raise ValueError if negative."""
    bonus = float(getattr(cfg, "LEARNED_PRIOR_BONUS", 0.0))
    if bonus < 0.0 or not math.isfinite(bonus):
        raise ValueError("LEARNED_PRIOR_BONUS must be finite and >= 0, not {!r}".format(bonus))
    return bonus


PF_MODES = ("off", "on")


def pf_settings(cfg):
    """(on, weight, path) of the per-frame member (Config.LEARNED_PF*); ValueError on a bad
    setting. Absent fields mean "off"."""
    mode = getattr(cfg, "LEARNED_PF", "off")
    if mode not in PF_MODES:
        raise ValueError("LEARNED_PF must be one of {}, not {!r}".format(PF_MODES, mode))
    weight = float(getattr(cfg, "LEARNED_PF_WEIGHT", 0.5))
    if not (0.0 <= weight <= 1.0) or not math.isfinite(weight):
        raise ValueError("LEARNED_PF_WEIGHT must lie in [0, 1], not {!r}".format(weight))
    path = getattr(cfg, "LEARNED_PF_WEIGHTS", "")
    return mode == "on", weight, (os.path.join(ROOT, path) if path else "")


def predict_ensemble_paired(nets, feats, feats_swapped):
    """float64 (bins, 3): the network ensemble with exact team-swap symmetry (LEARNED_PF "on").

    The two swapped copies are the features of the tracking with home and away exchanged
    (`feats_swapped`, learned_pf.swapped_tactical()) in place of transform(feats, swap=True),
    and each copy is added to its swapped partner first:
        pair(F, G) = P(F) + S(P(G)),  S exchanging the H and A columns,
        total = sum over models of pair(F, G) + pair(mirror F, mirror G).
    Under a team swap of the input F and G exchange exactly and every pair turns into S(pair)
    bit for bit (x + y = y + x), so the ensemble's H and A exchange exactly. The copies agree
    with transform(F, swap=True) up to the float rounding of the features.
    """
    f = np.asarray(feats, dtype=np.float64)
    g = np.asarray(feats_swapped, dtype=np.float64)
    copies = np.stack([f, lf.transform(f, mirror=True), g, lf.transform(g, mirror=True)])
    total = np.zeros((f.shape[0], N_CLASSES))
    for net in nets:
        p = softmax(forward(net, copies))
        for k in (0, 1):
            total += p[k] + p[k + 2][:, [1, 0, 2]]
    return total / (len(SYMMETRIES) * len(nets))


def frame_probabilities_paired(nets, tactical, game, cfg=None):
    """(bin, frame) probabilities as frame_probabilities(), from predict_ensemble_paired()."""
    from engine import learned_pf
    f = features_for(nets, tactical, game, cfg)
    g = features_for(nets, learned_pf.swapped_tactical(tactical), game, cfg)
    p5 = np.round(predict_ensemble_paired(nets, f, g), PROB_DECIMALS)
    return p5, upsample(p5, int(tactical["n_frames"]))


def blend_pf(net_frame_probs, pf_frame_probs, weight):
    """float64 (N, 3): weight x p_pf + (1 - weight) x p_network, rounded to PROB_DECIMALS.

    The network's bin probabilities are rounded before upsample() (the interpolated frame values
    between bin centres carry more decimals) and learned_pf rounds its own; the blend is rounded
    once more, so a last-bit difference of the products reaches the decoder only on a rounding
    boundary. (So weight 0 gives the network's frame probabilities rounded, not LEARNED_PF
    "off".)
    """
    if net_frame_probs.shape != pf_frame_probs.shape:
        raise ValueError("network {} and per-frame {} probabilities differ in shape".format(
            net_frame_probs.shape, pf_frame_probs.shape))
    w = float(weight)
    return np.round(w * pf_frame_probs + (1.0 - w) * net_frame_probs, PROB_DECIMALS)


def pf_frame_probabilities(cfg, tactical, game, model=None):
    """float64 (N, 3) rounded frame probabilities of the per-frame member for `tactical`."""
    from engine import learned_pf
    if model is None:
        _, _, path = pf_settings(cfg)
        model = learned_pf.load_pf(path)
    return learned_pf.frame_probabilities_pf(tactical, game, model, cfg)


def candidate_frame_probabilities(nets, cfg, tactical, game, pf_model, weight):
    """float64 (N, 3): the frame probabilities of the network + per-frame member that the
    decoder reads with LEARNED_PF on. The member's weights file names the swap mode
    (learned_pf.SWAP_MODES): "columns" keeps the network as without the member
    (frame_probabilities, symmetric copies by transform()); "tracking" takes the network's
    swapped copies from the swapped tracking, paired (frame_probabilities_paired), so that the
    whole output is exactly team-swap symmetric."""
    if pf_model.get("swap_mode", "columns") == "tracking":
        _, p_net = frame_probabilities_paired(nets, tactical, game, cfg)
    else:
        _, p_net = frame_probabilities(nets, tactical, game, cfg)
    return blend_pf(p_net, pf_frame_probabilities(cfg, tactical, game, pf_model), weight)


def kickoff_team_on(cfg):
    """bool Config.LEARNED_KICKOFF_TEAM (False when absent): the kickoff team rule of mode
    "model" (kickoff_teams(), kickoff_team_mask())."""
    return bool(getattr(cfg, "LEARNED_KICKOFF_TEAM", False))


def check_mode(cfg):
    """The LEARNED_MODE of `cfg`. Raise ValueError for a mode outside MODES."""
    mode = cfg.LEARNED_MODE
    if mode not in MODES:
        raise ValueError("LEARNED_MODE must be one of {}, not {!r}".format(MODES, mode))
    return mode


# -- weights -------------------------------------------------------------------------------------

def _layer_names():
    """The state dictionary names of model.Net, in a fixed order."""
    names = ["inp.0.weight", "inp.0.bias", "conv.weight", "conv.bias"]
    for layer in range(GRU_LAYERS):
        for suffix in ("", "_reverse"):
            for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                names.append("gru.{}_l{}{}".format(kind, layer, suffix))
    return names + ["head.weight", "head.bias"]


LAYER_NAMES = tuple(_layer_names())


def _check_shapes(net, where, n_in=lf.NF):
    """Raise ValueError unless the arrays of one model have the shapes of model.Net with
    `n_in` input features."""
    hidden = net["inp.0.weight"].shape[0]
    want = {"inp.0.weight": (hidden, n_in), "inp.0.bias": (hidden,),
            "conv.weight": (hidden, hidden, KERNEL), "conv.bias": (hidden,),
            "head.weight": (N_CLASSES, 2 * hidden), "head.bias": (N_CLASSES,)}
    for layer in range(GRU_LAYERS):
        width = hidden if layer == 0 else 2 * hidden
        for suffix in ("", "_reverse"):
            tail = "_l{}{}".format(layer, suffix)
            want["gru.weight_ih" + tail] = (3 * hidden, width)
            want["gru.weight_hh" + tail] = (3 * hidden, hidden)
            want["gru.bias_ih" + tail] = (3 * hidden,)
            want["gru.bias_hh" + tail] = (3 * hidden,)
    for name in LAYER_NAMES:
        if net[name].shape != want[name]:
            raise ValueError("{}: {} has shape {}, expected {}".format(
                where, name, net[name].shape, want[name]))


def weights_path(cfg):
    """The absolute path of cfg.LEARNED_WEIGHTS. A relative path starts at the repository root."""
    return os.path.join(ROOT, cfg.LEARNED_WEIGHTS)


def load_weights(path):
    """[{name: float64 array}] of the models in a weights file, in the order m0, m1, ...

    Raise ValueError if the feature version is not one that engine.learned_features computes
    (lf.VERSIONS), if a key is missing or extra, or if a shape is not that of model.Net with the
    width of that version. model_version() gives the version back from the loaded models.
    """
    with np.load(path, allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    version = str(stored.get("feature_version", ""))
    if version not in lf.VERSIONS:
        raise ValueError("{}: feature_version {!r}, the engine computes {}".format(
            path, version, list(lf.VERSIONS)))
    n_in = lf.n_features(version)
    n_models = int(stored["n_models"])
    if n_models < 1:
        raise ValueError("{}: n_models must be at least 1".format(path))
    expected = {"n_models", "feature_version"}
    nets = Models(version)
    for k in range(n_models):
        keys = ["m{}/{}".format(k, name) for name in LAYER_NAMES]
        missing = [key for key in keys if key not in stored]
        if missing:
            raise ValueError("{}: missing arrays {}".format(path, missing[:3]))
        expected.update(keys)
        net = {name: np.asarray(stored[key], dtype=np.float64)
               for name, key in zip(LAYER_NAMES, keys)}
        _check_shapes(net, "{} model {}".format(path, k), n_in)
        nets.append(net)
    extra = sorted(set(stored) - expected)
    if extra:
        raise ValueError("{}: unexpected arrays {}".format(path, extra[:3]))
    return nets


class Models(list):
    """The models of a weights file (a list of {name: array}) with the file's feature version
    (`feature_version`): load_weights() returns one. The weights file is authoritative because
    two versions can share a width ("stage50-v2" and "stage50-v2s")."""

    def __init__(self, feature_version, items=()):
        super().__init__(items)
        self.feature_version = feature_version


def model_version(nets):
    """The feature version of loaded models: the version recorded in their weights file
    (Models.feature_version, set by load_weights()); for a plain list of models, the version
    of their input width (lf.version_of_width(): 50 columns are "stage50-v2")."""
    widths = sorted(set(int(net["inp.0.weight"].shape[1]) for net in nets))
    if len(widths) != 1:
        raise ValueError("models of different input widths {}".format(widths))
    recorded = getattr(nets, "feature_version", None)
    if recorded is not None:
        if lf.n_features(recorded) != widths[0]:
            raise ValueError("models of width {} recorded as {}".format(widths[0], recorded))
        return recorded
    return lf.version_of_width(widths[0])


# -- the network ---------------------------------------------------------------------------------

_ERF = np.frompyfunc(math.erf, 1, 1)


def _erf(x):
    """math.erf on every element of a float64 array."""
    return _ERF(x).astype(np.float64)


def gelu(x):
    """Exact GELU, x / 2 (1 + erf(x / sqrt 2)), the default of torch.nn.GELU."""
    return x * 0.5 * (1.0 + _erf(x * math.sqrt(0.5)))


def sigmoid(x):
    """1 / (1 + exp(-x)). A large negative x gives 0 without a warning."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def linear(x, weight, bias):
    """x W^T + b over the last axis, as torch.nn.Linear."""
    return x @ weight.T + bias


def conv1d(x, weight, bias):
    """torch.nn.Conv1d over time with zero padding (kernel - 1) / 2, for x of shape (B, T, C).

    weight has shape (out, in, kernel). Output step t is b + sum_k W[:, :, k] x[t + k - pad].
    """
    kernel = weight.shape[2]
    pad = (kernel - 1) // 2
    steps = x.shape[1]
    padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    out = np.zeros(x.shape[:2] + (weight.shape[0],))
    for k in range(kernel):
        out += padded[:, k:k + steps, :] @ weight[:, :, k].T
    return out + bias


def gru_bidirectional(x, net, layer):
    """One bidirectional layer of torch.nn.GRU for x of shape (B, T, I). Output (B, T, 2 H).

    The forward and the reverse direction step together: the loop runs once over the steps,
    with a batched matrix product for the two directions.
    """
    tail = "_l{}".format(layer)
    w_ih = [net["gru.weight_ih" + tail], net["gru.weight_ih" + tail + "_reverse"]]
    w_hh = [net["gru.weight_hh" + tail], net["gru.weight_hh" + tail + "_reverse"]]
    b_ih = [net["gru.bias_ih" + tail], net["gru.bias_ih" + tail + "_reverse"]]
    b_hh = np.stack([net["gru.bias_hh" + tail], net["gru.bias_hh" + tail + "_reverse"]])
    hidden = w_hh[0].shape[1]
    batch, steps = x.shape[:2]
    # Input terms of all steps at once, time first: (T, B, 3 H) per direction.
    gi = [np.ascontiguousarray(linear(x, w_ih[d], b_ih[d]).transpose(1, 0, 2)) for d in (0, 1)]
    w_hh_t = np.stack([w_hh[0].T, w_hh[1].T])            # (2, H, 3 H)
    b_hh = b_hh[:, None, :]                               # (2, 1, 3 H)
    h = np.zeros((2, batch, hidden))
    out = np.empty((2, steps, batch, hidden))
    for t in range(steps):
        back = steps - 1 - t
        g = np.stack([gi[0][t], gi[1][back]])            # (2, B, 3 H)
        gh = np.matmul(h, w_hh_t) + b_hh
        r = sigmoid(g[..., :hidden] + gh[..., :hidden])
        z = sigmoid(g[..., hidden:2 * hidden] + gh[..., hidden:2 * hidden])
        n = np.tanh(g[..., 2 * hidden:] + r * gh[..., 2 * hidden:])
        h = (1.0 - z) * n + z * h
        out[0, t] = h[0]
        out[1, back] = h[1]
    return np.concatenate([out[0], out[1]], axis=-1).transpose(1, 0, 2)


def forward(net, x):
    """Logits (B, T, 3) of model.Net in eval mode for float64 input x of shape (B, T, NF)."""
    h = gelu(linear(x, net["inp.0.weight"], net["inp.0.bias"]))
    h = h + gelu(conv1d(h, net["conv.weight"], net["conv.bias"]))
    for layer in range(GRU_LAYERS):
        h = gru_bidirectional(h, net, layer)
    return linear(h, net["head.weight"], net["head.bias"])


def softmax(logits):
    """Softmax over the last axis."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def symmetric_copies(feats):
    """float64 (4, bins, NF): the four symmetric copies of a feature matrix, SYMMETRIES order.

    The feature version comes from the width of the matrix (lf.transform()).
    """
    feats = np.asarray(feats)
    return np.stack([lf.transform(feats, swap, mirror)
                     for swap, mirror in SYMMETRIES]).astype(np.float64)


def copy_probabilities(net, copies):
    """float64 (4, bins, 3) class probabilities of one model on the four symmetric copies.

    The H and A columns of a swapped copy are exchanged back, so every row reads H, A, D of
    the match as it is.
    """
    probs = softmax(forward(net, copies))
    for k, (swap, _) in enumerate(SYMMETRIES):
        if swap:
            probs[k] = probs[k][:, [1, 0, 2]]
    return probs


def predict_symmetric(net, feats):
    """float64 (bins, 3): the mean class probabilities of one model over the four copies."""
    return predict_ensemble([net], feats)


def predict_ensemble(nets, feats):
    """float64 (bins, 3): the mean over the 4 N rows of N models and four copies, unrounded.

    The sum runs over the models in file order and over the copies in SYMMETRIES order.
    """
    copies = symmetric_copies(feats)
    total = np.zeros((copies.shape[1], N_CLASSES))
    for net in nets:
        for p in copy_probabilities(net, copies):
            total += p
    return total / (len(SYMMETRIES) * len(nets))


def bin_probabilities(nets, feats):
    """float64 (bins, 3) ensemble probabilities at 5 fps, rounded to PROB_DECIMALS."""
    return np.round(predict_ensemble(nets, feats), PROB_DECIMALS)


def upsample(probs, n_frames, bin_size=lf.BIN):
    """Linear interpolation of bin probabilities (at the bin centres) onto every frame."""
    centres = np.arange(probs.shape[0]) * bin_size + bin_size // 2
    frames = np.arange(n_frames)
    out = np.empty((n_frames, probs.shape[1]))
    for k in range(probs.shape[1]):
        out[:, k] = np.interp(frames, centres, probs[:, k])
    return out


def features_for(nets, tactical, game, cfg=None):
    """float32 (bins, NF) 5 fps features of one match in the feature version of the models
    (model_version()); "stage50-v2" calls lf.frame_features() exactly as before the other
    versions existed."""
    version = model_version(nets)
    if version == lf.FEATURE_VERSION:
        f25 = lf.frame_features(tactical, game, cfg=cfg)
    else:
        f25 = lf.frame_features(tactical, game, cfg=cfg, version=version)
    return lf.downsample(f25)


def frame_probabilities(nets, tactical, game, cfg=None):
    """(bin probabilities, frame probabilities) of one match, float64 (bins, 3) and (N, 3).

    The bin probabilities are rounded to PROB_DECIMALS (bin_probabilities()). The frame
    probabilities interpolate them onto every frame. `tactical` is the tracking the features
    read: possession.engine_tactical(game) in the engine (apply()). The features are those of
    the version of the models (features_for()).
    """
    feats = features_for(nets, tactical, game, cfg)
    p5 = bin_probabilities(nets, feats)
    return p5, upsample(p5, int(tactical["n_frames"]))


# -- decoders ------------------------------------------------------------------------------------

def viterbi(logp, cost_ha, cost_ad):
    """Best state path (int8 codes 0 H, 1 A, 2 D) for (N, 3) log-probabilities.

    A change between H and A costs cost_ha. A change between an alive state and D costs
    cost_ad. On a tie the lower state code wins (tools/learned_spike/model.py viterbi()).
    """
    n = logp.shape[0]
    trans = np.array([[0.0, cost_ha, cost_ad],
                      [cost_ha, 0.0, cost_ad],
                      [cost_ad, cost_ad, 0.0]])
    back = np.empty((n, 3), dtype=np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        cand = score[:, None] - trans
        arg = cand.argmax(axis=0)
        back[t] = arg
        score = cand[arg, (0, 1, 2)] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def team_viterbi(logp2, cost):
    """Two-state path (0 H, 1 A) for (N, 2) log-probabilities, with switch cost `cost`."""
    lp = np.column_stack([logp2, np.full(len(logp2), -BLOCKED)])
    return viterbi(lp, cost, BLOCKED)


def pre_kickoff_mask(n_frames, blocks, kickoffs):
    """bool (n_frames,): the frames of each graded block before its kickoff.

    The rule of possession.dead_intervals() and possession.decode_blocks() (c22, c23): in block
    (half, first, last), with kick = kickoffs[half], the frames [first, kick) are dead ball when
    first < kick <= last. `kickoffs` is data.kickoff_frames(game), read from games.py.
    """
    mask = np.zeros(int(n_frames), dtype=bool)
    for half, first, last in blocks:
        kick = kickoffs.get(half)
        if kick is not None and first < kick <= last:
            mask[first:kick] = True
    return mask


def log_emissions(frame_probs):
    """float64 (N, 3) natural-log emission scores of decode_model(): log max(p, PROB_FLOOR)."""
    return np.log(np.clip(frame_probs, PROB_FLOOR, 1.0))


def apply_prior(logp, rule_codes, bonus):
    """The rule-engine prior (Config.LEARNED_PRIOR_BONUS) on (N, 3) log-probabilities.

    On every frame where `rule_codes` (int codes 0 H, 1 A, 2 D, 3 X, or an H/A/D/X label
    array, one entry per frame of `logp`) is H, A or D, `bonus` (natural-log units) is added to
    the log-probability of that class. X frames get nothing. bonus == 0 returns `logp` itself,
    untouched. bonus > 0 returns a new array and leaves `logp` as it was. A negative bonus, or
    a rule array of another length, raises ValueError.
    """
    bonus = float(bonus)
    if bonus < 0.0 or not math.isfinite(bonus):
        raise ValueError("the prior bonus must be finite and >= 0, not {!r}".format(bonus))
    if bonus == 0.0:
        return logp
    codes = np.asarray(rule_codes)
    if codes.dtype.kind in "USO":
        codes = labels_to_codes(codes)
    codes = codes.astype(np.int64)
    if codes.shape != (logp.shape[0],):
        raise ValueError("rule codes have shape {}, the decoder has {} frames".format(
            codes.shape, logp.shape[0]))
    if ((codes < H) | (codes > X)).any():
        raise ValueError("rule codes outside 0..3")
    out = np.array(logp, dtype=np.float64, copy=True)
    frames = np.flatnonzero(codes != X)
    out[frames, codes[frames]] += bonus
    return out


def kickoff_teams(codes, blocks, kickoffs):
    """{half: code}: the team that takes each half's kickoff (Config.LEARNED_KICKOFF_TEAM).

    Half 1: the team of the first H or A frame of `codes` (the decoded labels without this
    rule) at or after the half 1 kickoff frame, in the half 1 block. Half 2: the other team
    (Law 8: the team that did not take the kickoff of the first half takes the kickoff of the
    second half). {} when the half 1 block has no kickoff or no alive frame from it.
    """
    kick = kickoffs.get(1)
    block = next(((first, last) for half, first, last in blocks if half == 1), None)
    if kick is None or block is None or not block[0] <= kick <= block[1]:
        return {}
    seg = np.asarray(codes[kick:block[1] + 1])
    alive = np.flatnonzero(seg < D)
    if alive.size == 0:
        return {}
    first_team = int(seg[alive[0]])
    return {1: first_team, 2: A if first_team == H else H}


def kickoff_team_mask(n_frames, blocks, kickoffs, teams, floor_frames):
    """int8 (n_frames,): the forced team code on the first `floor_frames` frames from each
    half's kickoff frame (inside its block), -1 elsewhere."""
    forced = np.full(int(n_frames), -1, dtype=np.int8)
    for half, first, last in blocks:
        kick = kickoffs.get(half)
        if kick is None or half not in teams or not first <= kick <= last:
            continue
        forced[kick:min(kick + int(floor_frames), last + 1)] = teams[half]
    return forced


def decode_model(frame_probs, blocks, cost_ha, cost_ad, forced_dead=None, rule_codes=None,
                 prior_bonus=0.0, forced_team=None):
    """int8 codes of candidate M: Viterbi in each graded block, X outside the blocks.

    `forced_dead` (bool (N,) or None) is the match-structure step (Config.LEARNED_STRUCTURE):
    on its frames the H and A states get log-probability -inf, so the path is D there, as the
    forced frames of engine.decoder are in c22 and c23. None decodes the network alone.
    `prior_bonus` > 0 (Config.LEARNED_PRIOR_BONUS) adds that bonus to the log-probability of
    the rule engine's class on each H, A or D frame of `rule_codes` (apply_prior()), before
    the forcing and the Viterbi. 0 leaves the emission scores untouched.
    `forced_team` (int (N,) or None, Config.LEARNED_KICKOFF_TEAM): on a frame with a code 0 or
    1 every other state gets log-probability -inf, so the path takes that team there
    (kickoff_team_mask()). None or -1 everywhere changes nothing.
    """
    n = frame_probs.shape[0]
    out = np.full(n, X, dtype=np.int8)
    logp = log_emissions(frame_probs)
    if prior_bonus != 0.0:
        if rule_codes is None:
            raise ValueError("a prior bonus needs the rule engine's codes")
        logp = apply_prior(logp, rule_codes, prior_bonus)
    if forced_dead is not None:
        forced_dead = np.asarray(forced_dead, dtype=bool)
        logp[forced_dead, H] = -np.inf
        logp[forced_dead, A] = -np.inf
    if forced_team is not None:
        forced_team = np.asarray(forced_team, dtype=np.int64)
        rows = np.flatnonzero(forced_team >= 0)
        keep = logp[rows, forced_team[rows]]
        logp[rows, :] = -np.inf
        logp[rows, forced_team[rows]] = keep
    for _, first, last in blocks:
        out[first:last + 1] = viterbi(logp[first:last + 1], cost_ha, cost_ad)
    return out


def alive_runs(codes):
    """[(start, stop)] half-open runs of H or A frames in an int code array."""
    alive = np.isin(codes, (H, A))
    padded = np.concatenate(([False], alive, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def decode_hybrid(frame_probs, base_codes, cost_ha):
    """int8 codes of candidate H: the team of each alive run of `base_codes` from the model.

    D and X frames of `base_codes` stay. On each alive run, the H and A probabilities divided
    by their sum go through team_viterbi() with switch cost cost_ha.
    """
    out = np.asarray(base_codes, dtype=np.int8).copy()
    pair = frame_probs[:, :2]
    team = pair / np.maximum(pair.sum(axis=1, keepdims=True), TEAM_SUM_FLOOR)
    logt = np.log(np.clip(team, PROB_FLOOR, 1.0))
    for start, stop in alive_runs(out):
        out[start:stop] = team_viterbi(logt[start:stop], cost_ha)
    return out


def labels_to_codes(labels):
    """int8 codes of an H, A, D, X label array."""
    labels = np.asarray(labels).astype(str)
    codes = np.full(len(labels), -1, dtype=np.int8)
    for code, label in enumerate(LABELS):
        codes[labels == label] = code
    if (codes < 0).any():
        raise ValueError("labels outside {}".format(list(LABELS)))
    return codes


def codes_to_labels(codes):
    """The "<U1" label array of int codes."""
    return np.asarray(LABELS, dtype="<U1")[np.asarray(codes, dtype=np.int64)]


# -- the stage -----------------------------------------------------------------------------------

def apply(cfg, game, blocks, base_labels=None, return_probs=False, tactical=None):
    """Labels of the learned stage for one match, by cfg.LEARNED_MODE ("model" or "hybrid").

    `base_labels` are the c22 labels of possession.run(). The hybrid mode needs them. The model
    mode reads them only when cfg.LEARNED_PRIOR_BONUS > 0 (the rule-engine prior of
    decode_model()), and then needs them. With return_probs=True also return the frame
    probabilities.
    `tactical` is the ATD the features read. possession.run() passes
    possession.engine_tactical(), the ATD of c22 with the C9 goalkeeper guard
    (LEARNED_STAGE_PROTOCOL.md 1, LEARNED_STAGE_PROTOCOL_R2.md 9, D1). None reads
    data.load_tactical(game).
    """
    mode = check_mode(cfg)
    if mode == "off":
        raise ValueError("apply() needs LEARNED_MODE model or hybrid")
    nets = load_weights(weights_path(cfg))
    if tactical is None:
        tactical = data.load_tactical(game)
    pf_on, pf_weight, pf_path = pf_settings(cfg)
    if pf_on:
        from engine import learned_pf
        probs = candidate_frame_probabilities(nets, cfg, tactical, game,
                                              learned_pf.load_pf(pf_path), pf_weight)
    else:
        _, probs = frame_probabilities(nets, tactical, game, cfg)
    if mode == "model":
        forced = None
        if getattr(cfg, "LEARNED_STRUCTURE", False):
            forced = pre_kickoff_mask(len(probs), blocks, data.kickoff_frames(game))
        bonus = prior_bonus(cfg)
        if bonus == 0.0:
            codes = decode_model(probs, blocks, float(cfg.LEARNED_COST_HA),
                                 float(cfg.LEARNED_COST_AD), forced)
        else:
            if base_labels is None:
                raise ValueError("LEARNED_PRIOR_BONUS > 0 needs the rule engine's labels")
            codes = decode_model(probs, blocks, float(cfg.LEARNED_COST_HA),
                                 float(cfg.LEARNED_COST_AD), forced,
                                 rule_codes=labels_to_codes(base_labels), prior_bonus=bonus)
        if kickoff_team_on(cfg):
            kickoffs = data.kickoff_frames(game)
            teams = kickoff_teams(codes, blocks, kickoffs)
            floor = int(round(float(cfg.LEARNED_KICKOFF_FLOOR_S) * lf.FPS))
            forced_team = kickoff_team_mask(len(probs), blocks, kickoffs, teams, floor)
            rule = labels_to_codes(base_labels) if bonus else None
            codes = decode_model(probs, blocks, float(cfg.LEARNED_COST_HA),
                                 float(cfg.LEARNED_COST_AD), forced, rule_codes=rule,
                                 prior_bonus=bonus, forced_team=forced_team)
    else:
        if base_labels is None:
            raise ValueError("the hybrid mode needs the c22 labels")
        codes = decode_hybrid(probs, labels_to_codes(base_labels),
                              float(cfg.LEARNED_COST_HA_HYBRID))
    labels = codes_to_labels(codes)
    if return_probs:
        return labels, probs
    return labels
