"""The PFF FC references of round 2 and their frozen md5s. Not part of the submission.

LEARNED_STAGE_PROTOCOL_R2.md Section 8 lists the md5 of the reference CSV of each of the 43 PFF
training matches and of the 15 sealed matches (`~/mtah2027/refs/pff/<id>/PFF-<id>_reference.csv`,
built by tools/pff_bundle.py). REFERENCE_MD5 below is that table, copied from the frozen
protocol. Nothing that builds a training cache or trains may run if one file differs:
check_references() raises SystemExit and names each difference.

check_references() reads the bytes of each file for the md5 only. It parses no file, so it
decodes nothing of the sealed set S. reference() gives the path and md5 of a training match
after the same check. Only the V1 tool of evaluate.py calls reference() with sealed=True.

Standard library only (the fold tests run without numpy). Python 3.9.
"""

import hashlib
import os

from tools.learned_spike import folds

DEFAULT_REFS_ROOT = os.path.join(os.path.expanduser("~"), "mtah2027", "refs")

# LEARNED_STAGE_PROTOCOL_R2.md Section 8 (frozen at 6fdded5).
REFERENCE_MD5 = {
    "PFF-3813": "bbbd369e8a6de56fd8ea2e961988bdfb",
    "PFF-3814": "4841e614d125da451f5079f5bbb0f357",
    "PFF-3815": "54ec90d04aa7ea8d3d5cbe6d366d1c6e",
    "PFF-3816": "a6be3a4d7748c4603c6ed383addc0e2d",
    "PFF-3817": "5ae3918d906ffa456b539ced6376eb83",
    "PFF-3818": "f65299ec356fd4c63c079191efced4ae",
    "PFF-3819": "2137c2ffc6666d976c830b81df91ef36",
    "PFF-3820": "8ccd7a00500f28c4e14f771cb1cd058a",
    "PFF-3821": "98487194da6996a51fbd03ebf20edfb6",
    "PFF-3822": "0a08348d3ac08e29510ff73241d1adb4",
    "PFF-3823": "77ecc36be801e288e39bdef161269454",
    "PFF-3824": "9c80775b43273e2d7303ee145eb1acda",
    "PFF-3825": "34c9135b8b4f1e1bbc22921ebf880d68",
    "PFF-3826": "c12ae5847efd347a92effa51cf39ac0f",
    "PFF-3827": "1a7e08f6f182a13a8b4e842ac4131d70",
    "PFF-3828": "467cade997df9a481547ba90e484f275",
    "PFF-3829": "adc9b2ec8fd33522a7ad5686b59c0e34",
    "PFF-3830": "cc66af725c155388717616cf0dcfc4fe",
    "PFF-3831": "bbbc2233b1bc9f50fc2e009dc2efa95e",
    "PFF-3832": "2ff01b76ee50d86d31e516dd53ae55ff",
    "PFF-3833": "0ce7512a9175f46e4e9344d4693d81b6",
    "PFF-3834": "d51a5ee51691696ba82f817a96ea06fe",
    "PFF-3835": "2267f91c7da7d4c540e5456941fb24a6",
    "PFF-3836": "0a8b7a007ed59231c4ff71b61c63d6ea",
    "PFF-3837": "0d882da9f83100e16c9d64be6f0e628d",
    "PFF-3838": "8d5f3ab6089e9e90d650287d7cbbe290",
    "PFF-3839": "e07bba162031e2ee9f209f6cbc375848",
    "PFF-3841": "89879d9b6675af1f43e408ce13eecfea",
    "PFF-3842": "2565207cfa2cb484af850b89b1b29088",
    "PFF-3843": "d583820c975a908892c62321125108d0",
    "PFF-3844": "69fc7e2caf5485f1a55990bac20656eb",
    "PFF-3846": "603cf1c350c9f5d8047a93380ce0ae0a",
    "PFF-3847": "2d37c59bc07b2675ee9f18c6131aaf55",
    "PFF-3848": "288112a0b033276ce23881ca5e6f6217",
    "PFF-3849": "f09b9220d49bf698a44f60882fe9681a",
    "PFF-3850": "e768de99119c7521d235f1aac43599f3",
    "PFF-3851": "438131892aae9269cc72c03d44c35498",
    "PFF-3852": "726de40465399c4d09e96380a3a85218",
    "PFF-3853": "f2e288b50c058f29801b418649c34036",
    "PFF-3854": "3bae971c109040d2bd3be3b05053c17c",
    "PFF-3855": "76ad5c1c1964e77499f3920b0341355f",
    "PFF-3856": "9ac7a84ccf91b216aedc71b7808a6b1d",
    "PFF-3857": "28a0eec63b73a9512913d33d43e8ac37",
    "PFF-3858": "01a475ae9549a0733351c90765fb6ac1",
    "PFF-3859": "96178b0fe3046b498fe372b763dd8bbb",
    "PFF-10503": "58f98e5cbbb404d8d6ef69e0ba330e47",
    "PFF-10504": "d27d7dfd23677b9ee3c69312c7651b8c",
    "PFF-10505": "bbb29aabff21b16cc1b3148ccf1e840e",
    "PFF-10506": "844bb3114e5e7b983b52b8d5a4ac9df4",
    "PFF-10507": "fa17fdab5a5544bcec6acde22079b3bc",
    "PFF-10508": "06f1f7adb34c3cc4fee570a4f30ca0e0",
    "PFF-10509": "92109eddfbaaa7343d469ca346c61def",
    "PFF-10510": "9a3409fe07f850db006482469b5f1630",
    "PFF-10511": "8cb8d67996b96d31e2f1eab5744711e2",
    "PFF-10513": "1e81f93419069b797bc3a4f83c4e482e",
    "PFF-10514": "6feaed6fb7e1d4aa028ca5aa2e3a5a42",
    "PFF-10515": "3d8d1ed1118a514472b19715c7b68eb0",
    "PFF-10516": "06d0601f0f1886fcedce2ed4882ecb1a",
}


def match_id(game):
    """The numeric PFF game id of a `PFF-<id>` name, as a string."""
    if not game.startswith("PFF-"):
        raise ValueError("{} is not a PFF match".format(game))
    return game[len("PFF-"):]


def reference_path(game, refs_root=DEFAULT_REFS_ROOT):
    """The reference CSV of a PFF match (tools/pff_bundle.py layout). The file can be missing."""
    mid = match_id(game)
    return os.path.join(os.path.expanduser(refs_root), "pff", mid,
                        "PFF-{}_reference.csv".format(mid))


def md5_of(path):
    """The hex md5 of the bytes of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def differences(games, refs_root=DEFAULT_REFS_ROOT):
    """[text] of each match whose reference is missing or whose md5 differs from the table."""
    out = []
    for game in games:
        expected = REFERENCE_MD5.get(game)
        if expected is None:
            out.append("{}: no frozen md5 (R2 Section 8)".format(game))
            continue
        path = reference_path(game, refs_root)
        if not os.path.isfile(path):
            out.append("{}: {} is missing".format(game, path))
            continue
        digest = md5_of(path)
        if digest != expected:
            out.append("{}: md5 {} differs from the frozen {}".format(game, digest, expected))
    return out


def check_references(refs_root=DEFAULT_REFS_ROOT, games=None):
    """Raise SystemExit if a PFF reference md5 differs from R2 Section 8.

    The default checks every match of the table: the 43 training matches and the 15 sealed
    matches. The check hashes bytes only and parses nothing.
    """
    names = sorted(REFERENCE_MD5) if games is None else list(games)
    bad = differences(names, refs_root)
    if bad:
        raise SystemExit("PFF references differ from LEARNED_STAGE_PROTOCOL_R2.md Section 8; "
                         "no cache build and no training:\n  " + "\n  ".join(bad))
    return len(names)


def reference(game, refs_root=DEFAULT_REFS_ROOT, sealed=False):
    """(path, md5) of the reference of a PFF training match, after the md5 check.

    A sealed match is allowed only with sealed=True (the V1 tool). A pilot or unknown match
    raises.
    """
    if sealed:
        folds.check_sealed_v1(game)
    else:
        folds.assert_not_sealed([game])
        if game not in folds.PFF_TRAIN:
            raise SystemExit("{} is not a PFF training match".format(game))
    check_references(refs_root, [game])
    return reference_path(game, refs_root), REFERENCE_MD5[game]
