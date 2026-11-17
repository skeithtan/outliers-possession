"""Every threshold, weight and offset of the engine, in one frozen dataclass (PLAN.md 3, 6).

Rules (team rules, PLAN.md 5.5 / 6):
  * Every field carries a provenance comment: a PLAN.md section, a task card, or a forensics case.
  * Units are metres, seconds, m/s, degrees and frames, as the field name or comment states.
    Velocity steps are in m/s over k frames (never m/s^2), headings in degrees (P0-B card).
  * Nothing match-specific: no frame numbers, offsets or team names (those live in tests/).
  * Constants of later cards (P1-A events, P1-B dead ball, P1-C decoder, P2 semantics) are
    declared here with the initial values their cards give, even before the module exists, so
    the ledger hash covers them from the start.

    cfg = Config.default()
    variant = dataclasses.replace(cfg, DV_TOUCH=2.0)
    cfg.config_hash()   # short, stable sha1 for experiments.md
    load(path)          # JSON of {FIELD: value} overrides on top of the defaults
"""

import dataclasses
import hashlib
import json

CONFIG_HASH_LENGTH = 10   # hex characters of the sha1 written to the ledger


@dataclasses.dataclass(frozen=True)
class Config:
    # -- ball cleaning and kinematics: engine/tracks.py (P0-B; PLAN.md 1.1, 3.2) ---------------
    # A frame displaced > 3 m from the previous accepted frame (75 m/s at 25 fps, beyond any shot)
    # is a teleport: 181 consecutive-frame jumps > 3 m in the match (PLAN 1.1, Appendix B).
    BALL_MAX_JUMP_M: float = 3.0
    # Visible runs shorter than this are isolated detections, dropped (PLAN 3.2).
    BALL_MIN_RUN: int = 3
    # Window of the velocity step dv_k and heading change, frames (PLAN 3.2: k = 3).
    DV_K: int = 3
    # Below this ball speed (m/s) the direction of motion is treated as meaningless and the
    # heading change is NaN: a (near-)stationary ATD ball's direction is tracker noise (PLAN 1.1,
    # 13755-13800 jitters at up to ~2 m/s, which this floor does NOT fully remove; consumers
    # also have `speed`). A passed or rolling ball is well above it. Added by P0-B (not in the card).
    HEADING_MIN_SPEED: float = 1.0
    # Ball faster than this is suspected airborne (PLAN 3.2; P0-B card: 15 m/s).
    AIRBORNE_SPEED: float = 15.0
    # Rolling-inconsistent deceleration: speed falls by more than this fraction ...
    AIRBORNE_DECEL_FRAC: float = 0.4          # P0-B card: "> 40%"
    # ... within this many frames ...
    AIRBORNE_DECEL_FRAMES: int = 3            # P0-B card: "within 3 frames"
    # ... with no player within this distance (m) over the window (else it is a touch).
    AIRBORNE_CLEAR_M: float = 2.0             # P0-B card: "no player within 2 m"
    # The deceleration clause only applies from this starting speed (m/s): near rest a 40% drop
    # is tracker jitter (PLAN 1.1, 13755-13800: up to ~2 m/s jitter on a still ball), while a
    # rolling ball loses ~0.06 m/s in 3 frames to friction (P1-B: k = 0.5 m/s^2). Added by P0-B.
    AIRBORNE_DECEL_MIN_SPEED: float = 5.0

    # -- per-frame player features: engine/tracks.py (P0-B; PLAN.md 1.2, 3.2) ------------------
    # Player counts within two radii of the ball, m (P0-B card: nT_15 / nT_30).
    N_RADIUS_TIGHT_M: float = 1.5             # = R_TOUCH (PLAN 3.3)
    N_RADIUS_WIDE_M: float = 3.0              # PLAN 3.2: r in {1.5, 3}
    # A team with more visible tracks than this is over-counted (PLAN 1.2: > 11 "home" tracks on
    # 8.1% of graded frames, > 11 "away" on 4.7%).
    MAX_TEAM_TRACKS: int = 11
    # Slot confidence factor in over-counted frames (P0-B card).
    OVERCOUNT_CONF_FACTOR: float = 0.5
    # Tracklets (maximal visible runs of a slot) shorter than this are suspect: 2 s at 25 fps
    # (PLAN 3.2 "tracklets shorter than 2 s"; PLAN 1.2: 2,211 track births).
    MIN_TRACKLET_FRAMES: int = 50
    # Slot confidence factor on short tracklets (P0-B card).
    SHORT_TRACKLET_CONF_FACTOR: float = 0.5
    # Goalkeeper slot-swap guard, tracks.guard_goalkeeper_swaps (C9; RESEARCH-2 1d, M14). The
    # tracker can carry the slot of one goalkeeper onto the body of the other goalkeeper.
    # Penalty area size, m (Laws of the Game, Law 1: 16.5 m deep, 40.32 m wide).
    PENALTY_AREA_DEPTH_M: float = 16.5
    PENALTY_AREA_WIDTH_M: float = 40.32
    # The goalkeeper-like slot of a team spends at least this share of its visible frames in its
    # own penalty area. M14: the winners have 0.86 to 0.96, every other slot with more than 60 s
    # visible has 0.29 or less.
    GK_OWN_BOX_SHARE: float = 0.5
    # The goalkeeper-like slot has at least this many frames in its own penalty area: 60 s at
    # 25 fps. A short visit to the box does not make a slot a goalkeeper (C9 brief).
    GK_MIN_OWN_BOX_FRAMES: int = 1500
    # A slot is frozen when it is visible, has no ATD speed and moves less than FROZEN_MOVE_M
    # (m) over FROZEN_WIN frames (1 s). A live player moves more than this: walking at 1 m/s
    # gives 1 m. M14: 5.3% of visible player frames, 98% of them on the goalkeeper slots.
    FROZEN_WIN: int = 25
    FROZEN_MOVE_M: float = 0.25
    # Largest distance, m, from the live goalkeeper-like slot beyond its half-way line to the
    # frozen goalkeeper-like slot of the other team. M14: p50 3.1 to 6.2 m, 15 m or less on 90%
    # to 100% of the frames. Public cases 11708 (3.8 m) and 15654 (4.8 to 5.9 m).
    GK_SWAP_R: float = 15.0

    # -- events: engine/events.py (P1-A; PLAN.md 1.3, 3.3) -------------------------------------
    # Touch = local max of dv_3 above this (m/s over 3 frames) with a player within R_TOUCH from
    # the contact to the end of the dv_3 window (TOUCH_ONSET_MAX_FRAMES). Set from the full-match
    # rate per visible-ball minute (graded frames, 62.8 visible-ball minutes), not from the public
    # flips (P1-A card, PLAN 1.1, 3.3). Measured touch rows (touch + release + contested):
    #   DV_TOUCH 2.0 -> 7.1/min, 1.5 -> 11.8/min, 1.0 -> 17.4/min
    #   (6.1, 10.4, 15.5/min after the airborne rule and the dead-ball suppression).
    # The plausible band is 15-50/min, and only 1.0 is inside it. PLAN 1.1 puts the lower limit
    # at ~1 m/s, where tracker jitter starts (a still ATD ball jitters up to ~2 m/s, 13755-13800),
    # so we do not go below 1.0. Margin: the rate is near the lower edge of the band.
    DV_TOUCH: float = 1.0
    # Touch radius, m. Physics, frozen; not a Phase 2 parameter (PLAN 3.3, team rule).
    R_TOUCH: float = 1.5
    # Contested touch if nearest H and A are both inside R_TOUCH and d2 - d1 < this, m (PLAN 3.3).
    CONTESTED_MARGIN_M: float = 0.4
    # Release (kick) = touch followed by a speed rise > this, m/s (P1-A card).
    DV_RELEASE: float = 3.0
    # The rise of a release is measured over this many frames after the contact, not over DV_K.
    # The ATD ball is smooth (PLAN 1.1, R-F3): a kick spreads its speed rise over ~9 frames
    # (PLAN 1.4: 16400 has dv_9 = 5.5 m/s while its largest 3-frame step is 2.0 m/s). Same value
    # and source as RESTART_K. P1-A forensics: the public kick before 13704 rises 2.9 -> 12.7 m/s
    # in 9 frames, and its dv_3 peak (4.9 m/s) comes 8 frames after the rise starts.
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class F. Set equal to RESTART_K (9). The md5 check
    # of 5, 9 and 15 gives one output on 17 match-views, so the field is inert under the bump
    # detector (BUCKET_B.md). The value is unchanged. The 8 Nov protocol now produces it. Formerly
    # P: The 9-frame rise of public kicks 16400 and 13704 selected this window. It applies to the
    # legacy dv 3 detector.
    RELEASE_K: int = 9
    # The contact of a touch is the start of the monotone speed change that holds the dv_3 peak:
    # walk back from the peak while the one-frame speed change keeps its sign and is at least
    # DV_TOUCH / DV_K (the mean one-frame change of the smallest touch, so friction, ~0.02 m/s
    # per frame, stops the walk), for at most this many frames (= RELEASE_K, the length of a
    # kick's rise, PLAN 1.4). The player test (R_TOUCH) covers [contact, peak + DV_K]. Reason:
    # the foot is at the ball where the speed starts to change, and with a smooth track that is
    # several frames before the dv_3 peak (P1-A forensics: 13704 kicker at 0.37 m at the start of
    # the rise, 1.8 m at the peak).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class F. Set equal to RELEASE_K (9) after the same
    # md5 inertness check (BUCKET_B.md). The value is unchanged. The 8 Nov protocol now produces
    # it. Formerly P: Public trace 13704 motivated the walk-back. Its length equals RELEASE_K.
    TOUCH_ONSET_MAX_FRAMES: int = 9
    # Control = ball within R_CONTROL (m) of one player and absolute ball speed below
    # CONTROL_MAX_SPEED (m/s) for CONTROL_FRAMES consecutive frames (PLAN 1.3: 11624, 17838;
    # no relative-speed clause because of the dribble 13674-13698).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {0.6, 0.8, 1.0, 1.2, 1.5}, judge
    # metrica, view matched. Final pass 2 per-match plateau midpoints (grid index): S1 2, S2 3.5,
    # S3 1.5. Weighted median 2, snapped to index 2, moved by the guard on the DFL mean to index 3.
    # The value is unchanged. The 8 Nov protocol now produces it. Formerly P: Public cases 11624
    # and 17838 motivated slow control. Case 13674–13698 motivated the absolute-speed clause. The
    # radius is a task-card value.
    R_CONTROL: float = 1.2
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {1.0, 1.5, 2.0, 2.5, 3.0}, judge
    # metrica, view matched. Final pass 2 per-match plateau midpoints (grid index): S1 4, S2 3.5,
    # S3 3. Weighted median 3.5, snapped to index 3. Value 2.0 -> 2.5. Formerly P: Public cases
    # 11624,17838 and 13674–13698 motivated the control mechanism. The speed threshold is a task-
    # card value.
    CONTROL_MAX_SPEED: float = 2.5
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class D. Grid {3, 5, 8, 12, 25}, judge metrica, view
    # matched. Final pass 2 per-match plateau midpoints (grid index): S1 0, S2 0, S3 0. The
    # midpoint sits at a grid edge on 100% of the judging weight, so the field is inconclusive and
    # takes the grid centre under the snapping-tie rule (protocol appendix, grid-edge rule). No fit
    # selects it. Value 5 -> 8. Formerly P: Public cases 11624,17838 and 13674–13698 motivated the
    # control mechanism. The duration is a task-card value.
    CONTROL_FRAMES: int = 8
    # Dribble = the same player slot is the nearest player and within R_DRIBBLE (m) of the
    # visible ball for at least DRIBBLE_FRAMES consecutive frames, with a run median of
    # |v_ball - v_player| below DRIBBLE_REL_SPEED (m/s) and a run median ball speed above
    # CONTROL_MAX_SPEED, so a dribble is never a control (RESEARCH 3 M10, 4 rank 2). Physics:
    # a free ball cannot stay within 1 m of one player for 1 s unless he moves with it.
    # DRIBBLE_FRAMES is 1 s at 25 fps. LEARNINGS C1: a per-frame relative-speed rule switched
    # about 30 frames early at 13674-13698. There the longest run within 1 m is 13 frames and
    # the run median relative speed is 2.81 m/s (M10). The run length, the run medians and the
    # absolute ball speed carry the rule, not a single frame. The row is at the first frame of
    # the run (Vidal-Codina: the gain is at zone entry).
    R_DRIBBLE: float = 1.0
    DRIBBLE_FRAMES: int = 25
    DRIBBLE_REL_SPEED: float = 2.0
    # Deflection = heading change below this (deg) and speed change below the fraction (PLAN 3.3).
    DEFLECTION_MAX_HEADING_DEG: float = 20.0
    DEFLECTION_MAX_SPEED_FRAC: float = 0.25
    # Deflection rows keep half their confidence (P1-A card).
    DEFLECTION_CONF_FACTOR: float = 0.5
    # A deflection switches possession only if the same team releases/controls within this many
    # frames: 1 s (PLAN 3.3; P1-C card: 25 frames).
    DEFLECTION_CONFIRM_FRAMES: int = 25
    # No events while dead-ball evidence exceeds this (P1-A card; 17380 re-play inside D).
    EVENT_SUPPRESS_DEAD: float = 0.5
    # Owner = team of the last release/control with at least this confidence (P1-C card).
    OWNER_MIN_CONFIDENCE: float = 0.5
    # Event confidence = EVENT_CONF_MAX x slot confidence x distance factor x margin factor.
    # EVENT_CONF_MAX: the chance that the team of the nearest player within 1 m owns the ball. The
    # public statistic gave 0.9 (PLAN 1.2, class P). External selection (EXTERNAL_SETTINGS.md,
    # order 7): equal to P_NEAR_CLOSE, the same quantity measured on the eight open references.
    # The card says one frame never gives 1.0 (P1-A).
    EVENT_CONF_MAX: float = 0.923
    # Lowest value of the distance factor (at R_TOUCH) and of the margin factor (at
    # CONTESTED_MARGIN_M). Design value, not tuned: with it an uncontested event of a slot with
    # full confidence is always above OWNER_MIN_CONFIDENCE (0.9 x 0.8 x 0.8 = 0.576), and an
    # event of a halved slot (OVERCOUNT_CONF_FACTOR, SHORT_TRACKLET_CONF_FACTOR) or a deflection
    # is always below it (<= 0.45). P1-A.
    EVENT_CONF_FACTOR_FLOOR: float = 0.8
    # Contact detector of the touch, release and contested rows (C8, RESEARCH-2 1g1). "bump":
    # a contact is a local maximum of the vector impulse |v(t + h) - v(t - h)|, stamped at the
    # bump centre (_contact_rows). "dv3": the P1-A rows from dv_3 local maxima, walked back to
    # the start of the speed rise (_touch_rows). `--mode m2` always uses "dv3", so its reference
    # md5 stays. The dead-ball module reads the same table. C8 measured that the "dv3" rows in
    # its place give a byte-identical file, so the simpler single table stays.
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md, appendix "bucket c record" and the coordinator's
    # rulings), class E. Bucket c comparison bump against dv3 on the registered views of
    # grids.json, matched and sigma 9.5, judge metrica: bump has the higher spellF1 on S1, S2 and
    # S3 in both views, and the dv3 matched intervals are below zero on S1 and S3
    # (COMPLIANCE_RESULTS.md, bucket c record). The raw view and DFL are recorded only. The value
    # is unchanged. The 8 Nov protocol now produces it. Formerly P: C8 designed and retained the
    # bump detector while inspecting public transitions and public leave-one-out results.
    CONTACT_DETECTOR: str = "bump"
    # Half window h of the impulse, frames. RESEARCH-2 M13: the ATD smooths the ball with a
    # symmetric kernel, the median |a| profile around 46 isolated strong peaks has its half
    # maximum at +-11 frames (sigma about 9.5). h = HWHM + 1 takes the whole bump on each side.
    # The local maximum is taken over +-h frames, and the attribution window is [c - h, c + h].
    CONTACT_HALF_WINDOW: int = 12
    # Smallest impulse |J| of a contact, m/s. The loosest 1 m/s step whose full-match H<->A
    # switch rate is at most 3.0 per alive minute (RESEARCH-2 1g1 band rule, no GT). The
    # research walk gave 10. C8 measured the band on the integrated engine at h = 12. On the
    # C10 engine: 7 -> 3.55, 8 -> 3.40, 9 -> 3.18, 10 -> 3.07, 11 -> 2.88 (rule: 11). On the
    # engine with C9 and C11: 7 -> 3.20, 8 -> 3.05, 9 -> 2.86, 10 -> 2.75, 11 -> 2.61,
    # 12 -> 2.48, 13 -> 2.45 (rule: 9). The value is the rule on the engine that ships.
    DV_CONTACT: float = 9.0
    # Kind of a contact from the signed speed change ds = speed(c + h) - speed(c - h): a kick
    # if ds >= REDIRECT_FRACTION x |J|, a receive if ds <= -REDIRECT_FRACTION x |J|, else a
    # redirect. A kick or a receive changes the speed by more than half the impulse. A redirect
    # changes the direction more than the speed (RESEARCH-2 1g1, design value of the brief).
    REDIRECT_FRACTION: float = 0.5
    # C15 team-shape confirmation of a redirect by the team that does not own the ball
    # (calibration/P9B_measurements.md, LEARNINGS P9-B avenue a). Such a redirect is a hand-over
    # when its next contact confirms it (C8) or when the shape response confirms it. The
    # response is R_T = vf_T(c + SHAPE_LAG_FRAMES) - vf_T(c), vf_T the forward velocity of team
    # T towards the goal it attacks (tracks.team_forward_velocity), and the feature is
    # R_toucher - R_other. On the other-team redirects of the eight open references the shape
    # separates true hand-overs from the rest, AUC 0.70 raw and 0.83 smoothed at 1.0 s (P9B
    # 2e), and contact rhythm does not (P9B 2e: next contact by the toucher within 2 s on 43 %
    # of hand-overs and 56 % of the rest raw, 17 % and 18 % smoothed). False reproduces c13b.
    REDIRECT_SHAPE_CONFIRM: bool = True
    # Lag of the shape response, frames: 1.0 s at 25 fps, the start of the measured plateau
    # (P9B 1b: pooled AUC 0.81 from 1.0 to 2.0 s smoothed, 0.74 to 0.76 raw). The start, not the
    # centre (1.5 s): on the rule's population the AUC is 0.83 at 1.0 s and 0.84 at 1.5 s
    # (P9B 2e smoothed), against the other team's touches it is highest at 0.75 to 1.0 s and
    # falls after (P9B 1c smoothed), and a shorter window is less often cut by a dead frame.
    SHAPE_LAG_FRAMES: int = 25
    # Smallest shape feature that confirms, m/s. Precision first: the p90 of the non-hand-over
    # touches of the team without the ball at 1.0 s on the smoothed view (P9B 1a, NHO-other,
    # n 261: p10 -2.20, p50 -0.31, p90 1.45; raw n 53: p90 1.29; all NHO p90 1.09 smoothed,
    # 1.11 raw). The smoothed view because the ATD ball is smoothed (P9-A4). The hand-over
    # median is 1.21 smoothed, 0.80 raw (P9B 1a), so about half the true hand-overs pass.
    SHAPE_CONFIRM_MS: float = 1.45
    # Half window of the centred mean of the team forward velocity, frames (about 0.25 s): the
    # definition of the P9-B measurement (tools/analysis/handover_confirm.py SHAPE_HALF).
    SHAPE_HALF_FRAMES: int = 6
    # Fewest visible players of a team for its forward velocity, else it is undefined and the
    # shape does not confirm. More than half of the 11 players, so the mean is the team's shape
    # and not one player's run. BAR-ATH graded frames: median 11 per team, fewer than 6 on 4.7 %
    # (H) and 3.8 % (A); the references track 10 or 11 per team on every frame (V1,
    # DFL_validation), so there the minimum never binds and the measurement holds as it was.
    SHAPE_MIN_PLAYERS: int = 6
    # C18 reach contact (calibration/P9B_wider_radius.md, LEARNINGS "P9-B wider radius" avenue
    # a). An impulse maximum with no player of either team within R_TOUCH over [c - h, c + h]
    # gives no contact row under C8. Under this switch it becomes a contact of the nearest
    # player's team when that player is within R_REACH_M and the shape response of that team
    # confirms it (shape_confirms, the C15 SHAPE_LAG_FRAMES and SHAPE_CONFIRM_MS, no new
    # threshold). The owner_near class (the owner within R_TOUCH) is not touched: its splice
    # was below zero on 8 of 8 raw references (P9B_wider_radius.md 5a). False reproduces c17.
    REACH_SHAPE_CONFIRM: bool = True
    # Reach of a reach contact, m. P9B_wider_radius.md 1c, the nobody class at shape >= 1.45:
    # within 3.0 m, precision 59 % raw (10 of 17), 60 % on the sigma 9.5 view (87 of 144) and
    # 68 % on the matched views (13 of 19); splice +0.001 raw, +0.025 smoothed, +0.025 and
    # +0.026 matched (5b). The 3.0 to 4.0 m band is the least precise on every view (1c: 4 %
    # raw, 42 % smoothed, 8 % matched, both classes). PLAN 1.1: the ATD displaces an airborne
    # ball by metres. R_TOUCH stays the touch radius of every other rule.
    R_REACH_M: float = 3.0
    # Contact point of a bump contact (K1, docs/plan/research/RESEARCH-4.md). "window" is the
    # C8 rule: the distance of each team is its nearest distance to the ball over [c - h, c + h].
    # "kink_min" takes, for each team, the smaller of that distance and the distance to the
    # contact point of a kink model. Physics: the export smooths the ball with a symmetric kernel
    # (RESEARCH-2 M13), so at a contact the smoothed ball is displaced from the foot by about
    # |v2 - v1| sigma / sqrt(2 pi), about 2 m for a 15 m/s pass at sigma 9.5 frames, and the
    # window rule can pick a player that the smoothed path passes. The kink model is the true
    # path, straight with one velocity change at tau, seen through a Gaussian kernel of width
    # sigma: x(t) = x_c + v1 (t - tau) + (v2 - v1) G(t - tau), with G(u) = u Phi(u / sigma) +
    # sigma phi(u / sigma). It is linear in (x_c, v1, v2 - v1) for each (tau, sigma), so a
    # least-squares fit over a small grid gives the contact point (x_c, y_c) at tau. The smaller
    # of the two distances keeps the window rule where the ball is not smoothed: on the raw
    # references the window rule is already right (RESEARCH-4 K1 measurement).
    # Class E (calibration/K1_contact_point.md). The rule "min" was chosen among the kink point
    # alone, the min and the mean of the two distances on the eight open references: against the
    # reference team 12 frames after each impulse maximum, min is the only rule that gains on the
    # smoothed views (matched: Metrica 0.949 -> 0.978, DFL 0.933 -> 0.961) and does not lose on
    # the raw views (0.989 -> 0.989, 0.969 -> 0.971). Engine sweep, paired spellF1 kink_min minus
    # window, 1,000 draws, seed 0: matched Metrica +0.0252 [+0.0172, +0.0341] (3 of 3 matches
    # above zero), DFL +0.0128 [+0.0064, +0.0193]; sigma 9.5 Metrica +0.0428 [+0.0299, +0.0564],
    # DFL +0.0557 [+0.0453, +0.0648]; raw Metrica +0.0008 [-0.0017, +0.0038], DFL +0.0016
    # [-0.0018, +0.0048]. Onset and restart recall are identical on every view. "window"
    # reproduces c21_compliance byte for byte.
    CONTACT_POINT: str = "kink_min"
    # Half width of the kink fit, frames. The fit also stops halfway to the neighbouring impulse
    # maxima and at the ends of the visible run. 30 frames is 1.2 s, the longest straight
    # stretch a kernel of sigma 9.5 leaves on each side of a contact (3 sigma) (M13).
    KINK_HALF_WINDOW: int = 30
    # Kernel widths of the fit, frames: from nearly sharp to wider than M13's 9.5. The grid
    # spans the views of ATD_VIEW.md (raw, sigma 3 and 5 matched, 9.5 frozen).
    KINK_SIGMAS: tuple = (1.0, 3.0, 5.0, 7.0, 9.5, 12.0)
    # Contact time search, frames around the impulse maximum: tau = c + k x KINK_TAU_STEP for
    # |k x KINK_TAU_STEP| <= KINK_TAU_HALF. Half the impulse half window h (12) on each side.
    KINK_TAU_HALF: float = 6.0
    KINK_TAU_STEP: float = 0.5
    # Fewest visible frames of a fit, with at least KINK_SIDE_FRAMES + 1 frames beyond c on each
    # side. Three parameters per axis need many more points than three. 12 frames is 0.5 s.
    KINK_MIN_POINTS: int = 12
    KINK_SIDE_FRAMES: int = 3
    # Players are read at round(tau) +- this many frames: the player track is not smoothed by
    # the ball kernel (ATD_VIEW.md 2.2), so a short window absorbs its own jitter only.
    KINK_PLAYER_HALF: int = 2

    # -- dead ball: engine/deadball.py (P1-B; PLAN.md 1.1, 1.4, 3.4) ---------------------------
    # Collective stop: mean player speed below this, m/s. Public alive gaps run at 2.6-3.1 m/s,
    # D segments at 1.1-1.5 m/s (PLAN 1.1, R-F2).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4},
    # judge all, view raw. Final pass 2 per-match plateau midpoints (grid index): MX 1.5, N1 6, OH
    # 3, OY 4.5, PY 2.5, S1 2, S2 2.5, S3 3.5. Weighted median 2.5, snapped to index 3. The value
    # is unchanged. The 8 Nov protocol now produces it. Formerly P: PLAN 1.1 separates public
    # alive-gap speeds 2.6–3.1 m/s from public dead-ball speeds 1.1–1.5 m/s. This is a label-
    # conditioned statistic.
    V_STOP: float = 1.6
    # Team-centroid separation above this adds D weight, m (PLAN 1.1: 11.4 in play, 13.5-15 in D).
    SEP_DEAD: float = 13.0
    # Ball outside the pitch rectangle by more than this margin, m (P1-B card).
    PITCH_MARGIN_M: float = 0.5
    # Set-piece placement: ball stationary (net displacement < STATIONARY_MAX_DISP_M over
    # STATIONARY_S) with one player within SETPIECE_TAKER_M and all others beyond
    # SETPIECE_OTHERS_M (PLAN 3.4; P1-B card).
    STATIONARY_S: float = 1.5
    STATIONARY_MAX_DISP_M: float = 0.5
    SETPIECE_TAKER_M: float = 1.0
    SETPIECE_OTHERS_M: float = 5.0
    # Ghost ball: stationary with nobody within GHOST_NOBODY_M after an event faster than
    # GHOST_PRIOR_SPEED -> suppress the stationary term, freeze the last team (PLAN 1.1, R-F5:
    # GK holding after a 27 m/s shot, 13755-13800).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class U. Valley of the nearest-player distance
    # histogram over the stationary-ball frames of the full BAR-ATH match, 0.5 m bins, no labels:
    # the bin 3.5 to 4.0 m (BUCKET_B.md). Value 3.0 -> 3.75. Formerly P: Public case 13755–13800
    # motivated the ghost-ball mechanism. The distance threshold is a design value.
    GHOST_NOBODY_M: float = 3.75
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class F. Set equal to AIRBORNE_SPEED, 15.0 m/s
    # (bucket b rule, BUCKET_B.md). The value is unchanged. The 8 Nov protocol now produces it.
    # Formerly P: Public case 13755–13800 motivated the ghost-ball mechanism after a 27 m/s shot.
    # The speed threshold is a design value.
    GHOST_PRIOR_SPEED: float = 15.0
    # Smart Tagging SET PIECES: weak restart prior on the last seconds of a window (PLAN 1.6, 3.4).
    SETPIECE_TAG_TAIL_S: float = 5.0
    # Out by extrapolation: friction-decayed roll, m/s^2 (P1-B card: k = 0.5 m/s^2).
    FRICTION_DECEL: float = 0.5
    # Out onset when the extrapolation does not reach the line: last_visible + this, frames.
    # Untuned zero-offset fallback, P9-F OUT_LAG_FALLBACK.md. The former +21 came from public
    # labels. Organizer issue 16 excludes that selection. Zero keeps the observable anchor.
    # This removes an unsupported adjustment. It does not estimate the true stoppage delay.
    # With 0, the reappearance rejection in `deadball` (`onset >= next_visible`) can no longer
    # fire for this branch (OUT_LAG_FALLBACK.md).
    DELTA_OUT_LAG: int = 0
    # Pursuit stop: ball faster than PURSUIT_MIN_BALL_SPEED (m/s), kicking team's distance
    # rising, mean speed of the PURSUIT_N_NEAREST players nearest the ball falling below
    # PURSUIT_STOP_SPEED (m/s) (PLAN 3.4; 15889, 17380).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {3, 4, 5, 6, 8, 10}, judge dfl, view
    # matched. Final pass 2 per-match plateau midpoints (grid index): MX 1.5, N1 2, OH 1.5, OY 2.5,
    # PY 4. Weighted median 2, snapped to index 2, moved by the guard on the Metrica mean to index
    # 1. Value 5.0 -> 4.0. Formerly P: Public cases 15889 and 17380 motivated pursuit-stop
    # detection. The speed threshold is a task-card value.
    PURSUIT_MIN_BALL_SPEED: float = 4.0
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {2, 4, 6, 8, 11}, judge dfl, view raw.
    # Final pass 2 per-match plateau midpoints (grid index): MX 4, N1 0.5, OH 0.5, OY 2, PY 2.5.
    # Weighted median 2, snapped to index 2, moved by the guard on the Metrica mean to index 3.
    # Value 6 -> 8. Formerly P: Public cases 15889 and 17380 motivated pursuit-stop detection. The
    # player count is a task-card value.
    PURSUIT_N_NEAREST: int = 8
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {1.0, 1.25, 1.5, 2.0, 2.5}, judge dfl,
    # view raw. Final pass 2 per-match plateau midpoints (grid index): MX 0.5, N1 2.5, OH 4, OY 2,
    # PY 1. Weighted median 2, snapped to index 2. The value is unchanged. The 8 Nov protocol now
    # produces it. Formerly P: Public cases 15889 and 17380 motivated pursuit-stop detection. The
    # stop-speed threshold is a task-card value.
    PURSUIT_STOP_SPEED: float = 1.5
    # Collective stop (ball visible): below V_STOP and sep > SEP_DEAD for at least this, s (PLAN 3.4).
    # Also (P1-B, M1): the dwell of an e_D > DEAD_EVIDENCE_HIGH run before it enters D (the public
    # alive gap 17161-17207 is 46 frames of gap + slow: stays alive, small margin), and the time
    # after a pursuit-stop onset in which no kicking-team player may reach the ball (18612).
    COLLECTIVE_STOP_S: float = 2.0
    # e_D above this counts as "dead" for event suppression and restart arming (P1-A, P1-B).
    DEAD_EVIDENCE_HIGH: float = 0.5
    # A restart is armed after e_D has been high for at least this, s (PLAN 3.4). At M1 the
    # orchestrator arms on time in the D state instead (deadball.restarts() has no state; e_D is
    # low before 16400 / 18389, whose D is carried by an onset): P1-B deviation.
    RESTART_ARM_S: float = 1.0
    # Restart kick: signed speed rise over RESTART_K frames above DV_RESTART (m/s) with a player
    # within RESTART_PLAYER_M (m); independent of pre-kick speed (PLAN 1.4: 16400 dv_9 = 5.5 m/s
    # at -3 frames while its largest 3-frame step is 2.0 m/s). A placed ball needs no player
    # within RESTART_PLAYER_M (PLACED_BALL_RESTART below).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {2, 2.5, 3, 4, 5}, judge metrica, view
    # pooled: raw + matched. Final pass 2 per-match plateau midpoints (grid index): S1@matched 1,
    # S1@raw 2.5, S2@matched 3.5, S2@raw 2, S3@matched 3.5, S3@raw 2. Weighted median 2.25, snapped
    # to index 2 (no guard: the other provider is recorded only). The value is unchanged. The 8 Nov
    # protocol now produces it. Formerly P: Public restart 16400 motivated a 9-frame rise above 3
    # m/s instead of the task card’s 3-frame step.
    DV_RESTART: float = 3.0
    # RESTART_K: public 16400 chose 9 (class P). External selection (EXTERNAL_SETTINGS.md, order
    # 9): the p90 of the frames from a kick contact to a gain above DV_RESTART, sigma 9.5 views of
    # the eight open references: 10, 9, 8, 9, 9, 9.6, 8, 11. Median 9, the same value.
    RESTART_K: int = 9
    RESTART_PLAYER_M: float = 1.5
    # Restart by reappearance: ball reappears after a gap already faster than this, m/s; team of
    # the nearest player if within REAPPEAR_TEAM_M (PLAN 1.4: 12989, 17838 reappear at ~9-10 m/s).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {0, 1, 2, 3, 4, 6, 8}, judge metrica,
    # view raw. Final pass 2 per-match plateau midpoints (grid index): S1 0, S2 1.5, S3 2. Weighted
    # median 1.5, snapped to index 2 (no guard: the other provider is recorded only). Value 4.0 ->
    # 2.0. Formerly P: Public restarts 12989 and 17838 motivated the reappearance mechanism.4 m/s
    # is a design margin below their 9–10 m/s speeds.
    REAPPEAR_SPEED: float = 2.0
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class F. Grid {1.5, 2, 3, 4, 5}, judge metrica, view
    # raw. Final pass 2 per-match plateau midpoints (grid index): S1 2, S2 2, S3 2. The judged
    # plateau {1.5, 2, 3, 4, 5} holds the current R_REACH_M = 3.0, so the field is set to R_REACH_M
    # (tie_to, class F). The value is unchanged. The 8 Nov protocol now produces it. Formerly P:
    # Public restarts 12989 and 17838 motivated the reappearance mechanism. The attribution radius
    # is a design value.
    REAPPEAR_TEAM_M: float = 3.0
    # Placed-ball restart (deadball.placed_kicks, C12 brief, F1 rank 1). Physics: a ball at rest
    # moves only when somebody plays it. (a) The first rise above DV_RESTART after the ball was
    # stationary (STATIONARY_S, STATIONARY_MAX_DISP_M) in the same visible run is a kick
    # candidate whatever the distance of the nearest tracked player. Its team is the nearest
    # player within REAPPEAR_TEAM_M, else none. It takes the usual confirmation
    # (RESTART_CONFIRM_S, RESTART_CHAIN_FRAMES), so a ball nudged inside a stoppage does not
    # restart play (LEARNINGS C4). (b) A reappearance whose ball comes to rest within
    # RESTART_CONFIRM_S in the same visible run is no restart candidate: the placed-ball kick
    # is. A rise with the ball beyond a line by more than PITCH_MARGIN_M is no placed-ball kick:
    # Laws 13, 16 and 17 put the ball of a free kick, a goal kick and a corner on the pitch
    # (186261: a ball at rest 6.6 m beyond the touchline comes back onto the pitch).
    # Cases: 77859 (taker untracked, nearest player 28 m), 86635 (3.4 m), 82254 and 143837
    # (clause b). No new number: every value is an existing constant. False turns it off.
    PLACED_BALL_RESTART: bool = True
    # Goalkeeper hold (deadball.keeper_hold_mask, possession.dead_intervals, C13 brief, F1
    # Section 6 rank 2). Law 12: a goalkeeper may hold the ball in his hands and play goes on.
    # The tracker does not see a ball in the hands, so the hold is a ball gap. Slow walking
    # players then make it a gap-plus-stop evidence entry. An evidence entry in a gap whose ball
    # was last seen inside a goalkeeper's own penalty area, within R_TOUCH of his slot and
    # slower than DV_RESTART, starts no `D` unless the stoppage it would start holds a deep stop
    # or a collective stop (V_STOP, SEP_DEAD, COLLECTIVE_STOP_S). Cases: 133932 (21 s hold,
    # then a punt) and 187146 become alive. 175384 (64 s of standing players) stays `D`. The
    # public hold 13755-13800 sits inside an alive spell. False reproduces c12_placed_ball.
    GK_HOLD_IS_PLAY: bool = True
    # Deep stop: players this slow, m/s, for this long, s. F1 Section 1 and Section 6 rank 2:
    # the slowest alive 20 s window of the match has a mean of 1.36 m/s, so 1.0 m/s for 10 s does
    # not happen in play. The players walk at 1.09 m/s or faster in 133932 and 1.06 m/s or
    # faster in 187146, and 175384 has a 33.8 s run below 1.0 m/s.
    GK_HOLD_DEEP_STOP_SPEED: float = 1.0
    GK_HOLD_DEEP_STOP_S: float = 10.0
    # Owner of a goalkeeper hold (possession.keeper_hold_rows, C13b). Law 12: a goalkeeper who
    # catches the ball controls it. A hold gap with no stoppage frame gets a `control` row of the
    # goalkeeper's team at its first frame, with the confidence of a control. The public ground
    # truth gives the hold 13755-13800 to the goalkeeper's team from 13785. No timing offset: one
    # public case cannot support one (PLAN 5.1). Case: 133932, the away goalkeeper's catch,
    # which C13 left with the attacking home team. False reproduces c13_gk_hold.
    GK_HOLD_OWNER: bool = True
    # e_D = sigmoid(DEAD_LOGIT_BIAS + sum of the active terms' weights), nats (P1-B card: "weighted
    # sum of logits; weights in config"). Calibrated on the card's strength classes, not on the
    # public score: nothing active -> 0.05; a weak term alone (W_SLOW, W_SEP, W_TAG_TAIL) stays
    # well below DEAD_EVIDENCE_HIGH, and W_SLOW + W_SEP together still do (-1 -> 0.27): slow,
    # separated players with the ball visible are common in play (PLAN 1.1: 32% of ball-visible
    # frames below V_STOP; alive H 17300-17380 and 18389+ look like that). A strong term with the
    # slow term clears it: gap + collective stop -3 + 3.5 + 1 -> 0.82 (0.92 with sep);
    # set-piece placement alone -> 0.73; ball out of the rectangle -> 0.99.
    DEAD_LOGIT_BIAS: float = -3.0
    W_GAP_STOP: float = 3.5        # ball missing AND meanv_all < V_STOP (PLAN 3.4, R-F2); W_SLOW adds
    W_SLOW: float = 1.0            # meanv_all < V_STOP, weak alone (PLAN 1.1)
    W_SEP: float = 1.0             # sep > SEP_DEAD, weak alone (PLAN 1.1: 11.4 m in play)
    W_OUT: float = 8.0             # ball outside the pitch rectangle + PITCH_MARGIN_M (PLAN 1.4)
    W_SETPIECE: float = 4.0        # stationary ball, one taker, others away (PLAN 3.4)
    W_TAG_TAIL: float = 0.75       # last SETPIECE_TAG_TAIL_S of a SET PIECES window (PLAN 1.6)
    # Out onset: the last heading must point at a boundary line within this distance, m. 12851
    # was lost 13 m from the touchline, heading for it (PLAN 1.4); beyond ~20 m a ball last seen
    # rolling cannot plausibly be going out, and the gap is left to the gap + stop evidence.
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Smallest 5 m multiple that holds at least
    # 90 % of the Metrica exit headings: 12 of the 13 exits with a heading lie within 25 m
    # (BUCKET_B.md, advisor ruling on the denominator). Value 20.0 -> 25.0. Formerly P: Public case
    # 12851 loses the ball 13 m from the line.20 m was the plausibility ceiling above that example.
    OUT_MAX_RANGE_M: float = 25.0
    # Observed out onset: a run beyond the line that starts with the ball reappearing beyond a
    # touchline is a throw-in, not an out, if y moves toward the pitch over this many frames
    # (fall back to vy when none of them is visible). Physics: a ball that goes out crosses the
    # line from inside. C10 brief (RESEARCH-2 1c, M18): all 20 graded cases move inward within
    # 4 frames, and any window of 2 to 10 frames gives the same set. Cases: 19772 and 27803
    # (throw-in tag), 163487 (no tag). The goal-line reappearance at 73951 keeps its onset.
    OUT_INWARD_FRAMES: int = 4
    # A missing-ball out onset and a pursuit stop are kept only if a collective stop (V_STOP and
    # SEP_DEAD for COLLECTIVE_STOP_S) starts within this many seconds after the onset: players
    # react to the whistle within a few seconds. Public: 12851 +1.1 s, 15889 +3.3 s, 17380 at once;
    # the two false outs inside the alive A spell (16625: meanv never below 1.8 m/s; 17025: slow
    # only +5.4 s later, for 1.8 s) fail it (P1-B forensics). Also the pursuit window: the
    # nearest players must fall below PURSUIT_STOP_SPEED within this after the flight starts.
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {2, 3, 4, 6, 8, 10}, judge all, view
    # raw. Final pass 2 per-match plateau midpoints (grid index): MX 5, N1 4.5, OH 5, OY 4.5, PY 5,
    # S1 2, S2 1.5, S3 1.5. Weighted median 3.25, snapped to index 3. Value 4.0 -> 6.0. Formerly P:
    # The public true onsets confirm after 0–3.3 s. Two false public outs fail the confirmation.
    # Those cases motivated 4 s.
    ONSET_CONFIRM_S: float = 6.0
    # Pursuit stop: the kicking team is the one with the smaller mean nearest distance over this
    # many frames before the fast run starts (15889: dH = dA = 2.7 m at the run start, A clearly
    # nearer 4-8 frames before; P1-B forensics).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Grid {0, 5, 10, 15, 20, 25}, judge dfl,
    # view raw. Final pass 2 per-match plateau midpoints (grid index): MX 3.5, N1 0, OH 2.5, OY 3,
    # PY 2.5. Weighted median 2.5, snapped to index 2. The value is unchanged. The 8 Nov protocol
    # now produces it. Formerly P: At public case 15889 the teams were equally near at the run
    # start, while A was nearer 4–8 frames earlier.
    PURSUIT_KICK_LOOKBACK: int = 10
    # Onset confidences by mechanism (P1-B card: collective stop lowest; an observed crossing is
    # near-certain). Ordinal; the P1-C decoder reads them as soft hints.
    CONF_OUT_OBSERVED: float = 1.0
    CONF_OUT_EXTRAPOLATED: float = 0.8
    CONF_OUT_LAG: float = 0.6
    CONF_PURSUIT: float = 0.6
    CONF_COLLECTIVE: float = 0.3
    # Restart confidences: base per kind, then +/- RESTART_PRIOR_STEP per agreeing/disagreeing
    # prior (out -> other team than the last toucher; Smart Tagging Type/Team; P1-B card:
    # "confidence boosts only", never a rule), clipped to [0, 1].
    CONF_RESTART_KICK: float = 0.6
    CONF_RESTART_REAPPEAR: float = 0.6
    RESTART_PRIOR_STEP: float = 0.15
    # A restart candidate is confirmed only if meanv_all rises above V_STOP within this many
    # seconds: after a real restart players re-engage; a re-play inside D does not make them move
    # (PLAN 1.4: 17380 re-played inside D). NOT physics-derived: the window was read off public
    # cases (P1-B forensics). Measured from the GT restart frames, 12989 / 16400 / 17838 / 18389
    # reach 2.99 / 1.83 / 2.78 / 1.99 m/s within 6 s, while the in-D re-plays at 16064 (15889 D)
    # and 17428 (17380 D) stay <= 1.37 m/s. But the DETECTED 18389 kick (18355, the smooth-ramp
    # early fire) fails it, so the M1 exit falls to the next kick (18415, +26). Full match: with
    # it D 50.2% / 98 segments, without it 47.6% / 118. Coordinator / P1-C decision.
    # C11 keeps the window and adds the chain rule below, which confirms 18355. External selection
    # (EXTERNAL_SETTINGS.md, order 3): the horizon that best separates reference restarts from
    # frames inside stoppages, eight open references: 6.0, 5.0, 7.0, 6.0, 5.0, 4.5, 6.0, 7.5.
    # Median 6.0, the same value.
    RESTART_CONFIRM_S: float = 6.0
    # Chain confirmation (deadball.chain_confirmed, C11 brief, RESEARCH-2 1b, M15). A candidate
    # that the players test rejects is confirmed when the next players-confirmed candidate starts
    # within this many frames and no out or pursuit_stop onset lies between the two. Physics:
    # play resuming is a property of the passing sequence, not of its first kick. Source: the
    # full-match median gap between consecutive kick candidates in alive play is 87 frames
    # (releases: 93.5). Any value from 60 to 124 gives the same full-match result. The public
    # cases bound it but did not choose it: 18389 needs at least 60 (the 18355 kick, next
    # confirmed candidate 60 frames later), and the in-stoppage re-plays need less than 335
    # (16058) and 401 (17425). 0 turns the chain off.
    RESTART_CHAIN_FRAMES: int = 87
    # Kickoff team (possession.mark_team, RESEARCH 4 rank 3, M9). Law 8: the ball is on the
    # centre mark and the taker stands at it. The opponents stay outside the centre circle until
    # the kick. The kickoff team is the team of the single tracked player nearest the centre
    # mark over the KICKOFF_SEARCH_FRAMES frames before the kickoff frame (3 s at 25 fps,
    # RESEARCH rank 3). Counting players in the circle does not work: only 0-4 players of each
    # team are tracked there (M9).
    KICKOFF_SEARCH_FRAMES: int = 75
    # That player counts only within this distance of the mark, m. Physics: the taker is at the
    # ball, so within the touch radius R_TOUCH (PLAN 3.3). C6 measurement on the full match:
    # where the taker is tracked (half-1 kickoff, three after-goal kickoffs), the nearest player
    # is 0.24-0.85 m from the mark on every frame of the 75-frame window. Where the taker is not
    # tracked (second-half kickoff, one after-goal kickoff), the nearest tracked player is at
    # best 1.93 m and 3.63 m away. At that after-goal kickoff he stands 3-7 m from the mark.
    # Law 8 does not let an opponent stand there, so he is not evidence of the taker. Beyond
    # this limit the restart candidate and Law 8 decide (possession.kickoff_decisions).
    KICKOFF_TAKER_M: float = 1.5

    # -- decoder: engine/decoder.py, engine/possession.py (P1-C; PLAN.md 3.7) ------------------
    # Emission: owner log(P_OWNER), other log(1 - P_OWNER) (P1-C card: "log(0.9)-ish").
    P_OWNER: float = 0.9
    # Smart Tagging team phase prior on tagged frames, as a likelihood ratio (P1-C card: log 1.5).
    # The tagged team gets +log(ratio) on the H-vs-A log-odds, the other team -log(ratio). Phase
    # boundaries are soft (+-4 s) and lean early, so the prior is weak (PLAN 1.6).
    TAG_TEAM_PRIOR_RATIO: float = 1.5
    # H<->A switch cost, nats, and its cost within SWITCH_EVENT_WINDOW frames of an event (touch,
    # release, control or dribble, any confidence) of the new team (PLAN 3.7, P1-C card start
    # values, dribble from RESEARCH 4 rank 2).
    # With the held owner (log-odds log 9 = 2.2 nats) a switch without an event needs about
    # C_SWITCH / 2.2 = 3.6 frames of opposite evidence.
    C_SWITCH: float = 8.0
    C_SWITCH_EVENT: float = 0.5
    SWITCH_EVENT_WINDOW: int = 3
    # alive->D cost, nats, and its cost within DEAD_EVENT_WINDOW frames of an onset (any
    # mechanism of deadball.dead_onsets) (PLAN 3.7, P1-C card start values).
    C_DEAD: float = 6.0
    C_DEAD_EVENT: float = 0.5
    DEAD_EVENT_WINDOW: int = 5
    # D->alive cost, nats, and its cost within ALIVE_EVENT_WINDOW frames of a restart (a
    # confirmed deadball.restarts candidate or a kickoff). The restart team gets
    # RESTART_TEAM_BONUS nats off, so its cost can be below 0 (PLAN 3.7, P1-C card start values).
    C_ALIVE: float = 6.0
    C_ALIVE_EVENT: float = 0.5
    ALIVE_EVENT_WINDOW: int = 5
    RESTART_TEAM_BONUS: float = 1.0
    # Minimum durations, frames (post-pass). Do not raise MIN_POSS toward 0.8 s: the public
    # window has a 19-frame possession (PLAN 1.3, 3.7; findings R-F10).
    MIN_POSS: int = 10
    MIN_DEAD: int = 25
    # Delayed nearest-player evidence (P1-C). The raw nearest-player team flips 14-25 frames
    # before the GT in 11 of the 16 public transitions into a possession (PLAN 1.3). The evidence
    # at frame t is the nearest player at t - NEAREST_LAG. The public cases chose 20 (class P).
    # External selection (docs/plan/calibration/EXTERNAL_SETTINGS.md, order 4): the best lag of the
    # all-distance agreement of the nearest team with the reference owner, matched views of the
    # eight open references: 11, 16, 19, 26, 18, 24, 22, 18 (DFL WMX, N1, OH, OY, PY, Metrica S1,
    # S2, S3). Median 18.5, rounded half up to 19.
    NEAREST_LAG: int = 19
    # The nearest player's team is the GT team on 90.6% of alive ball-visible frames when he is
    # within 1 m (PLAN 1.2). Log-odds of the delayed nearest player:
    # log(P_NEAR_CLOSE / (1 - P_NEAR_CLOSE)) within NEAR_CLOSE_M, log(P_NEAR_FAR / ...) within
    # NEAR_FAR_M, no evidence beyond NEAR_FAR_M or without a ball. Used only while the event
    # owner is unknown (after a new-team event that does not set the owner, P1-C card: "may move
    # it only near a new-team event"). The public value was 0.906 (class P). External selection
    # (EXTERNAL_SETTINGS.md, order 6): the median over the eight open references, matched views,
    # of the agreement within NEAR_CLOSE_M at lag NEAREST_LAG = 19. The matches spread over about
    # 6 points (DFL lower, Metrica higher). The median is the protocol's rule.
    P_NEAR_CLOSE: float = 0.923
    # External selection (EXTERNAL_SETTINGS.md, order 5): the upper edge of the distance band with
    # the largest median drop in agreement to the next band (1.0 m: 0.070, 1.5 m: 0.068).
    NEAR_CLOSE_M: float = 1.0
    # Class E (selected on external labelled matches), C17, 2 Nov 2026. The median over eight
    # open references of the agreement of the nearest team at t - NEAREST_LAG with the reference
    # owner at t, nearest player 1 to 5 m from the ball (the band this constant covers), on alive
    # ball-visible non-ghost frames: DFL-J03WMX 0.736, J03WN1 0.722, J03WOH 0.730, J03WOY 0.714,
    # J03WPY 0.741, METRICA-S1 0.635, S2 0.654, S3 0.699. Median 0.718 (log-odds 0.94). Leaving
    # out any one match moves the median by 0.004. Tool: tools/analysis/provenance_recheck.py,
    # table in docs/plan/calibration/PROVENANCE_RECHECK.md. It replaces the BAR-ATH public value
    # 0.6 (PLAN 1.2: about 60% at 3-5 m, class P, log-odds 0.41). Re-measured with the same
    # statistic at NEAREST_LAG = 19 on the matched views (EXTERNAL_SETTINGS.md): median 0.724.
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Median over the eight open references of
    # the nearest-team agreement over (1.0, 4.0] m at lag 19, matched views: 76.0, 74.2, 75.4,
    # 73.9, 76.4, 67.7, 69.4, 73.5 % (COMPLIANCE_RESULTS.md, Phase B). Value 0.724 -> 0.74.
    # Formerly E: Median of nearest-team agreement over 1–5 m at lag 19, matched views of eight
    # open references (EXTERNAL_SETTINGS); first selected at lag 20 by C17 (0.718).
    P_NEAR_FAR: float = 0.74
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Pooled rule at NEAREST_LAG 19 and
    # NEAR_CLOSE_M 1.0 on the matched views: the smallest whole edge E with the pooled (E, 8] m
    # agreement within 3 points of the 5 to 8 m agreement on every open reference. E = 3 fails on
    # METRICA-S3 by 0.02 points (COMPLIANCE_RESULTS.md, Phase B). Value 5.0 -> 4.0. Formerly P:
    # PLAN 1.2 used the nearest-player statistic at 3–5 m.5 m became its evidence cutoff, without
    # separate fitting.
    NEAR_FAR_M: float = 4.0
    # Smart Tagging phases whose team is the team in possession (PLAN 1.6, the list of
    # tools/sanity.py). DEFENDING*, DEFENSIVE TRANSITION and PLAYERS IN THE BOX name the
    # defending side or a sub-phase, so they carry no possession prior.
    TAG_TEAM_PHASE_CODES: tuple = ("BUILD UP", "PROGRESSION", "CREATING CHANCES",
                                   "ATTACKING TRANSITION", "LONG BALL", "BALL IN FINAL THIRD",
                                   "BALL IN THE BOX")
    # D emission (P1-C). Per-frame e_D cannot hold a stoppage on its own: e_D > 0.5 covers 38.5%
    # of the public GT D frames and 0% of 15889 (HANDOFF 4.3), and it is above 0.5 for 46 frames
    # in the public alive gap 17161-17207 (HANDOFF 4.11). The dead-ball state machine of M1
    # (possession.dead_intervals: onsets, the COLLECTIVE_STOP_S dwell, confirmed restarts) is the
    # stoppage. P(D) is at least P_DEAD_STOPPAGE inside it and at most 1 - P_DEAD_STOPPAGE
    # outside it. P_DEAD_STOPPAGE has the "0.9-ish" strength of P_OWNER (P1-C card).
    P_DEAD_STOPPAGE: float = 0.9
    # Numerical floor on every probability before the log, so that no log is -inf. It never
    # binds on alive frames (e_D >= sigmoid(DEAD_LOGIT_BIAS) = 0.05) and keeps an observed out
    # (e_D up to sigmoid(5) = 0.993) finite. Pre-kickoff frames are forced to D, not floored.
    P_FLOOR: float = 0.001

    # -- GT semantics: Phase 2 calibration (P2 card, PLAN.md 1.3, 1.4, 5.1) --------------------
    # tools/calibrate.py measured the numbers below on the public window, tol 25 frames. The
    # report with every grid point and the leave-one-out table is
    # docs/plan/calibration/P2_loo_report.md. The public score chose nothing (PLAN 5.1).
    #
    # Which events set the owner: "release" or "release_or_control". The "release" rule also
    # drops the control rows from the switch costs and the owner walk (RULE_TEAM_KINDS).
    # "release_or_control" includes the dribble rows. "release" leaves them out (RESEARCH 4
    # rank 2, C5 brief).
    # P2: "release" matches the same 10 public transitions as rule v1 at tol 25 and gains none,
    # so the incumbent stays. Forensic support for controls: 11624 and 17838 switch on a slow
    # ball at a player's feet with no kick (PLAN 1.3).
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md), class E. Bucket c comparison on the three Metrica
    # matches, raw and matched views: release is below release_or_control on every match by 0.23 to
    # 0.30 with every paired interval below zero (COMPLIANCE_RESULTS.md, POSSESSION_RULE decision).
    # The value is unchanged. The 8 Nov protocol now produces it. Formerly P: PLAN 1.3 inferred the
    # owner rule from public labels. The P 2 public-window grid retained release_or_control.
    POSSESSION_RULE: str = "release_or_control"
    # The event kinds that release the owner (possession.release_rows). A row of the other team
    # of one of these kinds makes the owner UNKNOWN when the row does not set the owner itself.
    # The delayed nearest player then decides the team. RULE_TEAM_KINDS still limits the kinds.
    # A bare `touch` is not in the tuple (C7, RESEARCH 4 rank 4). C7 measured the full match
    # (c6_kickoff). 45 bare touches of the other team released the owner, and 22 of them made
    # the decoder switch. 40 of the 45 have slot confidence 1.0 and 5 have 0.5. The team margin
    # d2 - d1 is 1.3 m at p10 and 3.1 m at p50. No margin is below 0.4 m, because the contested
    # rule already removes those touches. Slot confidence and margin do not find the bad
    # releases. The public spurious switch 12818 rests on a touch with slot confidence 1.0, a
    # margin of 1.26 m and confidence 0.82. RESEARCH proposed "slot confidence >= 0.8 and margin
    # >= 0.4 m, or two touches within 2 s". That rule changes 6 frames and no public transition.
    # A second touch of the same team within 1 s, 2 s or 3 s never changes the output. So the
    # two-touch clause equals no touch release, and the simpler rule stays. The GT switches
    # when the new team kicks or controls the ball (PLAN 1.3: 13674-13698). The touch at 13691
    # switched 13 frames before the GT 13704 and 6 frames before the kick at 13697. Within 2 s,
    # a release, control or dribble of the same team follows 34 of the 45 touches. That event
    # then switches the owner. Add "touch" to the tuple to get the c6_kickoff behaviour back.
    # 8 Nov protocol (COMPLIANCE_PROTOCOL.md, appendix "bucket c record" and the coordinator's
    # rulings), class E. Bucket c comparison of (release, control), (release, control, dribble)
    # and (release, control, dribble, touch) on the three Metrica matches, raw and matched views:
    # the raw view picks the touch set and the matched view the dribble set, so the views tie,
    # and the registered tie "fewer kinds" gives (release, control). The first two sets give
    # byte-identical outputs on all 24 external runs (COMPLIANCE_RESULTS.md, bucket c record).
    # The dribble rows still feed RULE_TEAM_KINDS; only the owner-release set changes.
    # Value (release, control, dribble) -> (release, control). Formerly P: PLAN 1.3 and public
    # cases 13674–13698 and 12818 motivated release, control and dribble as owner changes.
    OWNER_RELEASE_KINDS: tuple = ("release", "control")
    # Flight assignment: a release whose next event is an owner event of the other team gives
    # the ball to that team from the release (possession.flight_assignment, R-F9). A stoppage
    # before the next event cancels it (17380), and so does a contested next event. P2: on, it
    # gains 14846 (-9) and loses nothing, but it rests on that one case. Without 14846 the
    # choice goes back to off, so the simpler choice holds: off. Full match with it on: 27
    # flights (6.9% of 392 owner releases, 11.8% of possessions), median flight 98 frames,
    # p90 269, max 385 (far beyond a ball flight: the next event comes late because the
    # reception has no event). 56 walks stop at a stoppage, 0 paint through one.
    FLIGHT_ASSIGNMENT: bool = False
    # Per-kind timing offsets, frames (may be negative). possession.run() moves the event
    # frames of each kind before the decoder: the event table by DELTA_SWITCH, the onsets by
    # DELTA_DEAD, the restart candidates by DELTA_RESTART_KICK / DELTA_RESTART_REAPPEAR. The
    # stoppage edges move after the M1 state machine chooses them (possession.shift_stoppage),
    # so an offset never changes which candidate ends a stoppage. No shift crosses a block edge.
    # Choice rule (P2 card): the grid value nearest to minus the median matched latency of the
    # kind. On leave-one-out disagreement, the smallest |delta| of the choices.
    #
    # H<->A matched latencies +9 (15346), +11 (17293), -13 (13704): two receptions before our
    # first event and one lagged-nearest flip, so the sign is mixed. Choice -9, leave-one-out
    # gives 0 (15346 out) and +3 (17293 out): 0.
    DELTA_SWITCH: int = 0
    # alive->D matched latencies 0 (12851), -10 (15889), -9 (17380), -2 (18027): four onset
    # mechanisms (PLAN 1.4). Choice +5, leave-one-out gives 0 (15889 or 17380 out) and +6: 0.
    DELTA_DEAD: int = 0
    # D->alive kick: one matched case, -7 (16400). 18389 is the smooth ramp, missed at +26.
    # Choice +7, leave-one-out without 16400 has no case: 0 (P2).
    # C11 sets it to 12 from physics, not from the public window (RESEARCH-2 1b, M13, M16).
    # deadball._rise_start stamps a kick at the start of the speed rise. The ATD ball is smoothed
    # by a symmetric kernel (M13: half-width at half-maximum 11 frames), so the contact is the
    # centre of the acceleration bump. Over 654 graded kick candidates the bump centre sits
    # 8 / 12 / 15 frames (p10 / p50 / p90) after _rise_start (M16, re-measured on c9_gk_guard).
    # The value moves the chosen kick exits only (possession.shift_stoppage), after the state
    # machine (LEARNINGS C3, C4). It does not apply to reappear exits.
    DELTA_RESTART_KICK: int = 12
    # R1 fallback: date a confirmed reappearance at the observation, with no added offset.
    # P2 chose the previous +10 from two public transitions. Organizer issue 16 excludes
    # public-window threshold tuning. Remove that fitted offset (SEALED_OPENING.md Section 4,
    # PROVENANCE.md, CANDIDATE_B.md). Provenance selects the candidate, not its score.
    DELTA_RESTART_REAPPEAR: int = 0
    # Gap-dwell onset at the players' deceleration knee (deadball.deceleration_knee,
    # possession.knee_entries, COMPONENT_KNEE.md, IDEAS 2). A gap-plus-stop dwell entry is dated
    # when the players are already under V_STOP. The whistle comes before: players decelerate
    # after it (Lang, Wild, Isenko & Link 2022, Sci. Rep. 12:16291). LEARNINGS R3-C: Metrica
    # fouls enter D late by a median of 48 frames, 37 of 38 through this dwell. The rule moves
    # such an entry (ball missing at the entry, not an onset row) to the knee + DELTA_KNEE. It
    # never moves an entry later, never starts a stoppage and never adds an onset. Off by
    # default: false gives the base output byte for byte. Read in possession.run() only.
    # Acceptance test (COMPONENT_KNEE.md Section 7, 8 Nov 2026): Metrica +0.0058 raw, +0.0054
    # matched, both intervals above zero, 3/3 matches, DFL 0, BAR-ATH vetoes pass; but S3 gets
    # one new spell under 1 s in each view (criterion v), so variant 1 failed. Variant 2 adds
    # KNEE_MIN_PRE_SPELL_S and passes every criterion (Section 9). On by default since 8 Nov 2026
    # (COMPLIANCE_PROTOCOL.md appendix, new components): the pre-declared test decides it.
    ONSET_KNEE: bool = True
    # Frames from the knee to the onset. COMPONENT_KNEE.md Section 3: the pooled median over
    # METRICA-S1 to S3, raw and matched views, of (reference onset - knee), gap clamp off:
    # -15 over 183 entries (p25 -35, p75 +10; raw -20 over 71, matched -12.5 over 112). The
    # onset comes 0.6 s before the half-way frame of the fall. DFL has 1 such entry (D3).
    DELTA_KNEE: int = -15
    # Look-back cap, s: the knee is never searched earlier than this before the entry. Lang et
    # al. 2022: the +-4 s band of player movement carries the in-play status (IDEAS 2).
    KNEE_LOOKBACK_S: float = 4.0
    # Centred moving mean of meanv_all before the knee search, s (13 frames at 25 fps). Design
    # value of IDEAS 2, fixed before any run (COMPONENT_KNEE.md Section 2).
    KNEE_SMOOTH_S: float = 0.5
    # Pre-entry window, s before the entry: [entry - FROM, entry - TO). Its median is the speed
    # before the fall. The far end is the Lang band; the near end leaves the COLLECTIVE_STOP_S
    # (2 s) before the entry. Design value of IDEAS 2, fixed before any run.
    KNEE_PRE_FROM_S: float = 4.0
    KNEE_PRE_TO_S: float = 2.0
    # The knee is the last frame at or above V_STOP + KNEE_LEVEL_FRAC x (pre - V_STOP): half-way
    # down the fall. Design value of IDEAS 2, fixed before any run. DELTA_KNEE carries the lag
    # from this frame to the onset.
    KNEE_LEVEL_FRAC: float = 0.5
    # Never earlier than the start of the ball gap that holds the entry (the last sighting).
    # Advisor's ruling: on for BAR-ATH. Off on Metrica only as the control of the acceptance
    # test, because Metrica blanks the ball at the onset, so its gap start is the onset.
    KNEE_GAP_CLAMP: bool = True
    # Variant 2 (COMPONENT_KNEE.md Section 8, designed after variant 1 failed criterion (v)):
    # a re-timed entry never cuts the possession spell that ends at it below this, s. If it
    # would, the entry moves later to keep the spell at this length, never beyond the detected
    # entry (possession.spell_floors, one more decode). The value is the 1 s threshold of
    # criterion (v) itself (class F, not selected). 0 gives variant 1.
    KNEE_MIN_PRE_SPELL_S: float = 1.0

    # -- K2: owner changes moved back to where the new team took the ball ----------------------
    # K2 of RESEARCH-4 (Section 4.1, pre-registration calibration/COMPONENT_BACKDATE.md,
    # possession.backdate_changes). The engine changes the owner at an owner event of the new
    # team. When that event is a later kick, control run or dribble, the change comes late: the
    # error anatomy of RESEARCH-4 (Section 2) puts the late class at about 0.05 spellF1 on both
    # providers. The rule walks back from each direct H <-> A change of the decoded labels while
    # the ball is visible, the nearest player of the new team is within BACKDATE_RADIUS_M of the
    # ball and that team is the nearer team. It gives the new team the frames of that stretch.
    # It never crosses a block edge, never enters or leaves D, and never adds or removes a
    # change. False gives the c22 output byte for byte. Read in run() only. On since 10 Nov 2026
    # (c23_backdate): K2b's frozen acceptance test passed (calibration/COMPONENT_BACKDATE_B.md
    # Section 7: Metrica +0.0054 raw, +0.0077 matched; DFL +0.0023, +0.0044; 3 of 3 matches; the
    # BAR-ATH vetoes pass; criterion (v) with one reference-edge exemption).
    BACKDATE_CHANGES: bool = True
    # Touch range of the walk, m. Class F, tied to R_TOUCH (PLAN 3.3): the new team "had the
    # ball" while one of its players was within touch range of it.
    BACKDATE_RADIUS_M: float = 1.5
    # Longest move, s (RESEARCH-4 4.1.5). Class E: the 90th percentile of the lag of the late
    # changes of the c22 engine on METRICA-S1 to S3, raw and matched views pooled
    # (COMPONENT_BACKDATE.md Sections 3 and 7: n = 577 lags, 70 frames). Measured on 10 Nov 2026
    # before any switch-on run.
    BACKDATE_CAP_S: float = 2.8
    # Shortest move, s. Class F: 2 x R_TOUCH / AIRBORNE_SPEED = 0.2 s (5 frames at 25 fps). A ball
    # at or above AIRBORNE_SPEED (suspected airborne, PLAN 3.2) crosses the touch disc in less
    # than this, so a shorter stretch of proximity can be a ball in flight over the player; a
    # longer one cannot. The splice already used 5 frames; the formula ties the value, it did not
    # choose it. This is why the walk does not read the airborne flag.
    BACKDATE_MIN_S: float = 0.2
    # The spell before a moved change never gets shorter than this, s (RESEARCH-4 4.1.5). If it
    # would, the walk stops there. Class F: the 1 s threshold of acceptance criterion (v), as
    # KNEE_MIN_PRE_SPELL_S (COMPONENT_KNEE.md Section 8). A spell already shorter keeps its
    # change where it is.
    BACKDATE_MIN_PRE_SPELL_S: float = 1.0
    # K2b (calibration/COMPONENT_BACKDATE_B.md, RESEARCH-4 4.1): the new team's spell that starts
    # at the change, in the decoded labels, must last at least this, s, or the change is not
    # moved. Class F: the same 1 s threshold as BACKDATE_MIN_PRE_SPELL_S. With both floors the
    # rule never moves the start or the end of a spell shorter than 1 s. All four short spells
    # that failed K2's criterion (v) were spells under 1 s whose start the walk moved
    # (research/C23_CYCLE.md Section 2).
    BACKDATE_MIN_POST_SPELL_S: float = 1.0

    # -- learned stage: engine/learned.py (LEARNED_STAGE_PROTOCOL.md Sections 1, 2, 6) --------
    # Which labels possession.run() writes. "off" writes the labels of the PLAN 3.7 decoder (c22)
    # and loads no learned artefact. "model" writes the labels of the learned ensemble alone,
    # decoded by a three-state Viterbi (candidate M). "hybrid" keeps the alive and dead frames of
    # c22 and takes the team on each alive run of c22 from the ensemble (candidate H). The
    # protocol (Section 5) chooses the mode on external matches. Off on branch learned-r6.
    # Round 6 candidate E2, adopted by the owner (docs/plan/calibration/
    # LEARNED_STAGE_RESULT_R6.md Sections 1 and 2): the network (feature version stage50-v2t,
    # the attack direction that compares the two teams) averaged with the per-frame member at
    # weight 0.25, costs (12, 12), no prior. It replaced candidate E (stage50-v2s) on the
    # external tests of the round's rule; E's weights are in the git history (learned_v6.npz).
    LEARNED_MODE: str = "model"
    # The ensemble weights, a .npz file relative to the repository root. The engine loads it
    # with numpy and never trains (PLAN 3.8). tools/learned_spike/export.py writes the file. The
    # protocol evaluation (Section 4) passes the weights of each fold through --config. The
    # final weights of Section 6 go to this path. Read only when LEARNED_MODE is not "off".
    LEARNED_WEIGHTS: str = "engine/models/learned_v7.npz"
    # Viterbi switch costs of candidate M, in log-probability units, as in the PLAN 3.7
    # decoder: a change between H and A, and a change between an alive state and D.
    # Round 6 (LEARNED_STAGE_RESULT_R6.md Section 6): (12, 12), within 0.0002 of the best pair of
    # round 4's rule (the mean gain on left-out Metrica and DFL); scores within 0.002 are tied.
    # Before: the placeholders 32 and 24, the median of the spike folds (folds_v1.json).
    LEARNED_COST_HA: float = 12.0
    LEARNED_COST_AD: float = 12.0
    # Viterbi switch cost between H and A of candidate H, on the alive runs of c22 (PLAN 3.7).
    # Placeholder: the spike hybrid used the H-A cost of each fold, median 32
    # (tools/learned_spike/hybrid.py). The protocol (Sections 3 and 6) replaces it.
    LEARNED_COST_HA_HYBRID: float = 32.0
    # Match structure in mode "model" (engine.learned.decode_model): the frames of a graded block
    # before its kickoff of games.py are forced to D in the Viterbi, as c22/c23 force them in
    # possession.decode_blocks() (PLAN 3.4; challenge FAQ: the 2,934 frames before the
    # second-half kickoff "are graded and all of them are D"). No new constant: the kickoff
    # frames are data.kickoff_frames(). The hybrid mode keeps the D frames of c22 and is not
    # affected. Off keeps the frozen round 3 decoder (R3_DEVIATION_STRUCTURE.md).
    LEARNED_STRUCTURE: bool = True
    # Rule-engine prior in mode "model" (engine.learned.apply_prior), in natural-log units, the
    # units of the emission scores and switch costs of the PLAN 3.7 Viterbi: on every frame
    # where the labels of LEARNED_MODE "off" (possession.run() steps 1 to 8, after back-dating)
    # say H, A or D, this bonus is added to the log-probability of that class before the
    # Viterbi; X frames get none. 0.0 is off and decodes exactly as before. Inert in modes
    # "off" and "hybrid". Evaluated in round 4; 0 for round 6 (LEARNED_STAGE_RESULT_R6.md
    # Section 6).
    LEARNED_PRIOR_BONUS: float = 0.0
    # The per-frame member (engine/learned_pf.py, round 6): gradient-boosted trees at 25 fps.
    # "on" averages its frame probabilities with the network's before the PLAN 3.7 decoder:
    # p = LEARNED_PF_WEIGHT x p_pf + (1 - LEARNED_PF_WEIGHT) x p_network, rounded again to
    # PROB_DECIMALS (engine.learned.blend_pf). "off" leaves every output as it was. On with weight
    # 0.25 for the round 6 candidate E2 (LEARNED_STAGE_RESULT_R6.md Sections 5 and 6).
    # Read only when LEARNED_MODE is "model" or "hybrid".
    LEARNED_PF: str = "on"
    LEARNED_PF_WEIGHT: float = 0.25
    LEARNED_PF_WEIGHTS: str = "engine/models/learned_pf_v7.npz"
    # Kickoff team in mode "model" (engine.learned.kickoff_teams, kickoff_team_mask): from each
    # half's kickoff frame of games.py the first LEARNED_KICKOFF_FLOOR_S seconds are forced to
    # the team that takes the kickoff. Half 1: the team of the first alive frame that the
    # decoder gives without this rule. Half 2: the other team (Laws of the Game, Law 8: the team
    # that did not take the first-half kickoff takes the second-half kickoff). After the floor
    # the Viterbi decides. The floor of 1 s is a fixed assumption (the kicking team plays the
    # ball first and keeps it for at least the first pass), set before any run (round 8 arm K).
    # On by default since round 8 (candidate E3 = E2 + this rule; LEARNED_STAGE_RESULT_R8.md):
    # on the competition match it changes frames 113741..113761 from A to H. False writes E2's
    # output byte for byte.
    LEARNED_KICKOFF_TEAM: bool = True
    LEARNED_KICKOFF_FLOOR_S: float = 1.0

    # -- full-match sanity vetoes (PLAN.md 5.3) -------------------------------------------------
    # Reject a version outside these; they catch pathology, they are not a ranking.
    VETO_POSSESSIONS_MIN: int = 180
    VETO_POSSESSIONS_MAX: int = 450
    VETO_D_SHARE_MIN: float = 0.18
    VETO_D_SHARE_MAX: float = 0.55
    VETO_MIN_D_SEGMENTS: int = 55      # 59 tagged set pieces minus margin (PLAN 5.3)
    VETO_MAX_GRADED_X: int = 0         # never X on a graded frame (team rule, PLAN 1.7)

    @classmethod
    def default(cls):
        """The configuration `python -m engine.run` uses when no --config is given."""
        return cls()

    def as_dict(self):
        """{field: value}, in declaration order."""
        return dataclasses.asdict(self)

    def config_hash(self):
        """Short sha1 over repr of the (name, value) pairs sorted by name; stable across runs."""
        items = sorted(self.as_dict().items())
        digest = hashlib.sha1(repr(items).encode("utf-8")).hexdigest()
        return digest[:CONFIG_HASH_LENGTH]


def load(path):
    """Config.default() with the overrides of a JSON file {FIELD: value} applied.

    Unknown fields raise (a typo must not silently fall back to the default). Values are cast to
    the default's type, so `"DV_TOUCH": 2` stays a float and the hash does not depend on it.
    """
    with open(path, "r") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError("{}: expected a JSON object of overrides".format(path))
    base = Config.default()
    known = base.as_dict()
    unknown = sorted(set(overrides) - set(known))
    if unknown:
        raise ValueError("{}: unknown config fields {}".format(path, unknown))
    cast = {}
    for name, value in overrides.items():
        kind = type(known[name])
        if kind is bool and not isinstance(value, bool):
            raise ValueError("{}: {} must be true/false".format(path, name))
        cast[name] = kind(value)
    return dataclasses.replace(base, **cast)
