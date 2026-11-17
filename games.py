"""Per-game constants shared by the evaluation scripts and the starter tooling.

This module deliberately imports nothing beyond the standard library, so it can be loaded by any
interpreter and by code that has no numpy or pandas available.

The alignment is per half: id_frame = video_frame + OFFSETS[half]. The two halves genuinely
disagree, so no single integer offset is frame-accurate for both, and frame accuracy is what the
evaluation in evaluation/ needs -- a one-frame error moves every transition it scores.

Everything a submission contains is indexed by *video* frame. The alignment below only matters
when you read the tracking data, which runs on its own counter.

Data location
-------------
The match assets are gated: they are not in this repository. Download them (see DATA.md) and put
them in ./data, or point MTAH_DATA_DIR somewhere else:

    export MTAH_DATA_DIR=/path/to/mtah2027        # macOS / Linux
    $env:MTAH_DATA_DIR = "D:\\mtah2027"           # Windows PowerShell

The evaluation itself needs none of this -- it reads only CSVs -- so a missing data directory is
not an error here.
"""

import os

GAME = "BAR-ATH"

# Root of the downloaded data bundle. Nothing in this module opens it; the paths below are built
# so that tooling which does need the video has one place to look.
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")

GAMES = {
    "BAR-ATH": {
        # The match comes from two cameras, each with its own folder, its own stem and its own
        # timeline -- see SOURCES below and asset_paths(). The tactical video is the frame index
        # authority for this challenge: row i of a submission is frame i of that file. The
        # broadcast recording is an optional extra input and is NOT aligned to it.
        #
        # n_video_frames, fps and pitch describe the TACTICAL source, because that is the
        # submission contract; per-source counts live in SOURCES.
        "n_video_frames": 199319,
        "fps": 25,
        # id_frame = video_frame + offsets[half], measured frame by frame against the video.
        #
        # The halves differ by one frame. A single offset therefore cannot be exact for both,
        # which is why this is a per-half table and not a scalar. It started life as one number
        # (1702056) derived from the first-half kickoff, which was good enough for watching the
        # video and not good enough for scoring transition timing to the frame.
        "offsets": {1: 1702057, 2: 1702058},
        # IdFrame span of each half, read off the two contiguous blocks in the tracking data (the
        # only gap is halftime). These are what decide which offset applies to a frame, so they
        # are data, not decoration.
        "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
        # Kickoff anchors, kept as convenient playback start positions. They are observations
        # about the video -- "the ball is struck on this frame" -- so they do not move when the
        # offsets are corrected, and they do not feed the offsets.
        "kickoff": {"minute": 7, "second": 27, "frame": 11},
        "kickoff_2nd": {"minute": 75, "second": 49, "frame": 16},
        # Real pitch size in metres.
        "pitch": (105.3, 68.0),
    },
    # Metrica Sports sample data, Sample Game 3 (github.com/metrica-sports/sample-data), used for
    # the generalisation smoke test of PLAN.md 5.5. Every value comes from its metadata XML or from
    # its tracking file. The sample has no video, so its video frame is the tracking frame_id minus
    # one, and one offset of 1 holds for both halves. The two halves are contiguous: there is no
    # halftime gap in the frame numbers. The sample has no Smart Tagging file and no homography.
    "METRICA-S3": {
        "n_video_frames": 143761,
        "fps": 25,
        "offsets": {1: 1, 2: 1},
        # first_half_start..first_half_end and second_half_start..second_half_end of the XML.
        "halves": {1: (1, 69661), 2: (69662, 143761)},
        # The ball first appears on row 360 of the tracking file, at the centre. The first KICK OFF
        # event is at frame_id 361. The second-half KICK OFF event is at frame_id 70090.
        "kickoff": {"minute": 0, "second": 14, "frame": 10},
        "kickoff_2nd": {"minute": 46, "second": 43, "frame": 14},
        # The first half block starts at frame_id 1, 360 frames before the kickoff. So the
        # half-1 kickoff is the "kickoff" anchor (engine.data.KICKOFF_ANCHOR_FIELD).
        "kickoff_from_anchor": True,
        # <FieldSize> of the metadata XML.
        "pitch": (105.0, 68.0),
        # The files of the sample, relative to MTAH_DATA_DIR. Set MTAH_DATA_DIR to the data/
        # folder of the sample-data clone. SOURCES below names the BAR-ATH files only, so
        # engine/data.py reads this field first (engine.data.ASSETS_FIELD).
        "assets": {
            "atd_meta": "Sample_Game_3/Sample_Game_3_metadata.xml",
            "atd_raw": "Sample_Game_3/Sample_Game_3_tracking.txt",
        },
    },
    # Metrica Sports sample data, Sample Game 1 (github.com/metrica-sports/sample-data), an
    # external reference of PLAN.md Phase 9 (P9-A). The sample ships per-team tracking CSVs.
    # tools/metrica_bundle.py converts them to an ATD bundle under MTAH_DATA_DIR. Every value
    # comes from Sample_Game_1_RawTrackingData_{Home,Away}_Team.csv or from
    # Sample_Game_1_RawEventsData.csv. The sample has no video, so its video frame is the tracking
    # Frame minus one, and one offset of 1 holds for both halves. The halves are contiguous in
    # the frame numbers: the files leave out the half-time break. No Smart Tagging, no homography.
    "METRICA-S1": {
        # Data rows of each tracking CSV. The Frame column runs 1..145006 without a gap.
        "n_video_frames": 145006,
        # Time [s] = Frame / 25 on every row.
        "fps": 25,
        "offsets": {1: 1, 2: 1},
        # First and last Frame of each value of the Period column.
        "halves": {1: (1, 71268), 2: (71269, 145006)},
        # The first KICK OFF event starts at Frame 1 (video frame 0). The first KICK OFF event
        # of period 2 starts at Frame 72135 (video frame 72134).
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 48, "second": 5, "frame": 9},
        # The half-1 kickoff is the "kickoff" anchor, as for METRICA-S3. For this sample the
        # anchor and the first graded frame are the same frame.
        "kickoff_from_anchor": True,
        # Metrica sample-data README: the field is 105 x 68 m for Sample Games 1 and 2.
        "pitch": (105.0, 68.0),
        # The bundle that tools/metrica_bundle.py writes, relative to MTAH_DATA_DIR. Set
        # MTAH_DATA_DIR to the root of the external references (~/mtah2027/refs in P9-A).
        "assets": {
            "atd_meta": "metrica/Sample_Game_1/Sample_Game_1_atd.xml",
            "atd_raw": "metrica/Sample_Game_1/Sample_Game_1_atd.txt",
        },
    },
    # Metrica Sports sample data, Sample Game 2. The same source, layout and conversion as
    # METRICA-S1. Every value comes from Sample_Game_2_RawTrackingData_{Home,Away}_Team.csv or
    # from Sample_Game_2_RawEventsData.csv.
    "METRICA-S2": {
        # Data rows of each tracking CSV. The Frame column runs 1..141156 without a gap.
        "n_video_frames": 141156,
        # Time [s] = Frame / 25 on every row.
        "fps": 25,
        "offsets": {1: 1, 2: 1},
        # First and last Frame of each value of the Period column.
        "halves": {1: (1, 67941), 2: (67942, 141156)},
        # The first KICK OFF event starts at Frame 51 (video frame 50). The first KICK OFF event
        # of period 2 starts at Frame 67992 (video frame 67991). It has the RETAKEN subtype: the
        # ball is played, then the kick-off is taken again at Frame 68416.
        "kickoff": {"minute": 0, "second": 2, "frame": 0},
        "kickoff_2nd": {"minute": 45, "second": 19, "frame": 16},
        # The first half block starts at Frame 1, 50 frames before the kickoff.
        "kickoff_from_anchor": True,
        # Metrica sample-data README: the field is 105 x 68 m for Sample Games 1 and 2.
        "pitch": (105.0, 68.0),
        # Relative to MTAH_DATA_DIR, as for METRICA-S1.
        "assets": {
            "atd_meta": "metrica/Sample_Game_2/Sample_Game_2_atd.xml",
            "atd_raw": "metrica/Sample_Game_2/Sample_Game_2_atd.txt",
        },
    },
    # -- Sportec Open DFL references (P9-A) ---
    # Seven matches of the Sportec Open DFL dataset (Bassek et al. 2025, Sci Data 12, 195, CC-BY
    # 4.0, figshare DOI 10.6084/m9.figshare.28196177). We use them as external references for
    # validation (docs/plan/calibration/DFL_validation.md). Two of them are sealed
    # (docs/plan/calibration/SPLITS.md). tools/dfl_bundle.py converts each match to the files
    # that "assets" names and prints the entry. Every value comes from the match files, as the
    # comments say. The video frame is the DFL frame counter N minus the first N of half 1, for
    # both halves. The halftime jump of N becomes frames without data, and the reference labels
    # them X. The matches have no video. Set MTAH_DATA_DIR to the folder that the "assets" paths
    # start from (~/mtah2027/refs on our machines).
    "DFL-J03WMX": {
        # 1. FC Köln (home, H) v FC Bayern München (guest, A). DFL-MAT-J03WMX.
        "n_video_frames": 165259,  # last N of half 2 - offset + 1
        "fps": 25,  # step of the T times of the ball frames
        "offsets": {1: 10000, 2: 10000},  # first N of half 1
        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.
        "halves": {1: (10000, 80707), 2: (100000, 175258)},
        # First frame of each half with BallStatus 1.
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 60, "second": 0, "frame": 0},
        "pitch": (105.0, 68.0),  # PitchX, PitchY of the match information
        "assets": {
            "atd_meta": "dfl/J03WMX/DFL-J03WMX_atd.xml",
            "atd_raw": "dfl/J03WMX/DFL-J03WMX_atd.txt",
        },
    },
    "DFL-J03WN1": {
        # VfL Bochum 1848 (home, H) v Bayer 04 Leverkusen (guest, A). DFL-MAT-J03WN1.
        "n_video_frames": 162273,  # last N of half 2 - offset + 1
        "fps": 25,  # step of the T times of the ball frames
        "offsets": {1: 10000, 2: 10000},  # first N of half 1
        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.
        "halves": {1: (10000, 79287), 2: (100000, 172272)},
        # First frame of each half with BallStatus 1.
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 60, "second": 0, "frame": 1},
        "pitch": (105.0, 68.0),  # PitchX, PitchY of the match information
        "assets": {
            "atd_meta": "dfl/J03WN1/DFL-J03WN1_atd.xml",
            "atd_raw": "dfl/J03WN1/DFL-J03WN1_atd.txt",
        },
    },
    "DFL-J03WOH": {
        # Fortuna Düsseldorf (home, H) v SSV Jahn Regensburg (guest, A). DFL-MAT-J03WOH.
        "n_video_frames": 157790,  # last N of half 2 - offset + 1
        "fps": 25,  # step of the T times of the ball frames
        "offsets": {1: 10000, 2: 10000},  # first N of half 1
        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.
        "halves": {1: (10000, 79423), 2: (100000, 167789)},
        # First frame of each half with BallStatus 1.
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 60, "second": 1, "frame": 23},
        "pitch": (105.0, 68.0),  # PitchX, PitchY of the match information
        "assets": {
            "atd_meta": "dfl/J03WOH/DFL-J03WOH_atd.xml",
            "atd_raw": "dfl/J03WOH/DFL-J03WOH_atd.txt",
        },
    },
    "DFL-J03WOY": {
        # Fortuna Düsseldorf (home, H) v F.C. Hansa Rostock (guest, A). DFL-MAT-J03WOY.
        "n_video_frames": 163425,  # last N of half 2 - offset + 1
        "fps": 25,  # step of the T times of the ball frames
        "offsets": {1: 10000, 2: 10000},  # first N of half 1
        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.
        "halves": {1: (10000, 79110), 2: (100000, 173424)},
        # First frame of each half with BallStatus 1.
        "kickoff": {"minute": 0, "second": 0, "frame": 1},
        "kickoff_2nd": {"minute": 60, "second": 0, "frame": 0},
        "kickoff_from_anchor": True,  # half 1 does not start alive
        "pitch": (105.0, 68.0),  # PitchX, PitchY of the match information
        "assets": {
            "atd_meta": "dfl/J03WOY/DFL-J03WOY_atd.xml",
            "atd_raw": "dfl/J03WOY/DFL-J03WOY_atd.txt",
        },
    },
    "DFL-J03WPY": {
        # Fortuna Düsseldorf (home, H) v 1. FC Nürnberg (guest, A). DFL-MAT-J03WPY.
        "n_video_frames": 167080,  # last N of half 2 - offset + 1
        "fps": 25,  # step of the T times of the ball frames
        "offsets": {1: 10000, 2: 10000},  # first N of half 1
        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.
        "halves": {1: (10000, 79130), 2: (100000, 177079)},
        # First frame of each half with BallStatus 1.
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 60, "second": 0, "frame": 0},
        "pitch": (105.0, 68.0),  # PitchX, PitchY of the match information
        "assets": {
            "atd_meta": "dfl/J03WPY/DFL-J03WPY_atd.xml",
            "atd_raw": "dfl/J03WPY/DFL-J03WPY_atd.txt",
        },
    },
    "DFL-J03WQQ": {
        # Fortuna Düsseldorf (home, H) v FC St. Pauli (guest, A). DFL-MAT-J03WQQ.
        "n_video_frames": 163141,  # last N of half 2 - offset + 1
        "fps": 25,  # step of the T times of the ball frames
        "offsets": {1: 10000, 2: 10000},  # first N of half 1
        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.
        "halves": {1: (10000, 79203), 2: (100000, 173140)},
        # First frame of each half with BallStatus 1.
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 60, "second": 0, "frame": 0},
        "pitch": (105.0, 68.0),  # PitchX, PitchY of the match information
        "assets": {
            "atd_meta": "dfl/J03WQQ/DFL-J03WQQ_atd.xml",
            "atd_raw": "dfl/J03WQQ/DFL-J03WQQ_atd.txt",
        },
    },
    "DFL-J03WR9": {
        # Fortuna Düsseldorf (home, H) v 1. FC Kaiserslautern (guest, A). DFL-MAT-J03WR9.
        "n_video_frames": 168456,  # last N of half 2 - offset + 1
        "fps": 25,  # step of the T times of the ball frames
        "offsets": {1: 10000, 2: 10000},  # first N of half 1
        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.
        "halves": {1: (10000, 78353), 2: (100000, 178455)},
        # First frame of each half with BallStatus 1.
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 60, "second": 0, "frame": 0},
        "pitch": (105.0, 68.0),  # PitchX, PitchY of the match information
        "assets": {
            "atd_meta": "dfl/J03WR9/DFL-J03WR9_atd.xml",
            "atd_raw": "dfl/J03WR9/DFL-J03WR9_atd.txt",
        },
    },
    # -- SkillCorner Open Data references ---
    # Twenty matches of SkillCorner Open Data (github.com/SkillCorner/opendata, commit 4340d27,
    # MIT licence, copyright SkillCorner), A-League 2024/2025. We use them as a third external
    # reference beside Metrica and DFL, for validation only
    # (docs/plan/calibration/SKILLCORNER_REFERENCES.md). tools/skillcorner_bundle.py converts each
    # match from 10 Hz to 25 Hz and prints the entry. Every value comes from the match files, as
    # the comments say. The video frame is 2.5 times the tracking frame. Frames outside the two
    # periods have no data, and the reference labels them X. The matches have no video. Set
    # MTAH_DATA_DIR to the folder that the "assets" paths start from (~/mtah2027/refs on our
    # machines).
    "SC-1874553": {
        # Home (H): Brisbane Roar FC. Away (A): Adelaide United Football Club.
        # SkillCorner match 1874553.
        "n_video_frames": 150751,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..28990 and 29000..60300 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 72476), 2: (72501, 150751)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 48, "second": 20, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1874553/SC-1874553_atd.xml",
            "atd_raw": "skillcorner/1874553/SC-1874553_atd.txt",
        },
    },
    "SC-1886347": {
        # Home (H): Auckland FC. Away (A): Newcastle United Jets FC.
        # SkillCorner match 1886347.
        "n_video_frames": 147651,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..27790 and 27800..59060 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 69476), 2: (69501, 147651)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 46, "second": 20, "frame": 0},
        "pitch": (104.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1886347/SC-1886347_atd.xml",
            "atd_raw": "skillcorner/1886347/SC-1886347_atd.txt",
        },
    },
    "SC-1899585": {
        # Home (H): Auckland FC. Away (A): Wellington Phoenix FC.
        # SkillCorner match 1899585.
        "n_video_frames": 151323,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..29760 and 29770..60529 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 74401), 2: (74426, 151323)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 49, "second": 37, "frame": 0},
        "pitch": (104.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1899585/SC-1899585_atd.xml",
            "atd_raw": "skillcorner/1899585/SC-1899585_atd.txt",
        },
    },
    "SC-1925299": {
        # Home (H): Brisbane Roar FC. Away (A): Perth Glory Football Club.
        # SkillCorner match 1925299.
        "n_video_frames": 153251,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..30240 and 30250..61300 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 75601), 2: (75626, 153251)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 50, "second": 25, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1925299/SC-1925299_atd.xml",
            "atd_raw": "skillcorner/1925299/SC-1925299_atd.txt",
        },
    },
    "SC-1927964": {
        # Home (H): Western Sydney Wanderers FC. Away (A): Wellington Phoenix FC.
        # SkillCorner match 1927964.
        "n_video_frames": 147673,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..28380 and 28390..59069 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 70951), 2: (70976, 147673)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 47, "second": 19, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1927964/SC-1927964_atd.xml",
            "atd_raw": "skillcorner/1927964/SC-1927964_atd.txt",
        },
    },
    "SC-1953632": {
        # Home (H): Central Coast Mariners Football Club. Away (A): Melbourne City FC.
        # SkillCorner match 1953632.
        "n_video_frames": 148123,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..28970 and 28980..59249 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 72426), 2: (72451, 148123)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 48, "second": 18, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1953632/SC-1953632_atd.xml",
            "atd_raw": "skillcorner/1953632/SC-1953632_atd.txt",
        },
    },
    "SC-1959846": {
        # Home (H): Melbourne Victory Football Club. Away (A): Western United.
        # SkillCorner match 1959846.
        "n_video_frames": 146351,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 0..27960 and 27970..58540 (`period` 1 and 2), times 2.5.
        "halves": {1: (1, 69901), 2: (69926, 146351)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 46, "second": 37, "frame": 0},
        "pitch": (106.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1959846/SC-1959846_atd.xml",
            "atd_raw": "skillcorner/1959846/SC-1959846_atd.txt",
        },
    },
    "SC-1986691": {
        # Home (H): Western Sydney Wanderers FC. Away (A): Auckland FC.
        # SkillCorner match 1986691.
        "n_video_frames": 147723,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..27780 and 27790..59089 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 69451), 2: (69476, 147723)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 46, "second": 19, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1986691/SC-1986691_atd.xml",
            "atd_raw": "skillcorner/1986691/SC-1986691_atd.txt",
        },
    },
    "SC-1996435": {
        # Home (H): Sydney Football Club. Away (A): Adelaide United Football Club.
        # SkillCorner match 1996435.
        "n_video_frames": 144051,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..28020 and 28030..57620 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 70051), 2: (70076, 144051)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 46, "second": 43, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1996435/SC-1996435_atd.xml",
            "atd_raw": "skillcorner/1996435/SC-1996435_atd.txt",
        },
    },
    "SC-1996436": {
        # Home (H): Auckland FC. Away (A): Macarthur FC.
        # SkillCorner match 1996436.
        "n_video_frames": 150698,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..30140 and 30150..60279 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 75351), 2: (75376, 150698)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 50, "second": 15, "frame": 0},
        "pitch": (104.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/1996436/SC-1996436_atd.xml",
            "atd_raw": "skillcorner/1996436/SC-1996436_atd.txt",
        },
    },
    "SC-2006229": {
        # Home (H): Melbourne City FC. Away (A): Macarthur FC.
        # SkillCorner match 2006229.
        "n_video_frames": 148173,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..28480 and 28490..59269 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 71201), 2: (71226, 148173)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 47, "second": 29, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2006229/SC-2006229_atd.xml",
            "atd_raw": "skillcorner/2006229/SC-2006229_atd.txt",
        },
    },
    "SC-2006363": {
        # Home (H): Melbourne Victory Football Club.
        # Away (A): Central Coast Mariners Football Club.
        # SkillCorner match 2006363.
        "n_video_frames": 147726,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..28950 and 28960..59090 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 72376), 2: (72401, 147726)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 48, "second": 16, "frame": 0},
        "pitch": (106.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2006363/SC-2006363_atd.xml",
            "atd_raw": "skillcorner/2006363/SC-2006363_atd.txt",
        },
    },
    "SC-2007448": {
        # Home (H): Perth Glory Football Club. Away (A): Brisbane Roar FC.
        # SkillCorner match 2007448.
        "n_video_frames": 145201,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..28530 and 28540..58080 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 71326), 2: (71351, 145201)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 47, "second": 34, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2007448/SC-2007448_atd.xml",
            "atd_raw": "skillcorner/2007448/SC-2007448_atd.txt",
        },
    },
    "SC-2007721": {
        # Home (H): Auckland FC. Away (A): Central Coast Mariners Football Club.
        # SkillCorner match 2007721.
        "n_video_frames": 143901,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..27710 and 27720..57560 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 69276), 2: (69301, 143901)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 46, "second": 12, "frame": 0},
        "pitch": (104.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2007721/SC-2007721_atd.xml",
            "atd_raw": "skillcorner/2007721/SC-2007721_atd.txt",
        },
    },
    "SC-2010085": {
        # Home (H): Melbourne City FC. Away (A): Central Coast Mariners Football Club.
        # SkillCorner match 2010085.
        "n_video_frames": 147476,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 10..29440 and 29450..58990 (`period` 1 and 2), times 2.5.
        "halves": {1: (26, 73601), 2: (73626, 147476)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 0, "second": 1, "frame": 0},
        "kickoff_2nd": {"minute": 49, "second": 5, "frame": 0},
        "pitch": (106.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2010085/SC-2010085_atd.xml",
            "atd_raw": "skillcorner/2010085/SC-2010085_atd.txt",
        },
    },
    "SC-2011166": {
        # Home (H): Wellington Phoenix FC. Away (A): Melbourne Victory Football Club.
        # SkillCorner match 2011166.
        "n_video_frames": 178476,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 2300..31290 and 41090..71390 (`period` 1 and 2), times 2.5.
        "halves": {1: (5751, 78226), 2: (102726, 178476)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 3, "second": 50, "frame": 0},
        "kickoff_2nd": {"minute": 68, "second": 29, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2011166/SC-2011166_atd.xml",
            "atd_raw": "skillcorner/2011166/SC-2011166_atd.txt",
        },
    },
    "SC-2013725": {
        # Home (H): Western United. Away (A): Sydney Football Club.
        # SkillCorner match 2013725.
        "n_video_frames": 174651,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 4040..31930 and 40960..69860 (`period` 1 and 2), times 2.5.
        "halves": {1: (10101, 79826), 2: (102401, 174651)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 6, "second": 44, "frame": 0},
        "kickoff_2nd": {"minute": 68, "second": 16, "frame": 0},
        "pitch": (106.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2013725/SC-2013725_atd.xml",
            "atd_raw": "skillcorner/2013725/SC-2013725_atd.txt",
        },
    },
    "SC-2015213": {
        # Home (H): Western United. Away (A): Auckland FC.
        # SkillCorner match 2015213.
        "n_video_frames": 178376,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 2210..31040 and 40280..71350 (`period` 1 and 2), times 2.5.
        "halves": {1: (5526, 77601), 2: (100701, 178376)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 3, "second": 41, "frame": 0},
        "kickoff_2nd": {"minute": 67, "second": 8, "frame": 0},
        "pitch": (106.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2015213/SC-2015213_atd.xml",
            "atd_raw": "skillcorner/2015213/SC-2015213_atd.txt",
        },
    },
    "SC-2016236": {
        # Home (H): Western United. Away (A): Adelaide United Football Club.
        # SkillCorner match 2016236.
        "n_video_frames": 176376,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 1170..31720 and 41060..70550 (`period` 1 and 2), times 2.5.
        "halves": {1: (2926, 79301), 2: (102651, 176376)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 1, "second": 57, "frame": 0},
        "kickoff_2nd": {"minute": 68, "second": 26, "frame": 0},
        "pitch": (106.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2016236/SC-2016236_atd.xml",
            "atd_raw": "skillcorner/2016236/SC-2016236_atd.txt",
        },
    },
    "SC-2017461": {
        # Home (H): Melbourne Victory Football Club. Away (A): Auckland FC.
        # SkillCorner match 2017461.
        "n_video_frames": 176051,  # floor(2.5 * last tracking frame of period 2) + 1
        "fps": 25,  # resampled from 10 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Tracking frames 2510..30210 and 39670..70420 (`period` 1 and 2), times 2.5.
        "halves": {1: (6276, 75526), 2: (99176, 176051)},
        # First video frame of each half (SkillCorner starts each period clock there).
        "kickoff": {"minute": 4, "second": 11, "frame": 0},
        "kickoff_2nd": {"minute": 66, "second": 7, "frame": 0},
        "pitch": (105.0, 68.0),  # pitch_length, pitch_width of the match file
        "assets": {
            "atd_meta": "skillcorner/2017461/SC-2017461_atd.xml",
            "atd_raw": "skillcorner/2017461/SC-2017461_atd.txt",
        },
    },
    # -- PFF FC World Cup 2022 references ---
    # PFF FC FIFA World Cup 2022 data, broadcast tracking and event data, from the Drive folder
    # 1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa. No licence is stated; the team approved the use on
    # 9 November 2026 with credit to PFF FC (docs/plan/calibration/PFF_REFERENCES.md, which also
    # says how the matches may be used). A fourth external reference provider. tools/pff_bundle.py
    # converts each match from 29.97 Hz to 25 Hz and prints the entry. Only periods 1 and 2 are in
    # the bundle (no extra time). The matches have no video. Set MTAH_DATA_DIR to the folder that
    # the "assets" paths start from (~/mtah2027/refs on our machines). Excluded matches have no
    # entry (PFF_REFERENCES.md lists them with the reason).
    "PFF-3812": {
        # Home (H): Senegal. Away (A): Netherlands.
        # PFF FC game 3812. Tracking frameNum 4226..90021 and 91119..190179 (periods 1 and 2).
        "n_video_frames": 158641,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (3527, 75093), 2: (76010, 158641)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 21, "frame": 1},
        "kickoff_2nd": {"minute": 50, "second": 40, "frame": 9},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3812/PFF-3812_atd.xml",
            "atd_raw": "pff/3812/PFF-3812_atd.txt",
        },
    },
    "PFF-3813": {
        # Home (H): England. Away (A): Iran.
        # PFF FC game 3813. Tracking frameNum 5406..111551 and 114026..218466 (periods 1 and 2).
        "n_video_frames": 182237,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4511, 93053), 2: (95118, 182237)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 0, "frame": 10},
        "kickoff_2nd": {"minute": 63, "second": 24, "frame": 17},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3813/PFF-3813_atd.xml",
            "atd_raw": "pff/3813/PFF-3813_atd.txt",
        },
    },
    "PFF-3814": {
        # Home (H): Qatar. Away (A): Ecuador.
        # PFF FC game 3814. Tracking frameNum 3267..93198 and 96084..186594 (periods 1 and 2).
        "n_video_frames": 155651,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2727, 77743), 2: (80152, 155651)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 49, "frame": 1},
        "kickoff_2nd": {"minute": 53, "second": 26, "frame": 1},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3814/PFF-3814_atd.xml",
            "atd_raw": "pff/3814/PFF-3814_atd.txt",
        },
    },
    "PFF-3815": {
        # Home (H): United States. Away (A): Wales.
        # PFF FC game 3815. Tracking frameNum 3233..91284 and 92457..192328 (periods 1 and 2).
        "n_video_frames": 160434,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2698, 76147), 2: (77126, 160434)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 47, "frame": 22},
        "kickoff_2nd": {"minute": 51, "second": 25, "frame": 0},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3815/PFF-3815_atd.xml",
            "atd_raw": "pff/3815/PFF-3815_atd.txt",
        },
    },
    "PFF-3816": {
        # Home (H): Argentina. Away (A): Saudi Arabia.
        # PFF FC game 3816. Tracking frameNum 3056..95422 and 96816..202590 (periods 1 and 2).
        "n_video_frames": 168994,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2551, 79598), 2: (80762, 168994)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 42, "frame": 0},
        "kickoff_2nd": {"minute": 53, "second": 50, "frame": 11},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3816/PFF-3816_atd.xml",
            "atd_raw": "pff/3816/PFF-3816_atd.txt",
        },
    },
    "PFF-3817": {
        # Home (H): Denmark. Away (A): Tunisia.
        # PFF FC game 3817. Tracking frameNum 3010..92340 and 93594..188111 (periods 1 and 2).
        "n_video_frames": 156916,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2512, 77027), 2: (78074, 156916)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 40, "frame": 11},
        "kickoff_2nd": {"minute": 52, "second": 2, "frame": 23},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3817/PFF-3817_atd.xml",
            "atd_raw": "pff/3817/PFF-3817_atd.txt",
        },
    },
    "PFF-3818": {
        # Home (H): Mexico. Away (A): Poland.
        # PFF FC game 3818. Tracking frameNum 1654..86275 and 90622..184583 (periods 1 and 2).
        "n_video_frames": 153973,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1381, 71968), 2: (75595, 153973)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 0, "second": 55, "frame": 5},
        "kickoff_2nd": {"minute": 50, "second": 23, "frame": 19},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3818/PFF-3818_atd.xml",
            "atd_raw": "pff/3818/PFF-3818_atd.txt",
        },
    },
    "PFF-3819": {
        # Home (H): France. Away (A): Australia.
        # PFF FC game 3819. Tracking frameNum 2980..95071 and 97554..191384 (periods 1 and 2).
        "n_video_frames": 159646,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2488, 79305), 2: (81378, 159646)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 39, "frame": 12},
        "kickoff_2nd": {"minute": 54, "second": 15, "frame": 2},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3819/PFF-3819_atd.xml",
            "atd_raw": "pff/3819/PFF-3819_atd.txt",
        },
    },
    "PFF-3820": {
        # Home (H): Morocco. Away (A): Croatia.
        # PFF FC game 3820. Tracking frameNum 3275..87727 and 92629..184276 (periods 1 and 2).
        "n_video_frames": 153717,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2733, 73179), 2: (77270, 153717)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 49, "frame": 7},
        "kickoff_2nd": {"minute": 51, "second": 30, "frame": 19},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3820/PFF-3820_atd.xml",
            "atd_raw": "pff/3820/PFF-3820_atd.txt",
        },
    },
    "PFF-3821": {
        # Home (H): Germany. Away (A): Japan.
        # PFF FC game 3821. Tracking frameNum 3558..95232 and 97230..192846 (periods 1 and 2).
        "n_video_frames": 160866,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2969, 79439), 2: (81108, 160866)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 58, "frame": 18},
        "kickoff_2nd": {"minute": 54, "second": 4, "frame": 7},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3821/PFF-3821_atd.xml",
            "atd_raw": "pff/3821/PFF-3821_atd.txt",
        },
    },
    "PFF-3822": {
        # Home (H): Spain. Away (A): Costa Rica.
        # PFF FC game 3822. Tracking frameNum 1366..91376 and 93328..188775 (periods 1 and 2).
        "n_video_frames": 157470,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1141, 76223), 2: (77853, 157470)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 0, "second": 45, "frame": 15},
        "kickoff_2nd": {"minute": 51, "second": 54, "frame": 2},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3822/PFF-3822_atd.xml",
            "atd_raw": "pff/3822/PFF-3822_atd.txt",
        },
    },
    "PFF-3823": {
        # Home (H): Belgium. Away (A): Canada.
        # PFF FC game 3823. Tracking frameNum 3491..93370 and 95865..185831 (periods 1 and 2).
        "n_video_frames": 155014,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2914, 77887), 2: (79969, 155014)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 56, "frame": 13},
        "kickoff_2nd": {"minute": 53, "second": 18, "frame": 18},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3823/PFF-3823_atd.xml",
            "atd_raw": "pff/3823/PFF-3823_atd.txt",
        },
    },
    "PFF-3824": {
        # Home (H): Switzerland. Away (A): Cameroon.
        # PFF FC game 3824. Tracking frameNum 2999..89018 and 90064..182351 (periods 1 and 2).
        "n_video_frames": 152111,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2504, 74256), 2: (75130, 152111)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 40, "frame": 3},
        "kickoff_2nd": {"minute": 50, "second": 5, "frame": 4},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3824/PFF-3824_atd.xml",
            "atd_raw": "pff/3824/PFF-3824_atd.txt",
        },
    },
    "PFF-3825": {
        # Home (H): Uruguay. Away (A): South Korea.
        # PFF FC game 3825. Tracking frameNum 5597..88391 and 89810..184745 (periods 1 and 2).
        "n_video_frames": 154108,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4670, 73733), 2: (74918, 154108)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 6, "frame": 19},
        "kickoff_2nd": {"minute": 49, "second": 56, "frame": 17},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3825/PFF-3825_atd.xml",
            "atd_raw": "pff/3825/PFF-3825_atd.txt",
        },
    },
    "PFF-3826": {
        # Home (H): Portugal. Away (A): Ghana.
        # PFF FC game 3826. Tracking frameNum 2908..87417 and 88670..187765 (periods 1 and 2).
        "n_video_frames": 156627,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2427, 72920), 2: (73967, 156627)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 37, "frame": 1},
        "kickoff_2nd": {"minute": 49, "second": 18, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3826/PFF-3826_atd.xml",
            "atd_raw": "pff/3826/PFF-3826_atd.txt",
        },
    },
    "PFF-3827": {
        # Home (H): Brazil. Away (A): Serbia.
        # PFF FC game 3827. Tracking frameNum 5059..87820 and 88874..182587 (periods 1 and 2).
        "n_video_frames": 152308,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4222, 73257), 2: (74137, 152308)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 48, "frame": 21},
        "kickoff_2nd": {"minute": 49, "second": 25, "frame": 11},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3827/PFF-3827_atd.xml",
            "atd_raw": "pff/3827/PFF-3827_atd.txt",
        },
    },
    "PFF-3828": {
        # Home (H): Wales. Away (A): Iran.
        # PFF FC game 3828. Tracking frameNum 3142..91999 and 92872..195586 (periods 1 and 2).
        "n_video_frames": 163151,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2623, 76743), 2: (77472, 163151)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 44, "frame": 22},
        "kickoff_2nd": {"minute": 51, "second": 38, "frame": 21},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3828/PFF-3828_atd.xml",
            "atd_raw": "pff/3828/PFF-3828_atd.txt",
        },
    },
    "PFF-3829": {
        # Home (H): Qatar. Away (A): Senegal.
        # PFF FC game 3829. Tracking frameNum 5851..99484 and 100360..192080 (periods 1 and 2).
        "n_video_frames": 160227,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4882, 82986), 2: (83718, 160227)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 15, "frame": 6},
        "kickoff_2nd": {"minute": 55, "second": 48, "frame": 17},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3829/PFF-3829_atd.xml",
            "atd_raw": "pff/3829/PFF-3829_atd.txt",
        },
    },
    "PFF-3830": {
        # Home (H): Netherlands. Away (A): Ecuador.
        # PFF FC game 3830. Tracking frameNum 2239..90359 and 93278..185256 (periods 1 and 2).
        "n_video_frames": 154535,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1869, 75375), 2: (77811, 154535)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 14, "frame": 18},
        "kickoff_2nd": {"minute": 51, "second": 52, "frame": 10},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3830/PFF-3830_atd.xml",
            "atd_raw": "pff/3830/PFF-3830_atd.txt",
        },
    },
    "PFF-3831": {
        # Home (H): England. Away (A): United States.
        # PFF FC game 3831. Tracking frameNum 2376..85775 and 87555..176729 (periods 1 and 2).
        "n_video_frames": 147422,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1983, 71551), 2: (73037, 147422)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 19, "frame": 7},
        "kickoff_2nd": {"minute": 48, "second": 41, "frame": 11},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3831/PFF-3831_atd.xml",
            "atd_raw": "pff/3831/PFF-3831_atd.txt",
        },
    },
    "PFF-3832": {
        # Home (H): Tunisia. Away (A): Australia.
        # PFF FC game 3832. Tracking frameNum 6313..96274 and 98105..190309 (periods 1 and 2).
        "n_video_frames": 158750,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (5268, 80309), 2: (81837, 158750)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 30, "frame": 17},
        "kickoff_2nd": {"minute": 54, "second": 33, "frame": 11},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3832/PFF-3832_atd.xml",
            "atd_raw": "pff/3832/PFF-3832_atd.txt",
        },
    },
    "PFF-3833": {
        # Home (H): Poland. Away (A): Saudi Arabia.
        # PFF FC game 3833. Tracking frameNum 4216..103160 and 104502..198137 (periods 1 and 2).
        "n_video_frames": 165279,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (3518, 86053), 2: (87174, 165279)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 20, "frame": 17},
        "kickoff_2nd": {"minute": 58, "second": 6, "frame": 23},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3833/PFF-3833_atd.xml",
            "atd_raw": "pff/3833/PFF-3833_atd.txt",
        },
    },
    "PFF-3834": {
        # Home (H): France. Away (A): Denmark.
        # PFF FC game 3834. Tracking frameNum 2817..89195 and 91067..184885 (periods 1 and 2).
        "n_video_frames": 154225,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2351, 74404), 2: (75967, 154225)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 34, "frame": 0},
        "kickoff_2nd": {"minute": 50, "second": 38, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3834/PFF-3834_atd.xml",
            "atd_raw": "pff/3834/PFF-3834_atd.txt",
        },
    },
    "PFF-3835": {
        # Home (H): Argentina. Away (A): Mexico.
        # PFF FC game 3835. Tracking frameNum 3010..93755 and 94814..187379 (periods 1 and 2).
        "n_video_frames": 156305,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2512, 78208), 2: (79092, 156305)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 40, "frame": 11},
        "kickoff_2nd": {"minute": 52, "second": 43, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3835/PFF-3835_atd.xml",
            "atd_raw": "pff/3835/PFF-3835_atd.txt",
        },
    },
    "PFF-3836": {
        # Home (H): Japan. Away (A): Costa Rica.
        # PFF FC game 3836. Tracking frameNum 2922..85579 and 88071..180359 (periods 1 and 2).
        "n_video_frames": 150450,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2439, 71387), 2: (73467, 150450)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 37, "frame": 13},
        "kickoff_2nd": {"minute": 48, "second": 58, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3836/PFF-3836_atd.xml",
            "atd_raw": "pff/3836/PFF-3836_atd.txt",
        },
    },
    "PFF-3837": {
        # Home (H): Belgium. Away (A): Morocco.
        # PFF FC game 3837. Tracking frameNum 6205..93774 and 94700..185101 (periods 1 and 2).
        "n_video_frames": 154405,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (5178, 78224), 2: (78997, 154405)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 27, "frame": 2},
        "kickoff_2nd": {"minute": 52, "second": 39, "frame": 21},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3837/PFF-3837_atd.xml",
            "atd_raw": "pff/3837/PFF-3837_atd.txt",
        },
    },
    "PFF-3838": {
        # Home (H): Croatia. Away (A): Canada.
        # PFF FC game 3838. Tracking frameNum 2819..92753 and 94934..186702 (periods 1 and 2).
        "n_video_frames": 155741,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2353, 77372), 2: (79192, 155741)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 34, "frame": 2},
        "kickoff_2nd": {"minute": 52, "second": 47, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3838/PFF-3838_atd.xml",
            "atd_raw": "pff/3838/PFF-3838_atd.txt",
        },
    },
    "PFF-3839": {
        # Home (H): Spain. Away (A): Germany.
        # PFF FC game 3839. Tracking frameNum 2498..87565 and 89659..180993 (periods 1 and 2).
        "n_video_frames": 150978,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2085, 73044), 2: (74792, 150978)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 23, "frame": 9},
        "kickoff_2nd": {"minute": 49, "second": 51, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3839/PFF-3839_atd.xml",
            "atd_raw": "pff/3839/PFF-3839_atd.txt",
        },
    },
    "PFF-3840": {
        # Home (H): Cameroon. Away (A): Serbia.
        # PFF FC game 3840. Tracking frameNum 3754..97965 and 98975..191075 (periods 1 and 2).
        "n_video_frames": 159389,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (3133, 81719), 2: (82563, 159389)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 5, "frame": 7},
        "kickoff_2nd": {"minute": 55, "second": 2, "frame": 12},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3840/PFF-3840_atd.xml",
            "atd_raw": "pff/3840/PFF-3840_atd.txt",
        },
    },
    "PFF-3841": {
        # Home (H): South Korea. Away (A): Ghana.
        # PFF FC game 3841. Tracking frameNum 5702..95740 and 96733..197270 (periods 1 and 2).
        "n_video_frames": 164556,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4758, 79864), 2: (80693, 164556)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 10, "frame": 7},
        "kickoff_2nd": {"minute": 53, "second": 47, "frame": 17},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3841/PFF-3841_atd.xml",
            "atd_raw": "pff/3841/PFF-3841_atd.txt",
        },
    },
    "PFF-3842": {
        # Home (H): Brazil. Away (A): Switzerland.
        # PFF FC game 3842. Tracking frameNum 1871..84562 and 86578..178127 (periods 1 and 2).
        "n_video_frames": 148588,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1562, 70539), 2: (72222, 148588)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 2, "frame": 11},
        "kickoff_2nd": {"minute": 48, "second": 8, "frame": 21},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3842/PFF-3842_atd.xml",
            "atd_raw": "pff/3842/PFF-3842_atd.txt",
        },
    },
    "PFF-3843": {
        # Home (H): Portugal. Away (A): Uruguay.
        # PFF FC game 3843. Tracking frameNum 2694..89026 and 90279..189419 (periods 1 and 2).
        "n_video_frames": 158007,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2249, 74263), 2: (75309, 158007)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 29, "frame": 23},
        "kickoff_2nd": {"minute": 50, "second": 12, "frame": 8},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3843/PFF-3843_atd.xml",
            "atd_raw": "pff/3843/PFF-3843_atd.txt",
        },
    },
    "PFF-3844": {
        # Home (H): Ecuador. Away (A): Senegal.
        # PFF FC game 3844. Tracking frameNum 1532..93234 and 96221..189625 (periods 1 and 2).
        "n_video_frames": 158179,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1279, 77773), 2: (80266, 158179)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 0, "second": 51, "frame": 3},
        "kickoff_2nd": {"minute": 53, "second": 30, "frame": 15},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3844/PFF-3844_atd.xml",
            "atd_raw": "pff/3844/PFF-3844_atd.txt",
        },
    },
    "PFF-3846": {
        # Home (H): Wales. Away (A): England.
        # PFF FC game 3846. Tracking frameNum 5281..95192 and 97099..185216 (periods 1 and 2).
        "n_video_frames": 154501,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4407, 79406), 2: (80998, 154501)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 56, "frame": 6},
        "kickoff_2nd": {"minute": 53, "second": 59, "frame": 22},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3846/PFF-3846_atd.xml",
            "atd_raw": "pff/3846/PFF-3846_atd.txt",
        },
    },
    "PFF-3847": {
        # Home (H): Iran. Away (A): United States.
        # PFF FC game 3847. Tracking frameNum 4024..97679 and 100088..198769 (periods 1 and 2).
        "n_video_frames": 165807,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (3358, 81481), 2: (83492, 165807)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 14, "frame": 7},
        "kickoff_2nd": {"minute": 55, "second": 39, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3847/PFF-3847_atd.xml",
            "atd_raw": "pff/3847/PFF-3847_atd.txt",
        },
    },
    "PFF-3848": {
        # Home (H): Australia. Away (A): Denmark.
        # PFF FC game 3848. Tracking frameNum 4702..89209 and 91885..183685 (periods 1 and 2).
        "n_video_frames": 153224,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (3924, 74416), 2: (76649, 153224)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 36, "frame": 23},
        "kickoff_2nd": {"minute": 51, "second": 5, "frame": 23},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3848/PFF-3848_atd.xml",
            "atd_raw": "pff/3848/PFF-3848_atd.txt",
        },
    },
    "PFF-3849": {
        # Home (H): Tunisia. Away (A): France.
        # PFF FC game 3849. Tracking frameNum 2062..87008 and 89411..193181 (periods 1 and 2).
        "n_video_frames": 161145,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1722, 72580), 2: (74585, 161145)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 8, "frame": 21},
        "kickoff_2nd": {"minute": 49, "second": 43, "frame": 9},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3849/PFF-3849_atd.xml",
            "atd_raw": "pff/3849/PFF-3849_atd.txt",
        },
    },
    "PFF-3850": {
        # Home (H): Poland. Away (A): Argentina.
        # PFF FC game 3850. Tracking frameNum 2470..88980 and 91176..182904 (periods 1 and 2).
        "n_video_frames": 152573,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2062, 74224), 2: (76057, 152573)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 22, "frame": 11},
        "kickoff_2nd": {"minute": 50, "second": 42, "frame": 6},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3850/PFF-3850_atd.xml",
            "atd_raw": "pff/3850/PFF-3850_atd.txt",
        },
    },
    "PFF-3851": {
        # Home (H): Saudi Arabia. Away (A): Mexico.
        # PFF FC game 3851. Tracking frameNum 5573..97665 and 99619..194236 (periods 1 and 2).
        "n_video_frames": 162025,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4650, 81469), 2: (83100, 162025)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 5, "frame": 24},
        "kickoff_2nd": {"minute": 55, "second": 23, "frame": 24},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3851/PFF-3851_atd.xml",
            "atd_raw": "pff/3851/PFF-3851_atd.txt",
        },
    },
    "PFF-3852": {
        # Home (H): Croatia. Away (A): Belgium.
        # PFF FC game 3852. Tracking frameNum 5578..93616 and 95334..183408 (periods 1 and 2).
        "n_video_frames": 152993,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4655, 78091), 2: (79526, 152993)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 6, "frame": 4},
        "kickoff_2nd": {"minute": 53, "second": 1, "frame": 0},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3852/PFF-3852_atd.xml",
            "atd_raw": "pff/3852/PFF-3852_atd.txt",
        },
    },
    "PFF-3853": {
        # Home (H): Canada. Away (A): Morocco.
        # PFF FC game 3853. Tracking frameNum 1300..91385 and 93745..181676 (periods 1 and 2).
        "n_video_frames": 151548,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1086, 76230), 2: (78200, 151548)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 0, "second": 43, "frame": 10},
        "kickoff_2nd": {"minute": 52, "second": 7, "frame": 24},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3853/PFF-3853_atd.xml",
            "atd_raw": "pff/3853/PFF-3853_atd.txt",
        },
    },
    "PFF-3854": {
        # Home (H): Japan. Away (A): Spain.
        # PFF FC game 3854. Tracking frameNum 2809..85956 and 86949..180442 (periods 1 and 2).
        "n_video_frames": 150519,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2345, 71702), 2: (72531, 150519)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 33, "frame": 19},
        "kickoff_2nd": {"minute": 48, "second": 21, "frame": 5},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3854/PFF-3854_atd.xml",
            "atd_raw": "pff/3854/PFF-3854_atd.txt",
        },
    },
    "PFF-3855": {
        # Home (H): Costa Rica. Away (A): Germany.
        # PFF FC game 3855. Tracking frameNum 3069..85803 and 88446..187395 (periods 1 and 2).
        "n_video_frames": 156319,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2562, 71574), 2: (73780, 156319)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 42, "frame": 11},
        "kickoff_2nd": {"minute": 49, "second": 11, "frame": 4},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3855/PFF-3855_atd.xml",
            "atd_raw": "pff/3855/PFF-3855_atd.txt",
        },
    },
    "PFF-3856": {
        # Home (H): Ghana. Away (A): Uruguay.
        # PFF FC game 3856. Tracking frameNum 5290..100735 and 102793..201168 (periods 1 and 2).
        "n_video_frames": 167808,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4414, 84030), 2: (85748, 167808)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 56, "frame": 13},
        "kickoff_2nd": {"minute": 57, "second": 9, "frame": 22},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3856/PFF-3856_atd.xml",
            "atd_raw": "pff/3856/PFF-3856_atd.txt",
        },
    },
    "PFF-3857": {
        # Home (H): South Korea. Away (A): Portugal.
        # PFF FC game 3857. Tracking frameNum 1355..85979 and 87538..181050 (periods 1 and 2).
        "n_video_frames": 151026,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1132, 71721), 2: (73023, 151026)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 0, "second": 45, "frame": 6},
        "kickoff_2nd": {"minute": 48, "second": 40, "frame": 22},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3857/PFF-3857_atd.xml",
            "atd_raw": "pff/3857/PFF-3857_atd.txt",
        },
    },
    "PFF-3858": {
        # Home (H): Serbia. Away (A): Switzerland.
        # PFF FC game 3858. Tracking frameNum 1214..87450 and 89657..189084 (periods 1 and 2).
        "n_video_frames": 157728,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1014, 72948), 2: (74790, 157728)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 0, "second": 40, "frame": 13},
        "kickoff_2nd": {"minute": 49, "second": 51, "frame": 14},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3858/PFF-3858_atd.xml",
            "atd_raw": "pff/3858/PFF-3858_atd.txt",
        },
    },
    "PFF-3859": {
        # Home (H): Cameroon. Away (A): Brazil.
        # PFF FC game 3859. Tracking frameNum 3038..91058 and 92032..191027 (periods 1 and 2).
        "n_video_frames": 159349,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2536, 75958), 2: (76772, 159349)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 41, "frame": 10},
        "kickoff_2nd": {"minute": 51, "second": 10, "frame": 21},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/3859/PFF-3859_atd.xml",
            "atd_raw": "pff/3859/PFF-3859_atd.txt",
        },
    },
    "PFF-10502": {
        # Home (H): Netherlands. Away (A): United States.
        # PFF FC game 10502. Tracking frameNum 5366..89315 and 90535..182553 (periods 1 and 2).
        "n_video_frames": 152280,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4478, 74504), 2: (75523, 152280)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 59, "frame": 2},
        "kickoff_2nd": {"minute": 50, "second": 20, "frame": 22},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10502/PFF-10502_atd.xml",
            "atd_raw": "pff/10502/PFF-10502_atd.txt",
        },
    },
    "PFF-10503": {
        # Home (H): Argentina. Away (A): Australia.
        # PFF FC game 10503. Tracking frameNum 4818..89424 and 90453..184642 (periods 1 and 2).
        "n_video_frames": 154022,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4021, 74595), 2: (75454, 154022)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 40, "frame": 20},
        "kickoff_2nd": {"minute": 50, "second": 18, "frame": 3},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10503/PFF-10503_atd.xml",
            "atd_raw": "pff/10503/PFF-10503_atd.txt",
        },
    },
    "PFF-10504": {
        # Home (H): France. Away (A): Poland.
        # PFF FC game 10504. Tracking frameNum 5665..90211 and 91189..188649 (periods 1 and 2).
        "n_video_frames": 157365,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4727, 75252), 2: (76068, 157365)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 9, "frame": 1},
        "kickoff_2nd": {"minute": 50, "second": 42, "frame": 17},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10504/PFF-10504_atd.xml",
            "atd_raw": "pff/10504/PFF-10504_atd.txt",
        },
    },
    "PFF-10505": {
        # Home (H): England. Away (A): Senegal.
        # PFF FC game 10505. Tracking frameNum 2832..89920 and 90979..179434 (periods 1 and 2).
        "n_video_frames": 149678,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2364, 75009), 2: (75893, 149678)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 34, "frame": 13},
        "kickoff_2nd": {"minute": 50, "second": 35, "frame": 17},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10505/PFF-10505_atd.xml",
            "atd_raw": "pff/10505/PFF-10505_atd.txt",
        },
    },
    "PFF-10506": {
        # Home (H): Japan. Away (A): Croatia.
        # PFF FC game 10506. Tracking frameNum 2063..87709 and 88970..177109 (periods 1 and 2).
        "n_video_frames": 147739,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1722, 73164), 2: (74217, 147739)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 8, "frame": 21},
        "kickoff_2nd": {"minute": 49, "second": 28, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10506/PFF-10506_atd.xml",
            "atd_raw": "pff/10506/PFF-10506_atd.txt",
        },
    },
    "PFF-10507": {
        # Home (H): Brazil. Away (A): South Korea.
        # PFF FC game 10507. Tracking frameNum 2771..92675 and 94691..182757 (periods 1 and 2).
        "n_video_frames": 152450,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2313, 77307), 2: (78990, 152450)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 32, "frame": 12},
        "kickoff_2nd": {"minute": 52, "second": 39, "frame": 14},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10507/PFF-10507_atd.xml",
            "atd_raw": "pff/10507/PFF-10507_atd.txt",
        },
    },
    "PFF-10508": {
        # Home (H): Morocco. Away (A): Spain.
        # PFF FC game 10508. Tracking frameNum 1192..83934 and 85558..175821 (periods 1 and 2).
        "n_video_frames": 146664,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (996, 70015), 2: (71371, 146664)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 0, "second": 39, "frame": 20},
        "kickoff_2nd": {"minute": 47, "second": 34, "frame": 20},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10508/PFF-10508_atd.xml",
            "atd_raw": "pff/10508/PFF-10508_atd.txt",
        },
    },
    "PFF-10509": {
        # Home (H): Portugal. Away (A): Switzerland.
        # PFF FC game 10509. Tracking frameNum 2351..90389 and 91738..179999 (periods 1 and 2).
        "n_video_frames": 150149,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (1963, 75400), 2: (76526, 150149)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 18, "frame": 12},
        "kickoff_2nd": {"minute": 51, "second": 1, "frame": 0},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10509/PFF-10509_atd.xml",
            "atd_raw": "pff/10509/PFF-10509_atd.txt",
        },
    },
    "PFF-10510": {
        # Home (H): Croatia. Away (A): Brazil.
        # PFF FC game 10510. Tracking frameNum 5404..88271 and 89346..177194 (periods 1 and 2).
        "n_video_frames": 147809,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4509, 73633), 2: (74531, 147809)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 3, "second": 0, "frame": 8},
        "kickoff_2nd": {"minute": 49, "second": 41, "frame": 5},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10510/PFF-10510_atd.xml",
            "atd_raw": "pff/10510/PFF-10510_atd.txt",
        },
    },
    "PFF-10511": {
        # Home (H): Netherlands. Away (A): Argentina.
        # PFF FC game 10511. Tracking frameNum 2718..94499 and 96501..200186 (periods 1 and 2).
        "n_video_frames": 166989,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2269, 78828), 2: (80499, 166989)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 30, "frame": 18},
        "kickoff_2nd": {"minute": 53, "second": 39, "frame": 23},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10511/PFF-10511_atd.xml",
            "atd_raw": "pff/10511/PFF-10511_atd.txt",
        },
    },
    "PFF-10513": {
        # Home (H): England. Away (A): France.
        # PFF FC game 10513. Tracking frameNum 3134..92820 and 93860..193579 (periods 1 and 2).
        "n_video_frames": 161477,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2616, 77428), 2: (78296, 161477)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 44, "frame": 15},
        "kickoff_2nd": {"minute": 52, "second": 11, "frame": 20},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10513/PFF-10513_atd.xml",
            "atd_raw": "pff/10513/PFF-10513_atd.txt",
        },
    },
    "PFF-10514": {
        # Home (H): Argentina. Away (A): Croatia.
        # PFF FC game 10514. Tracking frameNum 2761..91240 and 93704..183663 (periods 1 and 2).
        "n_video_frames": 153206,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (2305, 76110), 2: (78166, 153206)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 1, "second": 32, "frame": 4},
        "kickoff_2nd": {"minute": 52, "second": 6, "frame": 15},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10514/PFF-10514_atd.xml",
            "atd_raw": "pff/10514/PFF-10514_atd.txt",
        },
    },
    "PFF-10515": {
        # Home (H): France. Away (A): Morocco.
        # PFF FC game 10515. Tracking frameNum 4867..91666 and 92686..184362 (periods 1 and 2).
        "n_video_frames": 153789,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4061, 76465), 2: (77317, 153789)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 42, "frame": 10},
        "kickoff_2nd": {"minute": 51, "second": 32, "frame": 16},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10515/PFF-10515_atd.xml",
            "atd_raw": "pff/10515/PFF-10515_atd.txt",
        },
    },
    "PFF-10516": {
        # Home (H): Croatia. Away (A): Morocco.
        # PFF FC game 10516. Tracking frameNum 5384..89794 and 91621..183572 (periods 1 and 2).
        "n_video_frames": 153130,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (4493, 74904), 2: (76429, 153130)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 59, "frame": 17},
        "kickoff_2nd": {"minute": 50, "second": 57, "frame": 3},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10516/PFF-10516_atd.xml",
            "atd_raw": "pff/10516/PFF-10516_atd.txt",
        },
    },
    "PFF-10517": {
        # Home (H): Argentina. Away (A): France.
        # PFF FC game 10517. Tracking frameNum 4630..99119 and 100240..196639 (periods 1 and 2).
        "n_video_frames": 164030,  # last video frame of half 2 + 1
        "fps": 25,  # resampled from 29.97 Hz
        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1
        # Video frames inside the tracking periods, cut to kickoff..END of the events.
        "halves": {1: (3864, 82682), 2: (83618, 164030)},
        # First video frame of each half (the kickoff event).
        "kickoff": {"minute": 2, "second": 34, "frame": 13},
        "kickoff_2nd": {"minute": 55, "second": 44, "frame": 17},
        "pitch": (105.0, 68.0),  # stadium pitch length, width of the metadata file
        "assets": {
            "atd_meta": "pff/10517/PFF-10517_atd.xml",
            "atd_raw": "pff/10517/PFF-10517_atd.txt",
        },
    },
}

# The anchors above are expressed as (minute, second, frame) at 25 fps, which is the rate the
# burned-in clock and the tracking counter both run at. This is a property of the anchor notation,
# not of the video file, so it stays a literal here rather than being read from the game entry.
ANCHOR_FPS = 25

# The two recordings of the match, each in its own folder with its own stem. Splitting the bundle
# by source rather than by asset type is what keeps "which timeline is this file on?" answerable
# from the path alone -- the question that matters most here, because a submission is indexed to
# the tactical video and to nothing else.
#
# The source is repeated in the file name as well as the folder so that a file which gets copied
# out of its directory is still identifiable. That redundancy is deliberate.
SOURCES = {
    "tactical": {
        "stem": "BAR-ATH_tactical",
        "n_video_frames": 199319,
        # Smart Tagging is timed against the tactical video, so it belongs to this source only.
        "assets": ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"),
    },
    "broadcast": {
        "stem": "BAR-ATH_broadcast",
        "n_video_frames": 164488,
        "assets": ("video", "atd_meta", "atd_raw", "homography"),
    },
}
DEFAULT_SOURCE = "tactical"

# Asset name -> file name suffix. One table so a rename in the bundle is one edit here and nothing
# anywhere else spells a filename out.
SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}


def asset_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Every provided file for one source of a game, keyed by asset, under DATA_DIR.

    The bundle is one folder per recording:

        data/
        |-- tactical/                           the evaluation reference
        |   |-- BAR-ATH_tactical.mp4
        |   |-- BAR-ATH_tactical_atd.xml        FIFA/EPTS metadata
        |   |-- BAR-ATH_tactical_atd.txt        FIFA/EPTS raw tracking, players and ball
        |   |-- BAR-ATH_tactical_homography.csv
        |   `-- BAR-ATH_tactical_smart_tagging.xml
        `-- broadcast/                          optional, NOT aligned to tactical
            |-- BAR-ATH_broadcast.mp4
            |-- BAR-ATH_broadcast_atd.xml
            |-- BAR-ATH_broadcast_atd.txt
            `-- BAR-ATH_broadcast_homography.csv

    Defaults to the tactical source, because that is the one the submission is indexed to and the
    one every existing caller means. Broadcast is opt-in:

        asset_paths()                     # tactical
        asset_paths(source="broadcast")   # broadcast -- no smart_tagging key, it has none

    The paths are not checked for existence -- see require() for that.
    """
    root = data_dir or DATA_DIR
    entry = _source(game, source)
    folder = os.path.join(root, source)
    return {name: os.path.join(folder, entry["stem"] + SUFFIXES[name])
            for name in entry["assets"]}


def _source(game, source):
    """The SOURCES entry, or a readable error naming the ones that exist."""
    if source not in SOURCES:
        raise KeyError("no such source {!r}; have {}".format(source, sorted(SOURCES)))
    if game not in GAMES:
        raise KeyError("no such game {!r}; have {}".format(game, sorted(GAMES)))
    return SOURCES[source]


def source_frames(source=DEFAULT_SOURCE, game=GAME):
    """How many video frames that source has. Tactical is the submission length."""
    return _source(game, source)["n_video_frames"]


def video_path(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Absolute path to a source's video. May not exist; that is only a problem for tooling
    that actually decodes video."""
    return asset_paths(game, data_dir, source)["video"]


def require(names, game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Paths for the named assets, or one readable error naming every missing file.

    The match data is gated and is not in the repository, so "it is not there" is the normal
    first-run state rather than a bug. Saying which files and where they were expected is the
    difference between a two-second fix and a support message.
    """
    paths = asset_paths(game, data_dir, source)
    unknown = [name for name in names if name not in paths]
    if unknown:
        raise KeyError("source {!r} has no asset {}; it provides {}".format(
            source, unknown, sorted(paths)))

    wanted = [(name, paths[name]) for name in names]
    missing = [path for _, path in wanted if not os.path.isfile(path)]
    if missing:
        raise IOError(
            "match data not found:\n  {}\n\nThe data is gated -- see DATA.md. Unpack the bundle "
            "into\n  {}\nor point MTAH_DATA_DIR at wherever you put it."
            .format("\n  ".join(missing), data_dir or DATA_DIR))
    return [path for _, path in wanted]


def anchor_video_frame(anchor):
    """Video frame of a (minute, second, frame) anchor, at 25 fps."""
    return (anchor["minute"] * 60 * ANCHOR_FPS + anchor["second"] * ANCHOR_FPS
            + anchor["frame"])


def halves(game):
    """[(half, first_id_frame, last_id_frame)] in play order."""
    table = GAMES[game]["halves"]
    return [(half, table[half][0], table[half][1]) for half in sorted(table)]


def frame_offset(game, half, adjust=0):
    """Offset such that id_frame = video_frame + offset, for one half."""
    return GAMES[game]["offsets"][half] + adjust


def half_video_blocks(game, adjust=0):
    """[(half, first_video_frame, last_video_frame)], each half mapped by its own offset.

    For BAR-ATH: half 1 covers video 11185..90651 and half 2 covers 110808..190197, leaving
    20156 video frames of halftime between them. The blocks are far apart, so a video frame
    belongs to at most one of them and video_to_id needs no tie-breaking.
    """
    blocks = []
    for half, first_id, last_id in halves(game):
        offset = frame_offset(game, half, adjust)
        blocks.append((half, first_id - offset, last_id - offset))
    return blocks


def half_of_video_frame(game, video_frame, adjust=0):
    """Which half a video frame falls in, or None for halftime and outside the match."""
    for half, first, last in half_video_blocks(game, adjust):
        if first <= video_frame <= last:
            return half
    return None


def video_to_id(game, video_frame, adjust=0):
    """IdFrame shown at a video frame, or None where the tracking data does not reach."""
    half = half_of_video_frame(game, video_frame, adjust)
    if half is None:
        return None
    return video_frame + frame_offset(game, half, adjust)


def id_to_video(game, id_frame, adjust=0):
    """Video frame an IdFrame belongs to, or None if it is in neither half."""
    for half, first_id, last_id in halves(game):
        if first_id <= id_frame <= last_id:
            return id_frame - frame_offset(game, half, adjust)
    return None
