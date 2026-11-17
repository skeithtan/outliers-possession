// The external tests of the submitted engine, the learned models. Source: Section 3 of the
// technical document. The figures of the technical document show the same numbers.
//
// gain: the engine's spellF1 minus the rule engine's, averaged over the test's matches.
// above: matches on which the engine scores above the rule engine. lo5: the lower 5% bound of
// the mean gain (paired bootstrap), where it was computed. earlier: the test ran on the earlier
// pair of models (same design and settings, an earlier attacking-direction rule).

export const LEARNED_EXTERNAL = [
  { id: "metrica_left_out", group: "unseen provider", label: "Metrica",
    note: "3 sample games", n: 3, gain: 0.154, above: 3, lo5: null, earlier: false },
  { id: "dfl_left_out", group: "unseen provider", label: "DFL",
    note: "5 training matches", n: 5, gain: 0.166, above: 5, lo5: null, earlier: false },
  { id: "skillcorner_left_out_phases", group: "unseen provider", label: "SkillCorner",
    note: "20 matches, phases of play", n: 20, gain: 0.101, above: 20, lo5: null, earlier: false },
  { id: "skillcorner_left_out_fine", group: "unseen provider", label: "SkillCorner",
    note: "20 matches, possession field", n: 20, gain: 0.004, above: 12, lo5: null,
    earlier: false },
  { id: "pff_benchmark_2s", group: "PFF FC World Cup 2022", label: "PFF design matches",
    note: "43 matches, 2-second labels", n: 43, gain: 0.163, above: 43, lo5: 0.156,
    earlier: false, engine: 0.608, rule: 0.445 },
  { id: "pff_sealed_2s", group: "PFF FC World Cup 2022", label: "held-out PFF matches",
    note: "15 matches, run once, earlier pair", n: 15, gain: 0.170, above: 15, lo5: 0.159,
    earlier: true },
];
