# Tier 1 — Segmenter Prompt Design (Round 4)

**Status:** design draft, ready for tier-1 implementation pass.
Composed during the round-4 quality work after rounds 1-3 of pause-
threshold patches failed in production.

**Constraint:** local-only (no frontier API calls in the runtime
segmenter). The prompt is what's loaded into a local Ollama text
model; the model's reasoning is what does the cross-domain selection.

**Background:** rounds 1-3 had the LLM drawing boundaries on
fragmented whisper output ("[206-234] do its work, this strange,
andragor, that has, pushed its way..."). The LLM was guessing
sentence ends in text whose ACTUAL sentence boundaries had been
shredded by ASMR cadence pauses. Result: clips that look snapped on
paper but cut mid-thought when played.

**Tier-1 fix shape:** clean up the input (sentence-reconstruct the
transcript), change the LLM's job (selection over candidate spans
instead of boundary-drawing), inject brand context, invoke
domain-specific reasoning explicitly. All local.

---

## The prompt template (Jinja-style placeholders)

```
You are selecting clips from a long-form ASMR/hypnosis video for
posting to TikTok, Instagram Reels, and YouTube Shorts.  Each clip
must be a complete, self-contained piece of work that lands as
intended on the brand's audience.

# Brand context

{{ channel_description }}

Target audience: {{ target_audience }}
Tone: {{ tone }}

{% if custom_instructions %}
Creator notes:
{{ custom_instructions }}
{% endif %}

# What you're picking from

The full transcript below has been processed for you:
- Punctuation has been restored at sentence boundaries
- Sentences have been reassembled from whisper's word-level fragments
- Each line is one sentence with its [start-end] timestamp range
- Pause-duration tags are inserted at natural pauses:
    [pause:0.4s]   = mid-sentence emphasis pause (ASMR cadence — INSIDE a thought)
    [pause:1.2s]   = thought-end / paragraph break (between thoughts)
    [pause:3.0s+]  = section break (between distinct moments)

These tags are derived from the speaker's own pause distribution.
The threshold between "emphasis" and "thought-end" is data-derived
from this specific speaker, not a global magic number.

{{ punctuated_transcript }}

# Your task

Select 5-10 clips from this transcript.  Each clip must be a
COMPLETE coherent unit — not a fragment, not a teaser, not a
mid-suggestion cut.  Each must be between {{ min_segment_duration }}
and {{ max_segment_duration }} seconds long.  Clips must NOT overlap.

# What "complete" means for this content

This is hypnosis ASMR.  Hypnotic structure has identifiable shapes
that you should recognize and respect when picking boundaries:

- **Setup → induction → suggestion → trigger → close** is the canonical
  arc.  A clip that starts mid-induction loses the setup that primes
  the listener.  A clip that ends before the suggestion lands wastes
  the induction work.

- **Fractionation cycles** (deepen → return-toward-surface → deepen
  further) are common.  Each cycle is a unit; cutting in the middle
  of a cycle leaves the listener stuck mid-transition.

- **Trigger implantation** sequences install a word/phrase as a
  conditioned response.  The implantation has a setup ("when I say X,
  you'll Y"), the implantation itself, and the test.  Cut the test
  out and the trigger doesn't anchor.

- **Nested loops** open multiple suggestion frames before closing
  them.  Nesting is fine within a clip; opening a loop without
  closing it within the clip is broken.

- **Comfort / safety language** at clip endings matters for the
  brand — listeners use this content for sleep and trance work.
  Ending on an open hook with no comfort beat is jarring.

# What "viral" means for short-form

A good ASMR/hypnosis clip for short-form needs:

- **Hook in first 2-3 seconds** — opening line that creates curiosity
  or a sensory reaction.  The first sentence carries this weight.
- **Pattern interrupt or unexpected shift** somewhere in the middle —
  a moment that makes the viewer commit to watching to the end.
- **Complete payoff** — the suggestion lands, the trigger fires, the
  trance state is invoked.  The viewer feels they got something.
- **Clean end beat** — not cut mid-breath; not on a cliffhanger that
  forces follow-up; lands on a resolution word ("sleep", "rest",
  "yes", "good", a sustained breath).

# Editing rhythm

When picking boundaries:

- **Cut on completion of a thought**, not in the middle.  The pause
  tags above tell you where thoughts end.
- **Hold on resolution beats** — don't truncate the last word of a
  suggestion or the trailing pause that lets it sink in.
- **Never cut in the middle of a suggestion delivery** — if a
  suggestion spans two sentences, both sentences are in the clip
  or neither.
- **Prefer [pause:1.2s+] tags as boundary candidates** over
  [pause:0.4s] tags.  Mid-sentence pauses are inside thoughts;
  thought-end pauses are between thoughts.

# Brand-specific exclusions

{% for line in red_lines %}
- {{ line }}
{% endfor %}

# Output format

Output ONLY valid JSON with NO additional text:

```json
{
  "clips": [
    {
      "start_time": 12.4,
      "end_time": 47.8,
      "segment_type": "induction" | "suggestion" | "trigger" | "deepener" | "comfort" | "hook",
      "hook_summary": "one sentence describing what makes this clip land",
      "rationale": "why these specific boundaries — what hypnosis structure this preserves, what would break if cut differently"
    }
  ]
}
```

The rationale field is mandatory.  If you cannot articulate why a
specific boundary is right, the boundary is probably wrong.
```

---

## Why this prompt design beats rounds 1-3

| Failure mode (rounds 1-3) | Tier-1 prompt addresses how |
|---|---|
| LLM saw fragmented "[206-234] do its work, this strange, andragor, that has, pushed..." and had to guess sentence ends | Input is now sentence-reconstructed prose with pause tags. The LLM doesn't have to reverse-engineer where sentences end. |
| LLM was asked for "viral segments" — vague target | Prompt explicitly defines what "complete" and "viral" mean for THIS content type, with hypnosis-specific structural concepts. |
| Pause-snap chased temporal pauses | Pause tags distinguish mid-sentence (don't cut) from thought-end (good cut candidate). LLM uses semantic categorization, not heuristics. |
| No brand context — generic prompt | Brand description, target audience, tone, custom instructions, red-lines all injected. The LLM judges "good clip for THIS channel." |
| LLM did boundary-drawing AND selection in one pass | Selection only — the candidate spans are derived deterministically from sentence-tokenize + pause-mode threshold; LLM picks 5-10 of them. |
| Magic-number thresholds (0.5s, 1.5s) | Pause-mode threshold is derived from this speaker's own pause distribution (bimodal histogram), not assumed. |
| No rationale required → impossible to debug bad boundaries | Rationale field is mandatory. When boundaries are bad, the rationale tells composer WHY the model thought they were right. |

---

## Knobs / variables for the implementation

Most of these come from `PipelineConfig` and `ContentProfile`:

| Variable | Source | Notes |
|---|---|---|
| `channel_description` | `content_profile.channel_description` | injected verbatim |
| `target_audience` | `content_profile.target_audience` | injected verbatim |
| `tone` | `content_profile.tone` | injected verbatim |
| `custom_instructions` | `content_profile.custom_instructions` | injected if non-empty |
| `min_segment_duration` | `cfg.min_segment_duration` | composer-tunable |
| `max_segment_duration` | `cfg.max_segment_duration` | composer-tunable |
| `red_lines` | new field — read from clip-mill corpus or config? | TODO: decide |
| `punctuated_transcript` | computed deterministically from whisper word-level + punctuation-restoration + sent_tokenize | the cleaned input |

For ASMR specifically, recommend `min_segment_duration: 25` and
`max_segment_duration: 90` so the LLM has room to keep complete
inductions intact.  For non-ASMR content, the existing 15-45s range
is fine.

---

## red_lines source — design call

The prompt's brand-exclusion section needs a list of strings that
the clip should NOT contain.  Two source options:

A. **A new `ContentProfile.red_lines: list[str]`** field.  Composer
   sets via CLI / INI.  Keeps yacg self-contained.

B. **Read from a file path** specified in `content_profile`.  Useful
   if red lines are maintained outside yacg (e.g. clip-mill's brand
   corpus at `{project_root}/.context/red-lines.md`).

Path B is more flexible; path A is simpler.  Recommend A initially
with the understanding that composer can pass red_lines as a
multi-line string from CLI.  If it becomes painful, layer B later.

---

## Implementation note for transcript_segmenter.py rewrite

When tier-0 clears, the actual rewrite path:

1. Add helper functions:
   - `_restore_punctuation(words)` — wraps deepmultilingualpunctuation
   - `_reconstruct_sentences(words, punctuated_text)` — maps punctuated
     text back to word-level timestamps, returns
     `[(start, end, sentence_text), ...]`
   - `_pause_histogram_threshold(words)` — returns the bimodal-split
     pause-duration that separates mid-sentence from thought-end for
     THIS speaker
   - `_format_punctuated_transcript_with_pauses(sentences, words, threshold)`
     — produces the `[start-end] sentence text [pause:Xs]` formatted
     input

2. Replace `_create_segmentation_prompt` with the template above
   (Jinja or simple `.format`).

3. Replace `segment_by_content` to:
   - Call the new helpers
   - Send the new prompt to Ollama
   - Parse JSON output (with rationale captured for logging)

4. `refine_boundaries` — keep the existing snap logic as a SAFETY
   NET, but the new boundaries should be already-sentence-aligned
   from the LLM's selection. Snap becomes a no-op for clean output.

The CrisperWhisper swap is OPTIONAL for the first iteration —
faster-whisper gives word-level timestamps that work with the
restoration + reconstruction pipeline. If the bimodal histogram
quality is poor on faster-whisper output, swap to CrisperWhisper.

---

## Validation against tier 4

The whole point of this design is that the OUTPUT is judgeable
quantitatively:

1. Run `test_segmenter_iso.py --ground-truth <gt-file>`
2. Recall = % of composer-marked clips that the LLM also picked
3. Precision = % of LLM-picked clips near a composer-marked one
4. Mean boundary error = how far off the LLM is on average

Acceptance threshold for tier 1 to ship to clip-mill:
- Recall ≥ 60% on the ai-takeover source against composer's GT
- Mean boundary error ≤ 2.0s

If we can't hit that, the design needs another pass before going to
production.
