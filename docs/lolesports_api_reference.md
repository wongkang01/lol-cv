# Lolesports API Reference

A practical, hand-probed catalogue of the unofficial lolesports.com REST endpoints
that power the project's metadata, VOD discovery, and (potentially) per-frame
analytics. Probed against the First Stand 2026 BFX vs BLG game 1 (game id
`115570888977308364`, match id `115570888977308363`) on 2026-04-07.

This is the reference the project should consult before adding any new analysis
that needs metadata, gold trajectories, item builds, or objective timing.

---

## Auth and hosts

Both APIs share the same hard-coded public key that the lolesports.com web client
ships with. No OAuth, no headers besides the key.

```
x-api-key: 0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z
```

| Host | Purpose | Versioning |
|------|---------|------------|
| `https://esports-api.lolesports.com/persisted/gw/` | Catalogue (leagues, schedule, teams, VODs, standings) | "persisted/gw" — no version |
| `https://feed.lolesports.com/livestats/v1/` | Per-frame in-game telemetry | `v1` (probed `v2` -> 403) |

Be polite: ~1 s between requests, never loop over hundreds of games. The shared
key is rate-limited globally for the whole community.

---

## Persisted GW endpoints (esports-api.lolesports.com)

All persisted endpoints accept `hl=en-US` (locale). Most ID-typed parameters are
big integer strings (Riot's snowflake IDs).

Top-level shape is always `{"data": {...}}`. Errors come back as
`{"errors": [{...}]}` with HTTP 200 or 4xx depending on the kind of failure.

### `getLeagues` — region/league listing

- **URL**: `GET /persisted/gw/getLeagues`
- **Params**: `hl` (required)
- **Status**: 200
- **Top keys**: `data.leagues[]` (41 entries as of 2026-04-07)
- **Per-league fields**: `id`, `slug`, `name`, `region`, `image`, `priority`,
  `displayPriority.{position,status}`
- **Example**:
  ```json
  {
    "id": "98767975604431411", "slug": "worlds", "name": "Worlds",
    "region": "INTERNATIONAL", "priority": 1
  }
  ```
- **Notes**: Persistent ground truth for all `leagueId` lookups elsewhere.
  Useful league IDs:
  - LCK `98767991310872058`
  - LEC `98767991302996019`
  - LCS `98767991299243165`
  - MSI `98767991325878492`
  - Worlds `98767975604431411`
  - First Stand `113464388705111224`
- Not time-series.

### `getTournamentsForLeague` — list tournaments inside a league

- **URL**: `GET /persisted/gw/getTournamentsForLeague`
- **Params**: `hl`, `leagueId` (required)
- **Status**: 200
- **Top keys**: `data.leagues[0].tournaments[]`
- **Tournament fields**: `id`, `slug`, `startDate`, `endDate`
- **Example** (First Stand 2026):
  ```json
  {"id":"115570858980956868","slug":"first_stand_2026",
   "startDate":"2026-03-16","endDate":"2026-03-23"}
  ```
- **Notes**: Returns the full historical tournament list per league (LCK has
  37 tournaments going back many splits). Use to translate between season + split
  and the tournament ID needed for `getStandings`.
- Not time-series.

### `getStandings` — bracket / group state for a tournament

- **URL**: `GET /persisted/gw/getStandings`
- **Params**: `hl`, `tournamentId` (required)
- **Status**: 200 (returns empty `standings: []` if the tournamentId is wrong)
- **Top keys**: `data.standings[].stages[].sections[].matches[]`
  - `stages[]`: each stage (`Group Stage`, `Knockouts`, etc.)
  - `sections[]`: groupings inside a stage (`Group A`, etc.)
  - `matches[]`: per-match `id`, `state`, `previousMatchIds[]`, `flags[]`,
    `teams[].{id,slug,name,code,image,result.{outcome,gameWins}}`
- **Notes**: Best endpoint for **bracket structure**, including
  `previousMatchIds` for tracking who advanced from where. Includes team slugs
  (handy for `getTeams` lookup).
- Not time-series.

### `getStandingsV3` — alternative standings with season/split metadata

- **URL**: `GET /persisted/gw/getStandingsV3`
- **Params**: `hl`, `tournamentId` (required)
- **Status**: 200
- **Top keys**: `data.standings[].{id,name,slug,scores,split,season}`
- **Notes**: Surfaces the parent **season** (`lolesports_2026`) and **split**
  metadata (`Split 1`, `First Stand`, `Split 2` …) including `startTime` /
  `endTime`. Use this when you want to know what split a tournament belongs to.

### `getSchedule` — upcoming + recent matches for a league

- **URL**: `GET /persisted/gw/getSchedule`
- **Params**: `hl`, `leagueId` (required), optional `pageToken` for paging
- **Status**: 200
- **Top keys**: `data.schedule.{pages,events[]}`
  - `pages.{older,newer}`: pageToken cursors (null if no more pages)
  - `events[]`: per-match `startTime`, `state` (`completed`/`unstarted`/
    `inProgress`), `type` (`match`/`show`), `blockName` (`Round 1`,
    `Knockouts`, …), `league.{name,slug}`, `match.id`, `match.flags[]`,
    `match.teams[]` (with `result.{outcome,gameWins}` and `record.{wins,losses}`)
- **Critical caveat**: `events[].match` does **not** contain game IDs. To get
  game IDs you must call `getEventDetails?id=<match.id>`.
- Not strictly time-series, but the schedule endpoint paginates by time.

### `getEventDetails` — match-level lookup with game IDs and VODs

- **URL**: `GET /persisted/gw/getEventDetails`
- **Params**: `hl`, `id` (required — must be a **match id**, NOT a game id;
  passing a game id returns `{"data":{"event":null}}`)
- **Status**: 200
- **Top keys**:
  - `data.event.{id, type, tournament.id, league.{id,slug,name,image},
    match.{strategy,teams,games[]}, streams[]}`
  - `match.games[]`: each game has `number`, `id` (the game id used by
    livestats), `state` (`completed`/`unstarted`/`inProgress`), `teams[].{id,side}`
    (blue/red mapping!), and `vods[]`
  - Each VOD entry: `id`, `parameter` (YouTube video id or Twitch broadcast id),
    `locale`, `mediaLocale.{englishName,translatedName}`, `provider`
    (`youtube`/`twitch`/`afreecatv`/…), `offset`, `firstFrameTime` (ISO8601),
    `startMillis`, `endMillis` (offsets into the VOD where the game starts/ends)
- **Why this is important**: `vods[].firstFrameTime` is the **wall-clock
  ISO8601 timestamp at which in-game time t=0 occurred**. Combined with
  `startingTime` on `livestats/window`, this is what lets you fetch frames at
  exact in-game offsets without OCR.
- **Already used by the project** (`vod_discovery.py`, `match_metadata.py`).
- Not time-series, but the source of truth for all timing math.

### `getGames` — single game lookup

- **URL**: `GET /persisted/gw/getGames`
- **Params**: `hl`, `id` (required, accepts a game id; comma-separated for batch?
  not verified)
- **Status**: 200
- **Top keys**: `data.games[].{id,state,number,vods[]}`
- **Notes**: Same VOD list as `getEventDetails` but stripped of all match-level
  context (no team mapping, no league info). Useful only if you already have a
  game id and just need its VOD parameters quickly. **Lighter than
  `getEventDetails`** for VOD-only lookups.

### `getLive` — currently live matches

- **URL**: `GET /persisted/gw/getLive`
- **Params**: `hl` (required)
- **Status**: 200
- **Top keys**: `data.schedule.events[]`
  - Each event: `id`, `startTime`, `state: "inProgress"`, `type: "match"`,
    `blockName`, `league`, `tournament`, `match`, `streams[]`
- **Notes**: Returns 0–N events. Used for live tooling; not relevant to the
  CV-only finished-game pipeline.

### `getCompletedEvents` — historical results across all leagues

- **URL**: `GET /persisted/gw/getCompletedEvents`
- **Params**: `hl`, `leagueId` (required, BUT see caveat below)
- **Status**: 200
- **Top keys**: `data.schedule.events[]` (300 events on first page)
- **Per-event fields**: `startTime`, `blockName`, `league.{name}`, `match.{id,
  type, teams[].{name,code,image,result.gameWins}, strategy}`, `games[]` with
  per-game `id` and `vods[].parameter`
- **Major caveat**: Even though we passed `leagueId=113464388705111224`
  (First Stand), the response was **dominated by Hellenic Legends League**
  matches. Either the param is being ignored or it's only used as a sort hint.
  **Filter client-side by `league.name`** to be safe.
- Effectively a global firehose of finished events with their game IDs and
  parameter strings — useful for backfilling historical datasets.

### `getVods` — global VOD listing

- **URL**: `GET /persisted/gw/getVods`
- **Params**: `hl`, `id` (the `id` param appears to be **ignored** — passing
  a specific game id still returns the same global list)
- **Status**: 200
- **Top keys**: `data.schedule.events[]`
- **Notes**: Same shape as `getCompletedEvents` but each game's `vods[]`
  contains `startMillis` / `endMillis` offsets. Used by
  `vod_discovery.VodDiscovery._fetch_vods()` already, but the project
  *implicitly* relies on the global firehose, then filters by league name in
  Python. The hardcoded `id` parameter in the existing code is decorative.

### `getTeams` — team metadata + roster

- **URL**: `GET /persisted/gw/getTeams`
- **Params**: `hl`, `id` (a team **slug** like `g2-esports`,
  `bilibili-gaming`; numeric IDs do NOT work; omitting `id` returns all 1514
  teams Riot has ever tracked)
- **Status**: 200
- **Top keys**: `data.teams[].{id, slug, name, code, image, alternativeImage,
  backgroundImage, status, homeLeague.{name,region}, players[]}`
- **Per-player fields**: `id`, `summonerName`, `firstName`, `lastName`,
  `image`, `role` (`top`/`jungle`/`mid`/`bottom`/`support`/`none`)
- **Example** (G2): includes Caps (mid), BrokenBlade (top), etc.
- **Notes**: New endpoint for the project. Useful for player photos, full
  names, and the canonical role mapping (independent of in-game role
  assignments).

### Endpoints that 400 (don't exist)

The following speculative endpoints all returned `400 BAD_REQUEST` with
`errors:[…]`:

- `getHighlights` — does not exist
- `getDraft` — does not exist (no public draft/pick-ban API)
- `getEventVideos` — does not exist
- `getPlayers` — does not exist as a public endpoint (use `getTeams` to
  enumerate players via rosters)
- `getEventByMatchId` — does not exist (use `getEventDetails?id=<matchId>`)

None returned 401/403 — the API is uniformly open with the public key.

---

## Livestats v1 endpoints (feed.lolesports.com)

Both endpoints are **per-frame time-series**. The response is always a list of
frames, plus (for `window`) some metadata.

### `window/{gameId}` — per-frame team & player snapshots

- **URL**: `GET /livestats/v1/window/{gameId}`
- **Params**:
  - `startingTime` (optional, ISO8601 with `Z` suffix and `.000` ms): MUST be
    aligned to a 10-second boundary, otherwise returns
    `400 BAD_QUERY_PARAMETER` ("startingTime must be evenly divisible by 10
    seconds"). Without this param, returns the very first ~10 s of frames
    starting at game start.
- **Status**: 200 for finished games, 200 for live games.
- **Top keys**: `esportsGameId`, `esportsMatchId`, `gameMetadata`, `frames[]`
- **`gameMetadata`**:
  - `patchVersion` (e.g. `"16.5.751.985"` — the **patch the game was played
    on**, including build number)
  - `blueTeamMetadata.{esportsTeamId, participantMetadata[]}`
  - `redTeamMetadata.{esportsTeamId, participantMetadata[]}`
  - Each `participantMetadata`: `participantId` (1–10), `esportsPlayerId`,
    `summonerName` (e.g. `"BFX Clear"`), `championId` (Riot internal name like
    `"Ambessa"`), `role` (`top`/`jungle`/`mid`/`bottom`/`support`)
- **`frames[]`**: each frame has
  - `rfc460Timestamp` (ISO8601 with millis)
  - `gameState` (`"in_game"`, `"paused"`, `"finished"`)
  - `blueTeam.{totalGold, inhibitors, towers, barons, totalKills, dragons[],
    participants[]}` (and the same for `redTeam`)
  - `dragons[]`: list of dragon **types** taken so far (e.g.
    `["chemtech","chemtech","ocean","infernal"]`). No timestamps inside —
    you have to diff between frames to know when each was taken.
  - `participants[]`: per-player `participantId, totalGold, level, kills,
    deaths, assists, creepScore, currentHealth, maxHealth`
- **Frame cadence inside one response** (this matters!):
  - Each call returns a ~10 s slice with **frames every ~200–500 ms** (so 30–55
    frames per call). They are *not* uniformly spaced — they follow the
    server's internal tick rate.
  - First frame's timestamp is essentially the `startingTime` you passed (within
    a few hundred ms).
  - Last frame is `startingTime + ~9.5 s`.
- **Walking a full game**: the **canonical step is 10 seconds**. Issue
  successive calls with `startingTime = firstFrameTime + N*10s` for
  `N = 0, 1, 2, …, ceil(duration/10)`. Each call returns the dense slice for
  that 10 s window. The number of HTTP calls per game ≈ `duration_seconds/10`,
  so a 30-minute game ≈ 180 calls. Be polite about this.
- **Works for finished games**: yes — confirmed for First Stand 2026 BFX vs BLG
  g1, finished 3 weeks before this probe.

### `details/{gameId}` — deep per-participant snapshots

- **URL**: `GET /livestats/v1/details/{gameId}`
- **Params**:
  - `startingTime` (same 10-second-aligned ISO8601 rule as `window`)
  - `participantIds` — underscore-separated participant IDs, e.g.
    `1_2_3_4_5` to filter to blue side, or `6_7_8_9_10` for red. Mixing both
    sides also works.
- **Status**: 200
- **Top keys**: `frames[]`
- **Each frame**: `rfc460Timestamp`, `participants[]`
- **Each participant entry**:
  - Identity: `participantId`
  - Combat: `level, kills, deaths, assists, totalGoldEarned, creepScore,
    killParticipation, championDamageShare, wardsPlaced, wardsDestroyed`
  - Stats: `attackDamage, abilityPower, criticalChance, attackSpeed,
    lifeSteal, armor, magicResistance, tenacity`
  - **`items[]`**: list of currently-held item IDs (Riot item IDs, e.g.
    `[1054, 3340, 6692, 1028, 3111]`). This is the inventory snapshot, not a
    purchase log.
  - **`perkMetadata`**: `{styleId, subStyleId, perks[]}` — primary and
    secondary rune trees plus the 8 selected runes.
  - **`abilities`**: list of recently used / leveled abilities (was empty in
    early-game frames).
- **Same cadence as `window`** (~250 ms within each 10 s slice).
- **Works for finished games**: yes.

### `livestats/v2` — does not exist

`GET /livestats/v2/window/{gameId}` returns `403`. There is no v2 API
publicly accessible with this key.

---

## Frame cadence and time-series behaviour

| Property | Value |
|---|---|
| Server tick rate | 200–500 ms (variable, ~30–55 frames per 10 s slice) |
| Required `startingTime` alignment | exactly multiples of 10 seconds UTC |
| Frames per call | one ~10 second window |
| Duration of one call | ~9.5–10.0 s of dense frames |
| To walk a full 30-minute game | ~180 calls per endpoint |
| Re-requesting the same `startingTime` | same frames (deterministic) |
| Available for finished games | yes (both `window` and `details`) |
| Available for live games | yes (`getLive` to find game id) |

**How to convert in-game time → API timestamp**:
1. Call `getEventDetails?id=<matchId>`.
2. Pick the relevant locale's VOD entry; read `firstFrameTime`
   (ISO8601 wall clock at in-game t=0). The `en-US` entry is most reliable.
3. For target in-game time `t` seconds (rounded to nearest 10 s),
   `startingTime = firstFrameTime + t` (in ISO8601, millisecond precision,
   with `.000Z` suffix and aligned to 10 s).
4. Issue `window/{gameId}?startingTime=<that>`.
5. The first frame in the response is your t-aligned snapshot; the rest of
   the slice gives you ~10 s of higher-resolution data if you want it.

---

## Answers to project-relevant questions

### 1. Per-frame gold differential time series for a finished game without OCR?

**Yes — fully supported.** Walk `livestats/v1/window/{gameId}` with
`startingTime` aligned to 10 s steps from `firstFrameTime` (game t=0) to
`firstFrameTime + duration_seconds`. Each call returns ~30+ frames in a
10 s window, each with `blueTeam.totalGold` and `redTeam.totalGold` (and
per-player `participants[i].totalGold`). Confirmed working on BFX vs BLG g1
(finished First Stand 2026), 3 weeks after the game ended.

Cost: ~1 HTTP call per 10 in-game seconds = ~180 calls for a 30 min game.
Polite throttling means ~3 minutes per game. **This can completely replace
the broken HUD-OCR gold-target pipeline** for any analysis the project wants
to layer on top of CV minimap features.

### 2. Per-objective timing (dragons, herald, baron, towers, inhibs)?

**Partial — by frame diff only, not as explicit events.** There is no
endpoint that returns "dragon taken at gameTime=14:32". The exact
in-game timestamp must be reconstructed by walking the `window` frames at
10 s steps and watching for changes in:

- `blueTeam.dragons[]` / `redTeam.dragons[]` — the list grows by one element
  (with the dragon type as a string) when one is taken
- `blueTeam.barons` / `redTeam.barons` — integer count increments
- `blueTeam.towers` / `redTeam.towers` — integer count increments
- `blueTeam.inhibitors` / `redTeam.inhibitors` — integer count increments

Heralds are **not separately tracked** in the `window` schema (their
spawn-grub successor in Patch 14.10+ doesn't appear here either). If
herald/grub timing is needed, it has to come from CV.

Resolution is **10 seconds** (the polling step). To pinpoint the exact second
of an objective, you'd need to grab the dense ~250 ms frames around the 10 s
slice in which the count first changes — that gives you ~250 ms resolution at
no extra HTTP cost.

This is good enough for "team grouping at t-30s before objective contest"
features: round-trip a single 10 s slice at 30 s before each detected
objective increment.

### 3. Champion item builds and rune choices

**Yes, in `details`.** Each `participants[i]` frame has:

- `items[]` — list of item IDs currently in the player's inventory
  (snapshot, not log).
- `perkMetadata.{styleId, subStyleId, perks[]}` — chosen primary/secondary
  rune trees and the 8 selected runes (Riot perk IDs).

To extract a **purchase timeline**, walk the dense 10 s slice and diff
`items[]` between consecutive frames; an item appearing in `items[]` for
the first time is a purchase. (Selling is the inverse but rare.) No native
"purchase log" endpoint exists.

### 4. Player roles — reliability

**100% populated in our existing data.** Cross-checked
`data/champion_picks.json` (45 First Stand 2026 games × 10 players = 450
participant entries): every entry has `role ∈ {top, jungle, mid, bottom,
support}`. No missing/`unknown`/`none` values were observed in this dataset.
Roles come from `livestats/v1/window` →
`gameMetadata.{blueTeamMetadata,redTeamMetadata}.participantMetadata[i].role`
(NOT from `getEventDetails`, which has no per-game player info).

Note that `getTeams?id=<slug>` returns players with `role` too, but those are
the **team's declared roles**, not necessarily the in-game role for a
specific game. For per-game role, always use the livestats source.

### 5. Patch / version

**Yes** — `gameMetadata.patchVersion` in the `window` endpoint response.
Example: `"16.5.751.985"` for First Stand 2026 g1 (Patch 16.5, build
751.985). This is the only place the patch surfaces — it's not in
`getEventDetails`, `getStandings`, or anywhere in the persisted/gw API.
You always need at least one `window` call per game to know the patch.

### 6. VOD URLs

**Yes** — both `getEventDetails` and `getGames` (and the noisy `getVods` /
`getCompletedEvents` global firehoses) return `vods[]` per game with:

- `parameter` — for YouTube, this is the video id (append to
  `https://www.youtube.com/watch?v=`); for Twitch, it's the broadcast id
  (append to `https://www.twitch.tv/videos/`); afreecatv similar.
- `provider` — `youtube`, `twitch`, `afreecatv`, plus a few more.
- `locale` / `mediaLocale.{englishName, translatedName}` — pick `en-US`
  for the English broadcast.
- `firstFrameTime` (ISO8601), `startMillis`, `endMillis` — offsets into
  the source VOD where the actual game starts and ends. The project's
  `match_metadata.json` already stores `start_time`/`end_time` in seconds
  derived from these.

There is no separate "official VOD URL" endpoint — VODs are always nested
under their game inside one of these match/game catalogue endpoints.

---

## What this unlocks for the project

1. **Replace the broken HUD-OCR gold pipeline.** The livestats `window`
   endpoint already provides `blueTeam.totalGold`, `redTeam.totalGold`, and
   per-player `participants[].totalGold` at every game-time offset. Build a
   fetcher that walks each finished game in 10 s steps and writes a
   `gold_trajectory.parquet` per game. This is ground-truth, not estimates.

2. **Per-objective targets without CV.** Same walker can write a
   `dragon_taken_at`, `baron_taken_at`, `tower_count_over_time`,
   `inhib_taken_at` table. Combined with the spatial CV features, this enables
   the "team grouping at t-30s before objective contest" feature the project
   plan calls for, without any minimap-based herald/dragon detector.

3. **Item / rune build features.** Pull `details` once per 10 s and you have
   per-player rune choices (categorical features for "rune trees per matchup")
   and item-build trajectories (one-hot or item-class features over time).
   Useful for downstream "build path → win prob" models that the CV pipeline
   alone can't supply.

4. **Patch-stratified analysis.** `gameMetadata.patchVersion` lets you
   stratify any model by patch — important because First Stand 2026 spanned
   only one patch but tournament cross-comparisons (MSI, Worlds, future First
   Stand 2027) will not.

5. **Cleaner roster lookup.** `getTeams?id=<slug>` gives photos, full names,
   and team-declared roles for all players. Useful for the report's
   "who played" table without scraping Liquipedia.

6. **Bracket / standings reconstruction.** `getStandings` (or `V3`) gives
   `previousMatchIds`, allowing automatic bracket reconstruction for any
   tournament. The project plan currently uses hand-typed match IDs.

7. **Cross-tournament backfill.** With `getTournamentsForLeague` +
   `getStandings` + `getEventDetails`, the project can be extended to MSI 2024,
   Worlds 2024, etc., without manual CSV editing.

---

## Rate limit observations

- All ~20 probe requests in this session returned 200 (or expected 4xx). No
  429s seen.
- ~1.2 s sleep between requests was used. Both hosts felt instant — typical
  TTFB ~150–300 ms.
- **Suggested limits** if scaling up:
  - Persisted API: ≤1 req/s sustained for catalogue calls.
  - Livestats: ≤1 req/s per game; for backfilling many games, parallelize at
    most 2-3 games at once and stop immediately on any 429.
- Both hosts share the same key, so the rate limit is presumably per-key,
  not per-host.

---

## Caveats

1. **`leagueId` is not always honoured.** `getCompletedEvents` and `getVods`
   return data from leagues you didn't ask for (saw Hellenic Legends League
   matches when filtering to First Stand). Always filter by `league.name` or
   `league.slug` client-side.
2. **`getEventDetails` requires a match id, not a game id.** Passing a game id
   silently returns `{"data":{"event":null}}` (no error code). The project's
   `match_metadata.json` schema correctly distinguishes the two; future code
   should keep that distinction.
3. **`startingTime` 10 s alignment is enforced server-side** with a clean
   `400 BAD_QUERY_PARAMETER`. Don't try to use sub-10 s windows.
4. **Frames inside one slice are not uniformly spaced.** Don't index by
   position — index by `rfc460Timestamp`.
5. **`dragons[]` is a list of *types* not events.** You only know "a chemtech
   was taken" — not "by whom" (the team is implied by which side's list it
   appears in) or "when" (you have to diff frames). For the dragon kill *time*
   you need the 10 s polling delta.
6. **`details` doesn't carry team membership.** It only has `participantId`
   1–10. Map back to teams via the `gameMetadata` from a single `window`
   call (1–5 = blue, 6–10 = red, by Riot convention and confirmed in this
   data).
7. **The shared API key is community-maintained, not officially blessed.**
   Riot can revoke it at any time. If they do, the project loses VOD discovery
   *and* any livestats-based analysis at once. Plan a manual-fallback path
   for at least the static metadata (it's already partly stored in
   `data/match_metadata.json`).
8. **`getHighlights`, `getDraft`, `getEventVideos`, `getPlayers`,
   `getEventByMatchId`** all 400 — they don't exist. Don't waste cycles
   re-probing them.
9. **No pick-and-ban / draft endpoint.** If draft order analysis is needed,
   it has to come from another source (Liquipedia, gol.gg) — the lolesports
   API does not expose it.
10. **No herald / void grub timing.** The objective tracking only covers
    dragons, barons, towers, inhibitors. Anything else needs CV or a third
    party source.
