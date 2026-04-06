# Spectator HUD Layout Reference — 2026 First Stand Tournament

**Source VODs**: 1080p YouTube broadcasts of the 2026 First Stand tournament
**Reference match**: G2 vs GEN semifinals, Game 1 (2026-03-21)
**Resolution**: 1920x1080 (all pixel coordinates assume this resolution)
**Patch**: 26.5 (visible in draft screen)

---

## 1. Screen Phases

A tournament broadcast VOD progresses through several distinct visual phases. Programmatic phase detection is essential because the HUD layout, and therefore the valid cropping regions, changes drastically between phases.

### 1.1 Pre-Game (Interviews, Ads, Teasers)

**Description**: Analyst desk, player interviews, sponsor bumpers, and hype reels before a game begins.

**Visual characteristics**:
- No standardised layout; highly variable content (talking heads, logos, montages)
- Full-screen camera footage of real people / studio sets — no game UI elements
- Sponsor logos and broadcast graphics fill the entire screen
- No minimap, no scoreboard bar, no champion icons in HUD positions

**Programmatic detection**:
- Absence of the game HUD scoreboard bar in the top ~80px
- Absence of the minimap in the bottom-right corner (1650-1920, 790-1080)
- Can check for dark/uniform bands in HUD-expected regions; pre-game footage will not have them
- High visual variance between frames (camera cuts), low structural consistency

### 1.2 Champion Select (Draft Phase)

**Description**: Both teams ban and pick champions. The broadcast shows a custom overlay with the stage/players in the upper portion and the draft UI in the lower portion.

**Visual characteristics (observed in frame_000104.png and frame_000350.png)**:
- **Upper ~60% of screen**: Live camera feed of the stage, players at their PCs, coaches standing behind. Varies between wide shots and close-ups of individual players.
- **Lower ~40% of screen (y: ~640-1080)**: Draft UI overlay with a dark background containing:
  - **Center**: Tournament/patch branding. Team logos side by side (G2 logo left, GEN logo right) with "PATCH 26.5" below. A "VS" graphic may appear between them.
  - **Left side (x: 0-480)**: Five player name labels stacked vertically (e.g., BROKENBLADE, SKEWMOND, CAPS, HANS SAMA, LABROV) with champion portrait slots that fill in as picks are made. Names are in ALL CAPS, white text, rotated 90 degrees (vertical orientation).
  - **Right side (x: 1440-1920)**: Five player name labels for the opposing team (e.g., KIIN, CANYON, CHOVY, RULER, DURO) with their champion portrait slots.
  - Champion portraits appear as rectangular cards (~120x180px each) arranged in a horizontal row at the very bottom of the screen.
- **Top-left corner**: Tournament logo badge (orange "26" shield icon with "GEN 1ST PICK / G2 BLUE SIDE" text)
- **Coach names**: Displayed in white text near the team areas (e.g., "DYLAN FALCO & RODRIGO" for G2, "RYU & LYN" for GEN)

**Programmatic detection**:
- Dark banner in the bottom ~40% of the screen with high contrast champion portrait slots
- Team logos (G2, GEN) visible at roughly screen center bottom (x: 860-1060, y: 700-780)
- Player name labels along bottom edge — vertical text in specific positions
- No minimap, no game timer, no kill scoreboard
- Check for high average pixel brightness in the upper portion (stage lighting) combined with low brightness / dark overlay in the lower portion
- Template match on team logos or the "PATCH XX.X" text at center-bottom

### 1.3 Loading Screen

**Description**: The transition screen after draft completes but before the game starts. Shows a splash art or loading progress indicator.

**Visual characteristics**: In the observed VOD, the draft phase transitions directly to gameplay without a visible standalone loading screen frame (the broadcast typically cuts to a replay of a previous game or a sponsor bumper during the actual loading phase). Loading screens in solo-queue show champion splash art with summoner names, but in tournament broadcasts, the production team overlays custom content.

**Programmatic detection**:
- Rarely visible in professional broadcasts (production cuts around it)
- If present: look for the characteristic loading bar or champion splash art grid
- Absence of both draft UI and gameplay HUD
- Can be detected as a "neither draft nor gameplay" gap between the two phases

### 1.4 In-Game (Spectator Mode) -- PRIMARY PHASE

**Description**: The main gameplay view. The spectator camera shows the game map with a full HUD overlay. This is the phase where all data extraction occurs.

**Visual characteristics (observed in frames 000500, 001200, 002000)**:
- **Top center**: Scoreboard bar spanning ~1920px wide, ~80px tall, containing team logos, kill counts, gold totals, turret counts, and game timer
- **Bottom center**: Player scorecards panel (~400px wide per team) showing champion portraits, player names, KDA, CS, items
- **Bottom-right**: Minimap (~270x290px)
- **Bottom-left and bottom-right corners**: Live player camera feeds (webcam overlays)
- **Bottom-left corner area**: Rotating sponsor logo (Coinbase, AWS, HyperX observed)
- **Center of screen**: Gameplay area (the Summoner's Rift map rendered by the spectator camera)
- **Top-left corner**: Dragon/Baron soul tracker icons
- **Right edge**: Event feed (kill/objective notifications) appears briefly

**Programmatic detection**:
- Presence of the scoreboard bar: check top ~80px for the characteristic dark semi-transparent background with bright text/numbers
- Presence of the minimap: bottom-right corner contains a distinctive green/brown minimap texture
- Game timer text present at top-center (~x:930, y:60-75)
- Template match on team logos in the scoreboard bar
- Player camera overlays in bottom corners create a distinctive dark rectangular region

### 1.5 Replay

**Description**: During gameplay, the broadcast sometimes replays a recent highlight. The spectator HUD changes slightly during replays.

**Visual characteristics**:
- A "REPLAY" text indicator may appear overlaid on the screen (typically top-center area)
- The scoreboard bar may be partially transparent or slightly different in style
- The game timer may show the replay timestamp rather than the live game time
- Camera movement is smoother/more cinematic than live spectating

**Programmatic detection**:
- Check the top-center band (y: 3-8% of height, x: 35-65% of width) for bright pixels — replay banners produce a bright pixel ratio >4% in this region versus <2% for normal gameplay (this heuristic is already implemented in `VodProcessor.is_replay_frame()`)
- Some broadcasts add a "REPLAY" watermark or border effect
- Gold/kill numbers may be stale (not updating) compared to adjacent frames

### 1.6 Post-Game (Victory/Defeat Screen, Stats)

**Description**: After the nexus falls. Shows the victory explosion, then may transition to a post-game stats screen, then back to the analyst desk.

**Visual characteristics**:
- Large "VICTORY" or "DEFEAT" text overlay in the center of the screen
- Greyed-out/frozen gameplay in the background
- Post-game stats screen may show a full scoreboard with final KDA, gold, damage, etc.
- Eventually transitions back to pre-game style broadcast content

**Programmatic detection**:
- Large bright text in center screen (template match "VICTORY")
- Gameplay area becomes static (low frame-to-frame pixel difference)
- Scoreboard bar may disappear or transform into a post-game summary
- High structural similarity between consecutive frames (no champion movement)

---

## 2. In-Game HUD Layout (1920x1080)

All coordinates are given as (x1, y1, x2, y2) representing the top-left and bottom-right corners of the bounding box. Coordinates were estimated from the reference frames at native 1920x1080 resolution.

### 2.1 Top Scoreboard Bar

The scoreboard bar spans the full width of the screen at the top and contains the highest-density information.

#### 2.1.1 Overall Scoreboard Bar
| Property | Value |
|----------|-------|
| **Region** | (0, 0, 1920, 80) |
| **Description** | Semi-transparent dark bar across the top of the screen. Contains team logos, kill count, gold, turret count, dragon/baron trackers, and the game timer. |
| **Data type** | Composite (contains multiple sub-elements) |

#### 2.1.2 Blue Team Logo
| Property | Value |
|----------|-------|
| **Region** | (825, 2, 870, 30) |
| **Shows** | Blue-side team logo icon (e.g., G2 Esports) |
| **Data type** | Icon (template matching) |
| **Extraction difficulty** | Medium — small icon, but consistent per team |
| **Value for analysis** | Low (team identity is known from VOD metadata) |

#### 2.1.3 Red Team Logo
| Property | Value |
|----------|-------|
| **Region** | (1050, 2, 1095, 30) |
| **Shows** | Red-side team logo icon (e.g., Gen.G) |
| **Data type** | Icon (template matching) |
| **Extraction difficulty** | Medium |
| **Value for analysis** | Low |

#### 2.1.4 Blue Team Ranking Badge
| Property | Value |
|----------|-------|
| **Region** | (830, 30, 905, 55) |
| **Shows** | Team league/seed label (e.g., "LEC #1") |
| **Data type** | Text (OCR) |
| **Extraction difficulty** | Medium — small white text on dark background |
| **Value for analysis** | Low (metadata) |

#### 2.1.5 Red Team Ranking Badge
| Property | Value |
|----------|-------|
| **Region** | (1020, 30, 1095, 55) |
| **Shows** | Team league/seed label (e.g., "LCK #1") |
| **Data type** | Text (OCR) |
| **Extraction difficulty** | Medium |
| **Value for analysis** | Low (metadata) |

#### 2.1.6 Kill Score
| Property | Value |
|----------|-------|
| **Region** | (895, 0, 1025, 30) |
| **Shows** | Kill count for both teams, displayed as "X [blue-kills] [separator] [red-kills] Y" — in the observed frames, blue kills and red kills are separated by a small icon/divider at center. E.g., "0 ... 0" in early game, "3 ... 4" in mid game. |
| **Data type** | Number (OCR) |
| **Extraction difficulty** | Easy — large white numbers with high contrast on dark background |
| **Value for analysis** | **High** — kill differential is a core predictive feature |

**Note**: The kill numbers appear to be located slightly to the left and right of center. Blue kills at approximately (900, 2, 940, 28) and red kills at approximately (978, 2, 1020, 28).

#### 2.1.7 Blue Team Gold
| Property | Value |
|----------|-------|
| **Region** | (750, 0, 850, 28) |
| **Shows** | Total gold for blue team, displayed in "XX.Xk" format (e.g., "7.83", "23.1k", "48.1k") |
| **Data type** | Number (OCR) |
| **Extraction difficulty** | Easy-Medium — clear numbers but the "k" suffix and decimal point must be parsed |
| **Value for analysis** | **High** — gold differential is one of the most predictive features |

#### 2.1.8 Red Team Gold
| Property | Value |
|----------|-------|
| **Region** | (1070, 0, 1170, 28) |
| **Shows** | Total gold for red team, same format as blue |
| **Data type** | Number (OCR) |
| **Extraction difficulty** | Easy-Medium |
| **Value for analysis** | **High** |

#### 2.1.9 Blue Team Turret Count
| Property | Value |
|----------|-------|
| **Region** | (700, 0, 740, 28) |
| **Shows** | Number of turrets destroyed by blue team (single digit, 0-11). Displayed next to a small turret icon. |
| **Data type** | Number (OCR) + icon |
| **Extraction difficulty** | Medium — small number, sometimes overlaps with other elements |
| **Value for analysis** | **High** — turret differential indicates map control |

#### 2.1.10 Red Team Turret Count
| Property | Value |
|----------|-------|
| **Region** | (1180, 0, 1220, 28) |
| **Shows** | Number of turrets destroyed by red team |
| **Data type** | Number (OCR) + icon |
| **Extraction difficulty** | Medium |
| **Value for analysis** | **High** |

#### 2.1.11 Dragon Soul Tracker (Blue Side)
| Property | Value |
|----------|-------|
| **Region** | (5, 2, 65, 25) |
| **Shows** | Small colored icons indicating which drakes the blue team has taken (Infernal, Mountain, Ocean, Hextech, Cloud, Chemtech). Each drake is a small icon ~15x15px. Up to 4 icons before soul. |
| **Data type** | Icon (color/template matching) |
| **Extraction difficulty** | Hard — very small icons, color-coded but subtle |
| **Value for analysis** | **High** — dragon stacking toward soul is a major strategic milestone |

#### 2.1.12 Dragon Soul Tracker (Red Side)
| Property | Value |
|----------|-------|
| **Region** | (1855, 2, 1915, 25) |
| **Shows** | Drake icons for red team |
| **Data type** | Icon (color/template matching) |
| **Extraction difficulty** | Hard |
| **Value for analysis** | **High** |

#### 2.1.13 "Global Power Rankings" Banner
| Property | Value |
|----------|-------|
| **Region** | (860, 0, 1060, 15) |
| **Shows** | "GLOBAL POWER RANKINGS powered by [sponsor]" — a persistent broadcast overlay between the kill counts. Visible in frame_000500. |
| **Data type** | Text (decorative, not game data) |
| **Extraction difficulty** | N/A |
| **Value for analysis** | None (broadcast branding, not extractable game data) |

### 2.2 Game Timer

| Property | Value |
|----------|-------|
| **Region** | (925, 55, 995, 78) |
| **Shows** | In-game clock in MM:SS format (e.g., "10:57", "12:37", "25:58"). This is the actual game time, not the VOD timestamp. Displayed in white text just below the scoreboard bar center. |
| **Data type** | Text/Number (OCR) |
| **Extraction difficulty** | Easy — high contrast white text, consistent position, well-separated from other elements |
| **Value for analysis** | **High** — essential for temporal alignment of all extracted data; needed to compute game phase (early/mid/late) |

**Existing config value**: `timer: (920, 0, 1000, 30)` — this is **incorrect** based on the observed frames. The timer sits below the main scoreboard line, at approximately y:55-78, not y:0-30. The current config region likely captures the kill score area instead. **Recommend updating to (925, 55, 995, 78).**

### 2.3 Gold Difference Indicator

| Property | Value |
|----------|-------|
| **Region** | (900, 32, 1020, 55) |
| **Shows** | A visual gold difference bar or number shown between the two gold totals. In some frames, a colored bar extends left (blue leading) or right (red leading) with a numeric gold difference. The format appears to be a small number like "+1.2k" or a colored bar. |
| **Data type** | Visual indicator + number |
| **Extraction difficulty** | Medium — position is consistent but the visual style (bar vs number) may vary. In the observed frames the gold values appear as numbers flanking the center, with the difference derivable by subtraction. |
| **Value for analysis** | **High** — but can be computed from the individual gold values, so extracting it directly may be redundant |

### 2.4 Event Feed (Kill/Objective Announcements)

| Property | Value |
|----------|-------|
| **Region** | (1500, 85, 1900, 400) |
| **Shows** | Scrolling feed of recent game events: champion kills (killer icon -> victim icon), dragon/baron kills, turret destructions. Each event appears as a small banner with champion icons and item/ability icons. Events fade after ~5 seconds. |
| **Data type** | Icons + text (complex composite) |
| **Extraction difficulty** | **Hard** — events are transient (appear and disappear), overlapping, and contain small icons. The feed uses champion-specific icons and color coding (blue/red team). |
| **Value for analysis** | Medium — kill events are valuable but are also captured in the kill score counter. Dragon/Baron events are high-value but hard to extract from the feed. |

### 2.5 Player Scorecards (Bottom Panel)

The bottom of the screen contains a dark panel showing detailed stats for all 10 players, arranged in two columns (blue team left, red team right). This panel is always visible during gameplay.

#### 2.5.1 Bottom Panel Overall
| Property | Value |
|----------|-------|
| **Region** | (195, 815, 1640, 1040) |
| **Description** | Dark semi-transparent panel containing 5 rows per team. Blue team on the left half, red team on the right half, with player stats in tabular format. |

#### 2.5.2 Blue Team Player Rows

Each blue-team player occupies a horizontal row. The 5 rows are stacked vertically. Per-player layout (approximate regions for Player 1 / top lane):

| Sub-element | Region (Player 1) | Description |
|-------------|-------------------|-------------|
| **Champion Portrait** | (195, 820, 235, 855) | Square champion icon (~40x35px). Identifies the champion being played. |
| **Player Name** | (240, 820, 370, 838) | Player's in-game name (e.g., "BrokenBlade"). White text on dark background. |
| **KDA** | (380, 820, 450, 838) | Kill/Death/Assist in "K/D/A" format (e.g., "0/0/0", "1/2/0"). White text. |
| **CS (Creep Score)** | (455, 820, 510, 838) | Minion/monster kills count. Numeric value (e.g., "121", "232", "315"). |
| **Items** | (520, 820, 650, 855) | Up to 6 item icons in a row (~20x20px each) plus a control ward slot. Small colored squares. |
| **Summoner Spell 1** | Embedded in champion portrait area | Tiny icon for first summoner spell (Flash, Teleport, etc.) |
| **Summoner Spell 2** | Embedded in champion portrait area | Tiny icon for second summoner spell |

**Row spacing**: Each subsequent player row is offset by approximately 38-42px vertically.

| Player (Blue) | Row Region (approximate) |
|---------------|-------------------------|
| Player 1 (Top) | (195, 818, 730, 855) |
| Player 2 (Jungle) | (195, 858, 730, 895) |
| Player 3 (Mid) | (195, 898, 730, 935) |
| Player 4 (ADC) | (195, 938, 730, 975) |
| Player 5 (Support) | (195, 978, 730, 1015) |

#### 2.5.3 Red Team Player Rows

The red team panel mirrors the blue team on the right side. The layout is the same but horizontally flipped (champion portrait on the right side of the row).

| Player (Red) | Row Region (approximate) |
|--------------|-------------------------|
| Player 1 (Top) | (1190, 818, 1640, 855) |
| Player 2 (Jungle) | (1190, 858, 1640, 895) |
| Player 3 (Mid) | (1190, 898, 1640, 935) |
| Player 4 (ADC) | (1190, 938, 1640, 975) |
| Player 5 (Support) | (1190, 978, 1640, 1015) |

#### 2.5.4 Champion Portrait (per player)
| Property | Value |
|----------|-------|
| **Size** | ~40x35px per champion |
| **Data type** | Icon (template matching / CNN classification) |
| **Extraction difficulty** | Medium — small but distinctive enough for template matching against known champion icon set. The YOLO minimap model already recognises 167 champions by icon. |
| **Value for analysis** | Medium — champion identity is useful for contextualising other features, but is also knowable from the draft phase |

#### 2.5.5 Player Name (per player)
| Property | Value |
|----------|-------|
| **Data type** | Text (OCR) |
| **Extraction difficulty** | Medium — small white text (~10px height), may contain mixed case and special characters |
| **Value for analysis** | Low — player identity is known from VOD metadata / draft |

#### 2.5.6 KDA (per player)
| Property | Value |
|----------|-------|
| **Data type** | Text/Number (OCR) — format "K/D/A" with slashes |
| **Extraction difficulty** | Medium — small text, slashes can be misread. Numbers 0-20+ range. |
| **Value for analysis** | **High** — individual KDA contributes to kill differential and is a strong performance indicator |

#### 2.5.7 CS / Creep Score (per player)
| Property | Value |
|----------|-------|
| **Data type** | Number (OCR) |
| **Extraction difficulty** | Medium — 1-3 digit number, small text |
| **Value for analysis** | **High** — CS differential indicates laning performance and farming efficiency |

#### 2.5.8 Items (per player)
| Property | Value |
|----------|-------|
| **Size** | 6 slots, each ~20x20px |
| **Data type** | Icon (image classification) |
| **Extraction difficulty** | **Hard** — icons are very small (20x20px), there are 200+ possible items, and empty slots look similar to certain items. Would require a dedicated item icon classifier or template library. |
| **Value for analysis** | Medium — item builds indicate power spikes but are difficult to extract reliably. Gold total is a more practical proxy for item power. |

#### 2.5.9 Champion Level
| Property | Value |
|----------|-------|
| **Region** | Small number overlaid on or near the champion portrait |
| **Data type** | Number (OCR), range 1-18 |
| **Extraction difficulty** | **Hard** — very small text (~8px), overlaid on the champion icon, often partially occluded |
| **Value for analysis** | Medium — level advantages matter but are correlated with gold and XP, which are partially captured by other features |

#### 2.5.10 Ultimate Ability Indicator
| Property | Value |
|----------|-------|
| **Region** | Small green dot or circular indicator near the champion portrait |
| **Data type** | Visual indicator (binary: ready / not ready, shown as a green dot vs grey/absent) |
| **Extraction difficulty** | **Hard** — tiny visual element, color-based detection needed |
| **Value for analysis** | Medium — ultimate availability matters for team fight potential but changes rapidly |

#### 2.5.11 Summoner Spells
| Property | Value |
|----------|-------|
| **Region** | Two tiny icons (~12x12px) near the champion portrait |
| **Data type** | Icon + cooldown overlay (greyed out with countdown when on cooldown) |
| **Extraction difficulty** | **Hard** — extremely small icons. Cooldown detection requires distinguishing colored (ready) from greyed-out (on cooldown) states. |
| **Value for analysis** | Medium — Flash availability matters for engages, but the icons are too small for reliable extraction |

### 2.6 Minimap

| Property | Value |
|----------|-------|
| **Region** | (1650, 790, 1920, 1080) |
| **Shows** | Top-down view of the entire Summoner's Rift map. Shows all 10 champion icons (full-vision spectator mode, no fog of war), turret positions, and the terrain. Champion icons are approximately 15-25px diameter circles on the minimap. |
| **Data type** | Spatial (YOLO object detection for champion positions) |
| **Extraction difficulty** | Medium — the minimap is a well-defined region, and the pyLoL YOLO model achieves 92.2% mAP on champion detection. The main challenges are champion overlaps, clone champions, and the minimap's small size (270x290px before resize). |
| **Value for analysis** | **Critical** — this is the primary data source for spatial features (zone transitions, team grouping, objective convergence). It is the centerpiece of the CV pipeline. |

**Current config value**: `minimap: (1650, 790, 1920, 1080)` — this appears **correct** based on the observed frames. The minimap occupies the bottom-right corner with a slight decorative border.

### 2.7 Player Camera Overlays (Webcam Feeds)

| Property | Value |
|----------|-------|
| **Blue team camera** | (0, 870, 195, 1080) — bottom-left corner |
| **Red team camera** | (1640, 870, 1920, 1080) — bottom-right corner (partially behind/beside minimap) |
| **Shows** | Live webcam footage of one player per team (the player currently being spectated or a featured player). Below each camera is a player name label. |
| **Data type** | Video feed (not directly useful for game data) |
| **Extraction difficulty** | N/A |
| **Value for analysis** | None for game data. Could theoretically be used for player emotion analysis but that is out of scope. |

**Note**: The player camera overlays contain player name labels at the bottom: blue side label at approximately (60, 1050, 195, 1075) and red side label at approximately (1640, 1050, 1780, 1075). These show the currently featured player's name in ALL CAPS (e.g., "CAPS", "HANS SAMA", "CHOVY", "RULER").

### 2.8 Sponsor Logo (Rotating)

| Property | Value |
|----------|-------|
| **Region** | (0, 1020, 190, 1080) — bottom-left corner, below/overlapping the player camera |
| **Shows** | Rotating sponsor logos: Coinbase, AWS, HyperX observed across different frames. Changes periodically (every ~30-60 seconds). |
| **Data type** | Logo/image |
| **Extraction difficulty** | N/A |
| **Value for analysis** | None |

### 2.9 Patch Version Indicator

| Property | Value |
|----------|-------|
| **Region** | (55, 795, 155, 815) |
| **Shows** | "PATCH 26.5" or similar patch version label. Small grey text visible in the bottom-left area above the player panels. |
| **Data type** | Text (OCR) |
| **Extraction difficulty** | Medium — small grey text |
| **Value for analysis** | Low — patch version is known from tournament metadata |

### 2.10 Broadcast Frame Counter / Stream Info

| Property | Value |
|----------|-------|
| **Region** | (0, 790, 50, 815) |
| **Shows** | Small "26" or broadcast production number in the bottom-left. May be a scene/camera indicator for the production team. |
| **Data type** | Text/number |
| **Extraction difficulty** | Easy but irrelevant |
| **Value for analysis** | None |

### 2.11 Baron/Dragon Objective Timers

| Property | Value |
|----------|-------|
| **Region** | Overlaid in the upper-left area, approximately (0, 0, 80, 30) |
| **Shows** | When a major objective (Baron Nashor, Dragon) is about to spawn, a timer and icon may appear in the top-left or as a popup near the minimap. In frame_000500, small dragon/baron indicators are visible at approximately (5, 2, 70, 25). In later frames, dragon icons with a respawn timer are visible. |
| **Data type** | Icon + number (timer countdown) |
| **Extraction difficulty** | Hard — icons are small and overlap with drake tracker; timers are transient |
| **Value for analysis** | **High** — objective timing is strategically important, but the timer is only visible briefly before spawn |

### 2.12 "W-28" Style Indicator (Ability Cooldown Overlay)

| Property | Value |
|----------|-------|
| **Region** | Variable — appears overlaid on the spectated champion's position |
| **Shows** | When the spectator focuses on a specific champion, ability cooldown indicators may appear near that champion. In frame_001200, "W-28" is visible near a champion, indicating the W ability has 28 seconds of cooldown. |
| **Data type** | Text overlay on gameplay area |
| **Extraction difficulty** | **Hard** — position varies with camera, transient, mixed with gameplay visuals |
| **Value for analysis** | Low — too unreliable and transient to extract systematically |

### 2.13 Champion Health/Mana Bars (In Gameplay Area)

| Property | Value |
|----------|-------|
| **Region** | Variable — above each champion model in the gameplay view |
| **Shows** | Green health bar, blue mana bar, champion name label, and level indicator above each visible champion. |
| **Data type** | Visual indicator (color bars) |
| **Extraction difficulty** | **Hard** — positions move with champions, bars are small, and the camera zooms/pans constantly |
| **Value for analysis** | Low for systematic extraction. Health states are transient and would require tracking at very high FPS. |

---

## 3. Extraction Feasibility Assessment

### 3.1 OCR-Based Extraction (Text/Number Elements)

| Element | OCR Feasibility | Notes |
|---------|----------------|-------|
| Game Timer | **High** | Large, high-contrast text. Consistent position. Already implemented. Region needs correction (see 2.2). |
| Kill Score | **High** | Large numbers, fixed position. Simple regex parsing ("X - Y"). Already partially implemented. |
| Team Gold | **High** | Clear numbers in "XX.Xk" format. Need to handle the "k" suffix. Position is stable. |
| Turret Count | **Medium** | Small single-digit number. May be confused with adjacent UI elements. |
| Player KDA | **Medium** | Small text but structured format (K/D/A). 10 separate OCR reads needed (one per player). Slash separators may cause OCR errors. |
| Player CS | **Medium** | Small numbers, 1-3 digits. Position is consistent within the player row. |
| Player Name | **Medium-Low** | Small text, mixed case, sometimes abbreviated. Not needed (available from metadata). |
| Patch Version | **Low** | Very small grey text. Not needed for analysis. |

### 3.2 Icon/Template Matching

| Element | Detection Feasibility | Notes |
|---------|----------------------|-------|
| Champion Icons (minimap) | **High** | pyLoL YOLO model achieves 92.2% mAP. Already implemented. 270x290px minimap resized to 512x512 for detection. |
| Champion Portraits (scorecards) | **Medium** | ~40x35px icons. Could use template matching against a library of champion square icons (available from Riot's Data Dragon CDN). |
| Team Logos | **Medium** | Distinctive logos, consistent position. Template matching feasible. Not high priority. |
| Dragon Type Icons | **Medium-Low** | ~15x15px, color-coded. Could use color histogram matching (Infernal=red, Ocean=blue, Mountain=brown, Cloud=white, Hextech=blue-white, Chemtech=green). Very small though. |
| Item Icons | **Low** | 20x20px with 200+ possible items. Too small and too many classes for reliable detection without a dedicated model. |
| Summoner Spell Icons | **Low** | ~12x12px, extremely small. Not feasible for reliable classification. |
| Ultimate Ready Indicator | **Low** | Tiny green dot. Binary detection might work with color thresholding but the region is very small. |

### 3.3 Positional Consistency

| Element | Consistency | Notes |
|---------|------------|-------|
| Top scoreboard bar | **Very consistent** | Fixed position across all gameplay frames. Same layout in every match. |
| Game timer | **Very consistent** | Always at top-center, same font and position. |
| Player scorecards | **Very consistent** | Fixed position. Player order (Top/Jg/Mid/ADC/Sup) is always the same top-to-bottom. |
| Minimap | **Very consistent** | Always bottom-right corner, same size. |
| Event feed | **Inconsistent** | Events appear and disappear, variable number of items, position shifts vertically as new events push old ones up. |
| Objective timers | **Inconsistent** | Only appear near spawn times. Position may vary. |
| Ability cooldown overlays | **Inconsistent** | Follow the spectated champion; position changes constantly. |

### 3.4 Occlusion Issues

| Element | Occlusion Risk | Notes |
|---------|---------------|-------|
| Top scoreboard bar | **None** | Always visible during gameplay, never obscured. |
| Game timer | **None** | Never obscured during gameplay. |
| Player scorecards | **Low** | Always visible. Sponsor logo may overlap slightly in the bottom-left corner but does not cover player stats. |
| Minimap | **Low** | The red-team player camera overlay sits adjacent to the minimap but does not cover it. |
| Event feed | **None** | Always rendered on top of gameplay. |
| Gameplay area | **High** | The spectator camera constantly pans, zooms, and cuts. Champion positions in the gameplay view are unreliable for tracking. |

---

## 4. Recommended Extraction Priority

### 4.1 Must Extract (Core to Research Questions)

These elements are essential for answering the research questions about predicting match outcomes from CV-extracted features.

| Element | Region | Method | Justification |
|---------|--------|--------|---------------|
| **Minimap champion positions** | (1650, 790, 1920, 1080) | YOLO detection | Primary data source for RQ1 spatial features (zone transitions, grouping, objective convergence). Already implemented. |
| **Game timer** | (925, 55, 995, 78) | OCR | Required for temporal alignment of all features. Defines early/mid/late game phases. Config needs correction. |
| **Kill score (blue & red)** | (895, 0, 1025, 28) | OCR | Kill differential is a top-tier predictive feature. Already partially implemented. |
| **Team gold (blue & red)** | Blue: (750, 0, 850, 28), Red: (1070, 0, 1170, 28) | OCR | Gold differential is the single most predictive feature in LoL match outcome prediction. |
| **Turret count (blue & red)** | Blue: (700, 0, 740, 28), Red: (1180, 0, 1220, 28) | OCR | Turret differential indicates map control progression. |

### 4.2 Should Extract (Significant Value)

These elements add meaningful analytical depth beyond the core features.

| Element | Region | Method | Justification |
|---------|--------|--------|---------------|
| **Dragon tracker (blue & red)** | Blue: (5, 2, 65, 25), Red: (1855, 2, 1915, 25) | Color/template matching | Dragon soul progress is a major strategic indicator. Detecting the number of drakes taken (even without type classification) is valuable. |
| **Player KDA (all 10 players)** | Per-player rows in (195, 818, 730, 1015) and (1190, 818, 1640, 1015) | OCR | Individual performance data enables richer analysis. 10 OCR reads per frame but structured format. |
| **Player CS (all 10 players)** | Same rows as KDA | OCR | CS differential, especially at early time points, indicates laning phase performance. |
| **Phase detection (draft vs gameplay vs replay)** | Various heuristics (see Section 1) | Pixel analysis | Needed to filter out non-gameplay frames and avoid corrupted feature extraction. Already partially implemented (replay detection). |

### 4.3 Nice to Have (Diminishing Returns)

| Element | Region | Method | Justification |
|---------|--------|--------|---------------|
| Champion portraits (scorecards) | Per-player in bottom panel | Template matching | Useful for automatic champion identification, but champion picks are known from the draft phase and match metadata. |
| Dragon type classification | (5, 2, 65, 25) and (1855, 2, 1915, 25) | Color histogram | Knowing Infernal vs Ocean vs Mountain drakes adds context, but drake count alone captures most of the value. |
| Champion level (per player) | Overlaid on portrait | OCR (very small text) | Level advantages are meaningful but strongly correlated with gold/XP, which are captured elsewhere. |
| Items (per player) | 6 slots per player in scorecards | CNN classification | Would require a dedicated item classifier. Gold total is a much more practical proxy for item power. |
| Event feed parsing | (1500, 85, 1900, 400) | OCR + icon detection | Kill events are already captured in kill score; dragon/baron events are valuable but the transient nature makes extraction unreliable. |

### 4.4 Skip (Too Complex for the Value)

| Element | Reason |
|---------|--------|
| Summoner spell icons & cooldowns | 12x12px icons — too small for reliable detection. Cooldown state is transient. |
| Ultimate ready indicator | Tiny visual cue with marginal analytical value for match outcome prediction. |
| Champion health/mana bars (gameplay area) | Positions move with camera. Would require champion tracking in the gameplay view (not minimap). Extremely complex for limited value. |
| Ability cooldown overlays (W-28 style) | Transient, position varies, only shown for spectated champion. |
| Player camera feeds | Not game data. |
| Sponsor logos | Not game data. |
| Broadcast branding text | Not game data. |

---

## 5. Config Update Recommendations

Based on the frame analysis, the following updates to `configs/default.yaml` are recommended:

```yaml
extraction:
  ocr:
    regions:
      # CORRECTED regions based on visual analysis of 2026 First Stand broadcast
      scoreboard: [0, 0, 1920, 80]           # Full top bar (unchanged)
      timer: [925, 55, 995, 78]               # CORRECTED: was [920, 0, 1000, 30]
      kill_score: [895, 0, 1025, 28]          # NEW: explicit kill score region
      blue_gold: [750, 0, 850, 28]            # NEW: blue team gold
      red_gold: [1070, 0, 1170, 28]           # NEW: red team gold
      blue_turrets: [700, 0, 740, 28]         # NEW: blue turret count
      red_turrets: [1180, 0, 1220, 28]        # NEW: red turret count
      blue_drakes: [5, 2, 65, 25]             # NEW: blue dragon tracker
      red_drakes: [1855, 2, 1915, 25]         # NEW: red dragon tracker
      blue_scorecards: [195, 818, 730, 1015]  # NEW: blue player stats panel
      red_scorecards: [1190, 818, 1640, 1015] # NEW: red player stats panel
      minimap: [1650, 790, 1920, 1080]        # Unchanged (confirmed correct)

vod_processing:
  hud_regions:
    scoreboard: [0, 0, 1920, 80]
    timer: [925, 55, 995, 78]                 # CORRECTED
    kill_score: [895, 0, 1025, 28]            # NEW
    blue_gold: [750, 0, 850, 28]             # NEW
    red_gold: [1070, 0, 1170, 28]            # NEW
  minimap_region: [1650, 790, 1920, 1080]     # Unchanged
```

### 5.1 Existing Code Impact

- **`src/lol_cv/extraction/ocr.py`**: The `DEFAULT_REGIONS["timer"]` is set to `(910, 0, 1010, 35)` which captures the area above the actual timer. Should be updated to `(925, 55, 995, 78)`. The `kill_score` region `(870, 0, 1050, 40)` appears reasonable but could be tightened. New regions should be added for gold and turrets.
- **`src/lol_cv/extraction/vod_processor.py`**: The `DEFAULT_HUD_REGIONS` dict has only `scoreboard` and `timer`. Should be expanded with the new regions.
- **`configs/default.yaml`**: Timer region needs correction as noted above.

---

## 6. Visual Reference Summary

```
+--[Drake]----[Turr][Gold][Blue Logo]--[Kills]--[Red Logo][Gold][Turr]--[Drake]--+  y=0
|                        [GLOBAL POWER RANKINGS]                                  |
|                              [  Timer  ]                                        |  y=55-78
|                                                                                 |  y=80
|                                                                                 |
|                                                                                 |
|                        [ GAMEPLAY AREA ]                          [Event Feed]  |
|                        [ Spectator Camera View ]                                |
|                                                                                 |
|                                                                                 |
|                                                                                 |
|  [Patch]                                                                        |  y=795
|  +----[Blue Scorecards]------+              +------[Red Scorecards]----+  [MAP] |  y=818
|  | P1: [icon] name  KDA  CS  items |        | P1: name KDA CS items [icon]| |M| |
|  | P2: [icon] name  KDA  CS  items |        | P2: name KDA CS items [icon]| |I| |
|  | P3: [icon] name  KDA  CS  items |        | P3: name KDA CS items [icon]| |N| |
|  | P4: [icon] name  KDA  CS  items |        | P4: name KDA CS items [icon]| |I| |
|  | P5: [icon] name  KDA  CS  items |        | P5: name KDA CS items [icon]| |M| |
|  +------[WEBCAM]-----[Sponsor]-----+        +--------[WEBCAM]-------------+[AP] |  y=1080
+-x=0---------------------------------------------------------------------x=1920-+
```

---

*Document generated from visual analysis of frames: frame_000104.png (draft), frame_000350.png (draft), frame_000500.png (early game ~10:57), frame_001200.png (mid game ~12:37), frame_002000.png (late game ~25:58) from G2 vs GEN semifinals Game 1, 2026 First Stand tournament.*
