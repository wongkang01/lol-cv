# Design System Document: The Cinematic Nexus
 
## 1. Overview & Creative North Star
**Creative North Star: The Kinetic Archive**
 
This design system is engineered to feel less like a traditional website and more like a high-end broadcast terminal or a tactical HUD. It is designed to house immense amounts of data—stats, schedules, and live streams—while maintaining a cinematic "big screen" presence. 
 
To break the "template" look, we move away from symmetrical grids and standard padding. We lean into **Intentional Asymmetry**: using large, aggressive typographic anchors on the left balanced by dense, data-rich modules on the right. Surfaces should overlap, creating a sense of depth where live content feels like it is projected onto a physical obsidian-glass interface.
 
## 2. Colors & Surface Logic
 
The palette is rooted in deep obsidian tones, punctuated by high-frequency electric blues.
 
### The "No-Line" Rule
**Explicit Instruction:** Traditional 1px solid borders for sectioning are strictly prohibited. The UI must feel like a single, seamless machine. Define boundaries using:
- **Tonal Shifts:** Placing a `surface-container-low` component against a `surface` background.
- **Vignettes:** Subtle radial gradients that darken toward the edges of a section to pull the eye inward.
- **Negative Space:** Aggressive use of the spacing scale to separate logical blocks.
 
### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. Use the `surface-container` tiers to define "Z-axis" importance:
1.  **Base Layer:** `surface` (#151313) — The infinite canvas.
2.  **Navigation/Layout Blocks:** `surface-container-low` (#1d1b1b) — Defines large regions.
3.  **Active Cards/Modules:** `surface-container-high` (#2c2929) — Floating elements.
4.  **Interactive Overlays:** `surface-container-highest` (#373434) — Modals and tooltips.
 
### The "Glass & Gradient" Rule
To achieve a premium, high-tech finish, utilize **Glassmorphism**. Floating sidebars or stats-panels should use semi-transparent surface colors with a `backdrop-filter: blur(20px)`. 
- **Signature Texture:** Use a linear gradient from `primary` (#b9f6ff) to `primary-container` (#10e4f9) at a 45-degree angle for primary CTAs and active state indicators. This provides a "neon-tube" glow that flat colors lack.
 
## 3. Typography: The Editorial Edge
 
The system utilizes a high-contrast pairing between **Space Grotesk** (Display/Headline) and **Inter** (Body/Labels).
 
*   **Display & Headlines (Space Grotesk):** These are your architectural anchors. Use `display-lg` for heroic moments (e.g., "WORLDS FINALS"). The wide, geometric stance of Space Grotesk provides the "High-Tech" authority required for an elite league.
*   **Body & Labels (Inter):** Inter handles the "data-rich" heavy lifting. It is optimized for legibility at small sizes in dense tables.
*   **Tonal Authority:** Use `label-md` in all-caps with a `0.05em` letter-spacing for metadata (e.g., "MATCH STARTING IN"). This mimics the technical readout of a flight computer.
 
## 4. Elevation & Depth: Tonal Layering
 
We do not use shadows to mimic light from above; we use layering to mimic light from within.
 
*   **The Layering Principle:** Place a `surface-container-lowest` card on a `surface-container-low` section to create a "recessed" look, or a `surface-container-high` card on a `surface` background for a "raised" look.
*   **Ambient Shadows:** For "floating" elements like match-selection tooltips, use a shadow with a 40px blur, 0% spread, and an opacity of 6%. The color should be `primary` (#b9f6ff) to create a subtle cyan atmospheric glow rather than a muddy black shadow.
*   **The "Ghost Border" Fallback:** If a container requires definition for accessibility, use the `outline-variant` (#3b494b) at **15% opacity**. This creates a "barely-there" structural guide that disappears into the cinematic aesthetic.
 
## 5. Components
 
### Buttons
*   **Primary:** `primary-container` (#10e4f9) background with `on-primary-container` text. Use `sm` (0.125rem) corner radius for a sharp, aggressive look.
*   **Secondary:** Ghost style. `outline-variant` border (at 20% opacity) with `primary` text. On hover, the background fills to 10% opacity.
 
### Data Chips
*   **Selection Chips:** Use `secondary-container` (#46456d). When active, add a 2px left-accent border of `primary-fixed-dim` (#00dbef).
*   **Status Chips (Live):** Background of `tertiary-container` (red tones) with a 1px `tertiary` inner glow.
 
### Input Fields
*   **State Styling:** Background must be `surface-container-highest`. No borders. A `primary` 2px bottom-bar appears only on `:focus`.
 
### Cards & Lists
*   **Prohibition:** Dividers/Lines are forbidden. 
*   **Layout:** Separate list items using a background shift to `surface-container-low`.
*   **Hover State:** Cards should not "lift" with a shadow. Instead, transition the background color to a slightly brighter `surface-bright` (#3b3838) and increase the `backdrop-filter` intensity.
 
### Cinematic Components
*   **The Progress HUD:** For match timers or loading, use a thin `primary` circular or linear stroke with a `primary_fixed` outer glow.
*   **The "Glow" Badge:** For "LIVE" indicators, use a solid `tertiary` (#ffe6e2) background with a pulse animation that affects a 10px-wide blurred drop-shadow.
 
## 6. Do's and Don'ts
 
### Do:
*   **Use Intentional Asymmetry:** Align headings to the far left and let data-heavy tables bleed off the right edge of the grid for a "command center" feel.
*   **Embrace the Dark:** Ensure 90% of the UI remains in the `surface` to `surface-container-high` range.
*   **Focus on Typography Scale:** Use the massive difference between `display-lg` and `label-sm` to create visual excitement.
 
### Don't:
*   **Don't use Rounded Corners:** Avoid `xl` or `full` rounding unless it is a profile avatar. Stick to `none`, `sm`, or `md` to keep the tech-edge sharp.
*   **Don't use 100% Opacity Borders:** High-contrast lines break the cinematic immersion. If you can see the line clearly, it's too thick/opaque.
*   **Don't use Pure White for Body Text:** Use `on-surface-variant` (#bac9cc) for long-form reading to reduce eye strain in dark environments. Save `on-surface` (#e7e1e0) for high-priority headings.