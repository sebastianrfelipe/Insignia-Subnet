# Insignia Technologies, Deck Design System

*Applied to `Insignia Investor Deck V2.af`, July 2026. Slide canvas: 6000 × 3375 units (16:9).*

---

## Philosophy, "Instrument Panel"

The deck reads like a precision instrument, not a brochure. Every element sits
on a fixed armature; nothing floats. The dark circuit photography stays as
atmosphere and never competes, content lives on translucent glass panels that
let the background breathe through at a controlled 11% white. A single lavender
accent does all the signalling: titles, rules, markers, borders. Everything else
is white. Restraint is the whole point, when only two colours and one typeface
are in play, hierarchy has to be carried by size, weight and position, which is
what makes the result read as deliberate rather than decorated.

---

## Grid

| Token | Value | Notes |
|---|---|---|
| Left / right margin | **360** | All content starts at x = 360, ends at x = 5640 |
| Content width | **5280** | |
| Title cap-top | **y = 340** | Identical on every slide |
| Second title line | **y = 730** | |
| Content top | **y = 1150** | First element below the title block |
| Content bottom | **y = 3060** | |
| Two-column split | **360–2940 / 3060–5640** | 120 gutter |
| Three-column split | **360 / 2163 / 3966**, width 1674 | 129 gutter |

---

## Type scale

One family: **Arial**. Tracking is **0 everywhere**, no optical fudging.

| Role | Size | Weight | Colour |
|---|---|---|---|
| Display (Thank You) | 460 | Regular | Accent |
| Slide title | 340 | Regular | Accent |
| Statement / stat | 260 | Bold (stats) / Regular (statements) | White |
| Lead paragraph | 190 | Regular | White |
| Card statement | 150 | Bold | White |
| Section heading / eyebrow | 120 | Bold | White (Accent for "Key Takeaway") |
| Body | 105 | Regular | White |
| Label / small caps | 90 | Bold | Accent |
| Caption / footnote | 72 | Regular | White |

---

## Colour

| Token | Value | Use |
|---|---|---|
| Accent | `#856ED1` | Titles, rules, markers, borders, dates |
| White | `#FFFFFF` | All body and heading text |
| Panel fill | `#FFFFFF` @ alpha 28 (≈11%) | Every card and panel |
| Panel stroke | `#856ED1` @ alpha 205, weight 6 | Every card and panel |

No greys, no secondary blues, no hyperlink blue. Two colours, full stop.

---

## Components

**Panel / card**, rounded rectangle, corner radius 48, panel fill + panel
stroke. Used identically for the Problem and Solution frames, the three
Business Model cards, the Key Takeaway box, the two Why Us cards and the
Roadmap bar. Inner text padding: 180.

**Accent rule**, 24-wide pill, radius 12, solid accent, inset 60 from the card
edge, running the card height minus 120. Marks a card as a claim.

**Timeline**, 10-unit accent track at 40% opacity, 96-diameter accent dots
centred on it, date label above in accent 90 bold, milestone name in white 120
bold.

**Corner marks**, two, and only two, on every content slide: the circuit
graphic at top-right (x 4315, y 340) and the dash bar at bottom-left (x 0,
y 3060). Both are pinned to identical coordinates on all nine content slides.
The wandering chevron cluster was removed, the two fixed marks carry the motif
without moving between corners.

---

## Rules of thumb

1. If it isn't on the grid, move it to the grid.
2. If it needs a new size, use one already in the scale.
3. Tracking is 0. Word spacing is 100%. Never hand-kern to fit.
4. New containers get the panel fill and stroke, radius 48, no exceptions.
5. Titles are left-aligned at 360 / 340. Centred titles break the system.
