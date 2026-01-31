# Recalculate

The Recalculate page helps you understand how different classification rules affect the size and composition of your validation set. It lets you model how changing thresholds can increase or reduce the number of detections you review, and helps you explore the trade-off between potential false positives and false negatives.

![Recalculate overview](assets/screenshots/recalculate.png){ width="1100" }

## What Recalculate does

Recalculate updates the number of detections included in your validation set based on alternative decision rules. As you adjust settings, PAMalytics:

- recalculates the **total number of detections**
- recomputes the **breakdown by your chosen grouping variable** (for example species, site, or any other metadata)

## Ways to adjust classification rules

There are two main approaches.

### Adjust the detection threshold

You can change the detection threshold used to classify a detection as “present” (either for presence/absence or for a particular species), relative to the default of **0.5**.

In general:

- higher thresholds reduce the number of detections (often reducing false positives)
- lower thresholds increase the number of detections (often reducing false negatives, but increasing review load)

### Apply a *k of n* rule

You can also enforce a *k of n* rule, which requires a minimum number of “present” detections within a short sequence.

For example, a **2 of 5** rule means a detection is only included if there are at least **2 classifier detections** within **5 consecutive audio files**.

This is especially helpful for taxa with longer calling patterns, where genuine presence is more likely to appear across neighbouring recordings rather than as isolated single detections.

When you apply a *k of n* rule, PAMalytics recalculates totals and group-level breakdowns using the updated rule.
