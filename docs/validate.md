# Validate

The Validate page is where you review detections, correct mistakes, and export a validated dataset.

![Validate overview](assets/screenshots/validate.png){ width="1100" }

## What Validate is for

Validation is designed for fast review:

- Track what proportion of clips have been validated by site × species (or any other grouping)
- Monitor classifier accuracy as you review
- Sort and filter by clip probability (min–max segment probability per file)
- Use high-resolution spectrograms for quick visual checks
- Use audio playback with automated time expansion for ultrasonic calls
- Review pending changes before saving

## Validation summary and filters

![Validate details](assets/screenshots/validate1.png){ width="1100" }

At the top of the page you’ll see a summary of validation progress, including:

- How many detections have been ingested
- The proportion reviewed
- The percentage changed
- The percentage the classifier got correct

Below the headline figures, PAMalytics breaks these metrics down by species, site (and other metadata), so you can balance your validation effort in line with your sampling strategy.

Use the expandable **Advanced filters** panel to focus your review effort. For example, you can:

- Order clips from lowest to highest classifier confidence (useful for targeting potential false positives)
- Filter to a specific site or species
- Set a fixed frequency range for spectrogram visualisation
- Pre-select a preferred time expansion factor for audio

## Review spectrogram cards

Below the summary and filters you’ll review detections using spectrogram “cards”, similar to the Detection Dashboard but with additional validation controls.

![Validate details](assets/screenshots/validate2.png){ width="1100" }

Each card includes two status indicators in the top-right:

- **Review status** (default: *Not reviewed*)
- **Classifier performance** (default: *Not assessed*)

### Mark a card as reviewed

If all detections on the card are correct, select **Mark card as reviewed**. The indicators update (e.g. *Reviewed* and *Classifier: all correct*) and the summary table refreshes immediately.

### Correct detections

If you disagree with any detection:

- Open the dropdown beneath the spectrogram
- Locate the detection you want to change
- Update it to **Absent** or to the correct species

After making edits, select **Mark card as reviewed**. The classifier performance indicator will update (for example *Classifier: mixed* or *Classifier: all incorrect*).

Any changes you make are recorded in the dynamic table beneath the spectrogram cards, so you can keep track of edits before exporting.

## Export your validated results

When you’ve completed your review, download the validated CSV using the **Download CSV** option at the bottom of the page.

Your export will contain seven additional columns relating to the validation effort:

- `species_name_original`: the species label provided by the classifier
- `presence_label_original`: the presence/absence label provided by the classifier
- `validation_state`: whether the reviewer agreed with the classifier prediction (e.g. correct/incorrect/not reviewed)
- `validation_label`: the reviewer’s final presence/absence decision
- `validation_species`: the reviewer’s final species label decision
- `validated_by`: the reviewer name
- `validated_at`: the date/time of the review
