# Set up your first project

When you log in for the first time, you’ll land on the Project Dashboard and be prompted to create your first project.

Enter a project name, select **Create project**, and PAMalytics will take you to the **Data mapping** screen.

![Data mapping screen](assets/screenshots/data_mapping.png)

## Choose your mapping method

On the Data mapping screen you can either:

- use a **pre-built adapter** (BirdNET or BatDetect2), or
- use **Manual mapping** for any other classifier output.

### Using a pre-built adapter (BirdNET / BatDetect2)

If you choose a pre-built adapter:

- navigate to the folder containing your **classifier detections**
- navigate to the folder containing your **audio files**
- select **Ingest folder**

You can also set a **detection threshold** (the classifier confidence required for a “present” detection). If you’re unsure, keep the default value of **0.5**.

After ingesting, PAMalytics shows a quick summary of the **percentage of detections with audio available for review**. If that figure looks wrong, you can re-run the data mapping. When it looks right, select **Launch PAMalytics** to continue to the main dashboard.

## Manual mapping for other classifiers

If you’re using a different classifier, choose **Manual mapping** and select your detections and audio folders in the same way.

Manual mapping involves two steps:

### Link detections to audio files

First, confirm which column in your results contains the **recording file identifier** (a filename or file path). PAMalytics uses this to link each detection row to its corresponding audio file.

Once you select the correct column, PAMalytics calculates the proportion of detection rows that can be linked to audio. If it looks correct, tick the checkbox to proceed to column mapping.

![Column mapping step](assets/screenshots/column_mapping.png)

### Map detection columns to the PAMalytics schema

Next, PAMalytics needs to identify required fields such as:

- start and end time
- presence/label fields
- probability/confidence fields (where available)

PAMalytics will suggest best-match mappings based on semantic similarity. You can accept the suggestions or override them where needed.

When you confirm the mapping, you’ll see an overview to review before finalising. Any extra metadata columns in your detection table are retained for downstream use.

When you’re happy, select **Launch PAMalytics dashboard** to enter the platform.
