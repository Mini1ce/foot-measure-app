# Google Colab training

Open `Foot_Segmentation_Training.ipynb` in Google Colab, select a GPU runtime,
and use **Run all**. The notebook is self-contained: it embeds the exact local
training and dataset-import code, downloads the pinned FootNet release from
Zenodo, verifies its SHA-256, and saves checkpoints under
`MyDrive/foot-measure-training`.

The final cell downloads `foot-segmentation-result.zip`. Keep the full Drive
folder as well because it contains the Keras checkpoints required for future
fine-tuning.

## SynFoot V2 + FootNet notebook

`SynFoot_V2_Foot_Segmentation_Training.ipynb` is the three-stage T4 workflow:

1. Verify and extract the private SynFoot package created by
   `python -m ml.footdata package-pretrain`.
2. Pretrain using group-safe `pretrain_split` values and top-view masks.
3. Fine-tune on FootNet all views and then dorsal views before evaluation and
   TFLite export.

The notebook rejects non-canonical or unexpected ZIP members, checks every
extracted asset against the packaged manifest, and resumes a stage only when
its full input fingerprint still matches. The downloaded result includes
`training_provenance.json` so the package, source, arguments, checkpoints, and
evaluation artifacts can be audited later.

Upload the complete package directory to
`MyDrive/foot-measure-training/synfoot-v2-package` before using **Run all**.
Do not share the package publicly; retain and follow the terms supplied by the
SynFoot publisher. Synthetic validation is an early-stopping signal only, not
real-device or millimetre accuracy evidence.

## Resume, mask review, and optional fine-tuning

Use `SynFoot_V2_Resume_FineTune.ipynb` after the three-stage run exists in
`MyDrive/foot-measure-training/synfoot-v2-run`. It verifies the saved stage
fingerprints and checkpoint hashes without restarting SynFoot pretraining,
recreates the pinned FootNet development split, and writes a visual dorsal-mask
audit to `MyDrive/foot-measure-training/footnet-measurement-review`.

The notebook starts with `FINE_TUNE_EPOCHS = 0`. Review every report item,
download its CSV, and place it back as
`footnet-measurement-review/measurement-review.csv`. Fine-tuning is rejected
until every reviewed row has an explicit `include` or `exclude` decision. A
fine-tuned checkpoint is written to the new `footnet-dorsal-refined` directory;
the three original checkpoint directories are never overwritten.

To rebuild the notebook after changing the pipeline:

```powershell
python ml/colab/build_notebook.py
python -m ml.colab.build_synfoot_notebook
python -m ml.colab.build_synfoot_resume_notebook
```

Do not hand-edit the base64 source archive inside the generated notebook.
