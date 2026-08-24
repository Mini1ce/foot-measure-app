# Google Colab training

Open `Foot_Segmentation_Training.ipynb` in Google Colab, select a GPU runtime,
and use **Run all**. The notebook is self-contained: it embeds the exact local
training and dataset-import code, downloads the pinned FootNet release from
Zenodo, verifies its SHA-256, and saves checkpoints under
`MyDrive/foot-measure-training`.

The final cell downloads `foot-segmentation-result.zip`. Keep the full Drive
folder as well because it contains the Keras checkpoints required for future
fine-tuning.

To rebuild the notebook after changing the pipeline:

```powershell
python ml/colab/build_notebook.py
```

Do not hand-edit the base64 source archive inside the generated notebook.
