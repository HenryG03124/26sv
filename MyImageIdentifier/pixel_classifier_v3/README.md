# Pixel Classifier V3 (A2D2)

V3 uses A2D2 for both road/vehicle segmentation and lane paint segmentation.
The network starts with the V2 architecture but lives independently in `Models/pixel_classifier_v3.py`.
Neither training nor preparation modifies V2 weights or the autopilot entry point.

Run from the project root:

```powershell
python -m MyImageIdentifier.pixel_classifier_v3.prepare_a2d2_pc_ds --watch
# After the full download and preparation have finished:
python -m MyImageIdentifier.pixel_classifier_v3.train
# Like V2, model, figure and test image paths are relative to the working directory.
# Put test.jpg in the working directory after training:
python -m MyImageIdentifier.pixel_classifier_v3.test
python -m unittest MyImageIdentifier.pixel_classifier_v3.test_a2d2_pc_ds -v
```

Data remains in `datasets/pc_lane_A2D2_ds` and `datasets/processed_pc_lane_A2D2_ds`.
Original files are hard-linked from `C:\Downloads\A2D2` without duplicating the image storage.
Existing lane masks are reused; V3 adds `train/road_masks` and `val/road_masks`.
CSV columns: `stem,image_path,mask_path,road_mask_path`.
The loader returns `(image, mask)` for `task="road"` or `task="lane"` at 640x352, using nearest-neighbor resizing for masks.
The active download is limited to 5,000 pairs: 4,000 training and 1,000 validation pairs with disjoint scenes.
Existing downloads are retained; the remaining frames are sampled across all 23 scenes with seed 42.
Final CSVs are generated when all 5,000 selected pairs have been downloaded and prepared.

## Class mapping

- Lane: 0=background, 1=solid/dashed paint, 255=ignore. Gaps are not filled.
- Road: 0=background, 1=road, 2=vehicle, 255=ignore.
- Road includes normal street, drivable cobblestone, slow-drive areas, speed bumps, lane paint, painted instructions and zebra crossings.
- Vehicle merges Car 1–4, Truck 1–3, Utility vehicle 1–2 and Tractor. It excludes ego car, pedestrians, bicycles and the separate Small vehicles classes.
- Parking, restricted/non-drivable areas, curbs and sidewalks are not road targets.
- Blur, rain/dirt and void are ignored by both losses.

Train and test keep V2's code structure, task-wise losses, alternating batches, evaluation and plotting.
Only the model import, A2D2 dataset access, checkpoint name and figure titles differ from V2.
V3 saves `pc_model_v3.pth` and `figure.png` in the working directory, like V2.
V2 retains BDD road/car data and BDD lane data with its existing alternating training routine.
