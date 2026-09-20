import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from MyImageIdentifier.pixel_classifier_v2.a2d2.prepare_pc_v2_a2d2_ds import (
    convert_mask , convert_road_mask , label_key , prepare_dataset , split_scenes
)


class PrepareA2D2Tests(unittest.TestCase):
    def test_datasets_share_train_and_test_structure(self):
        root = Path(__file__).resolve().parent.parent
        trees = []
        for dataset in ("bdd100k" , "a2d2"):
            source = (root / dataset / "train.py").read_text(encoding = "utf-8-sig")
            source = source.replace("v2_a2d2" , "v2_bdd100k")
            tree = ast.parse(source)
            trees.append([ast.dump(node) for node in tree.body
                          if isinstance(node , (ast.FunctionDef , ast.ClassDef , ast.For))])
        self.assertEqual(trees[0] , trees[1])
        original = (root / "bdd100k/test.py").read_text(encoding = "utf-8-sig")
        current = (root / "a2d2/test.py").read_text(encoding = "utf-8-sig")
        current = current.replace("v2_a2d2" , "v2_bdd100k")
        self.assertEqual(ast.dump(ast.parse(original)) , ast.dump(ast.parse(current)))

    def test_paint_classes_and_gaps(self):
        colors = np.array([[
            [128 , 0 , 255] , [255 , 0 , 255] , [128 , 0 , 255] ,
            [255 , 193 , 37] , [200 , 125 , 210] , [210 , 50 , 115] ,
            [96 , 69 , 143] , [53 , 46 , 82] , [0 , 0 , 0]
        ]] , dtype = np.uint8)
        mask = convert_mask(Image.fromarray(colors))
        self.assertEqual(mask.mode , "L")
        self.assertEqual(np.array(mask).tolist() , [[1 , 0 , 1 , 1 , 0 , 0 , 255 , 255 , 255]])

    def test_scene_split_is_deterministic(self):
        keys = [f"camera_lidar_semantic/scene_{scene}/camera/cam_front_center/frame_{frame}.png"
                for scene in range(10) for frame in range(3)]
        splits = split_scenes(keys)
        self.assertEqual(splits , split_scenes(list(reversed(keys))))
        self.assertEqual(sum(split == "val" for split in splits.values()) , 2)

    def test_road_includes_paint_and_merges_vehicles(self):
        colors = np.array([[
            [255 , 0 , 255] , [255 , 193 , 37] , [128 , 0 , 255] ,
            [200 , 125 , 210] , [210 , 50 , 115] , [255 , 0 , 0] ,
            [200 , 0 , 0] , [255 , 128 , 0] , [255 , 255 , 0] ,
            [0 , 0 , 100] , [72 , 209 , 204] , [180 , 150 , 200] , [96 , 69 , 143]
        ]] , dtype = np.uint8)
        self.assertEqual(np.array(convert_road_mask(Image.fromarray(colors))).tolist() ,
                         [[1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 0 , 0 , 255]])

    def make_source(self , root):
        entries = []
        for scene in range(5):
            key = f"camera_lidar_semantic/scene_{scene}/camera/cam_front_center/{scene}_camera_frontcenter_00001.png"
            for path_key in (key , label_key(key)):
                path = root / path_key
                path.parent.mkdir(parents = True , exist_ok = True)
                Image.new("RGB" , (32 , 16) , (128 , 0 , 255)).save(path)
                entries.append({"key": path_key})
        (root / "download_manifest.json").write_text(json.dumps({"files": entries}) , encoding = "utf-8")
        return entries

    def test_csv_layout_resume_and_existing_loader(self):
        from MyImageIdentifier.pixel_classifier_v2.a2d2.pc_v2_a2d2_ds import A2D2Dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "download"
            raw = root / "pc_lane_A2D2_ds"
            output = root / "processed_pc_lane_A2D2_ds"
            entries = self.make_source(source)
            rows = prepare_dataset(source , raw , output)
            self.assertEqual(len(rows["train"]) , 4)
            self.assertEqual(len(rows["val"]) , 1)
            for split in ("train" , "val"):
                with (output / (split + "_pairs.csv")).open(encoding = "utf-8-sig") as file:
                    records = list(csv.DictReader(file))
                for record in records:
                    self.assertTrue((output / record["image_path"]).is_file())
                    with Image.open(output / record["mask_path"]) as mask:
                        self.assertEqual(mask.size , (32 , 16))
                dataset = A2D2Dataset(output / (split + "_pairs.csv") , task = "lane")
                road_dataset = A2D2Dataset(output / (split + "_pairs.csv") , task = "road")
                image , mask = dataset[0]
                road_image , road_mask = road_dataset[0]
                self.assertEqual(tuple(image.shape) , (3 , 352 , 640))
                self.assertTrue(np.array_equal(image.numpy() , road_image.numpy()))
                self.assertEqual(tuple(road_mask.shape) , (352 , 640))
                self.assertEqual(road_mask.unique().tolist() , [1])
                self.assertEqual(tuple(mask.shape) , (352 , 640))
                self.assertEqual(mask.unique().tolist() , [1])
                Image.new("L" , (32 , 16) , 2).save(output / records[0]["road_mask_path"])
                self.assertEqual(road_dataset[0][1].unique().tolist() , [2])
                self.assertEqual(dataset[0][1].unique().tolist() , [1])
            self.assertTrue((source / entries[0]["key"]).samefile(raw / entries[0]["key"]))
            self.assertEqual(prepare_dataset(source , raw , output) , rows)

    def test_incomplete_download_does_not_publish_training_csvs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "download"
            entries = self.make_source(source)
            (source / entries[-1]["key"]).unlink()
            output = root / "processed"
            with self.assertRaisesRegex(RuntimeError , "incomplete"):
                prepare_dataset(source , root / "raw" , output)
            self.assertFalse((output / "train_pairs.csv").exists())
            self.assertFalse((output / "val_pairs.csv").exists())
            Image.new("RGB" , (32 , 16) , (128 , 0 , 255)).save(source / entries[-1]["key"])
            rows = prepare_dataset(source , root / "raw" , output)
            self.assertEqual(sum(map(len , rows.values())) , 5)


if __name__ == "__main__":
    unittest.main()
