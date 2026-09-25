"""Offline regression tests: no running game or private datasets required."""

import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock , patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader , TensorDataset

from Models.pc_v3 import UNet
from NetTrainer.pixel_classifier_v3.steering_dataset import (
    SteeringDataset , ensure_disjoint , split_steering_dataset ,
)
from NetTrainer.pixel_classifier_v3.tools.data_collector import Collector , SessionWriter


def telemetry(timestamp = 1 , **kwargs):
    return {"status": "ok" , "sdk_timestamp_us": timestamp , "age_s": 0 ,
            "user_steer": -0.4 , "game_steer": -0.2 , "speed_mps": 12.5 ,
            "throttle": 0.6 , "brake": 0.1 , **kwargs}


class CollectionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def session(self , name , count = 8):
        writer = SessionWriter(self.root , name , "png")
        try:
            for index in range(count):
                writer.write(Image.new("RGB" , (32 , 24) , (255 , 0 , 0)) , telemetry(index + 1) ,
                             1_700_000_000_000_000_000 + index * 1_000_000_000 , 2 , 1)
        finally:
            writer.close()
        return writer.directory / "samples.csv"

    def test_csv_rgb_and_both_label_choices(self):
        path = self.session("drive")
        dataset = SteeringDataset(path)
        image , target = dataset[0]
        self.assertEqual(tuple(image.shape) , (3 , 352 , 640))
        self.assertEqual(tuple(target.shape) , (1 , ))
        self.assertAlmostEqual(target.item() , -0.4)
        self.assertTrue(torch.all(image[0] == 1))
        self.assertTrue(torch.all(image[1:] == 0))
        self.assertAlmostEqual(SteeringDataset(path , "gameSteer")[0][1].item() , -0.2)
        with path.open(newline = "") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row["speed"] , "12.5")
        self.assertTrue((path.parent / row["image"]).is_file())
        metadata = json.loads((path.parent / "metadata.json").read_text())
        self.assertEqual(metadata["steering_positive"] , "left")

    def test_session_split_is_disjoint_and_reproducible(self):
        self.session("first")
        self.session("second")
        dataset = SteeringDataset(self.root)
        train , val = split_steering_dataset(dataset)
        ensure_disjoint(train , val)
        self.assertFalse({r.session for r in train.records} & {r.session for r in val.records})
        self.assertEqual(split_steering_dataset(dataset)[1].records , val.records)
        self.assertEqual(len(train) + len(val) , len(dataset))

    def test_single_drive_split_has_temporal_gap(self):
        train , val = split_steering_dataset(SteeringDataset(self.session("drive" , 20)) , gap_seconds = 2)
        self.assertGreater(min(r.timestamp for r in val.records) - max(r.timestamp for r in train.records) , 2e9)
        with self.assertRaisesRegex(ValueError , "share"):
            ensure_disjoint(train , train)

    def test_invalid_rows_and_missing_images_fail_before_training(self):
        path = self.session("drive")
        original = path.read_text()
        path.write_text(original.replace("-0.4" , "nan"))
        with self.assertRaisesRegex(ValueError , "Non-finite"):
            SteeringDataset(path)
        path.write_text(original)
        (path.parent / "images" / "00000000.png").unlink()
        with self.assertRaises(FileNotFoundError):
            SteeringDataset(path)

    def test_existing_session_is_never_overwritten(self):
        self.session("drive")
        with self.assertRaises(FileExistsError):
            SessionWriter(self.root , "drive")

    def test_collector_rejects_pause_duplicate_bad_values_and_lost_focus(self):
        writer = SessionWriter(self.root , "live" , "png")
        self.addCleanup(writer.close)
        reader = Mock()
        capture = Mock(return_value = Image.new("RGB" , (40 , 30) , (0 , 0 , 255)))
        collector = Collector(reader , capture , writer)
        reader.read.return_value = telemetry(1)
        self.assertEqual(collector.sample() , "warming up")
        self.assertEqual(collector.sample() , "duplicate SDK frame")
        reader.read.return_value = telemetry(2)
        self.assertEqual(collector.sample() , "recording")
        self.assertEqual(writer.count , 1)
        for bad in (telemetry(3 , status = "paused") , telemetry(3 , status = "stale") ,
                    telemetry(3 , user_steer = float("nan"))):
            reader.read.return_value = bad
            collector.sample()
            self.assertEqual(writer.count , 1)
        reader.read.return_value = telemetry(3)
        capture.side_effect = RuntimeError("Lost focus")
        self.assertEqual(collector.sample() , "Lost focus")
        self.assertEqual(writer.count , 1)
        writer.file.flush()
        self.assertTrue(torch.all(SteeringDataset(writer.directory)[0][0][2] == 1))

    def test_capture_delay_and_pause_during_capture_are_rejected(self):
        writer = Mock()
        reader = Mock()
        capture = Mock(return_value = Image.new("RGB" , (40 , 30)))
        collector = Collector(reader , capture , writer)
        collector.last_sdk_timestamp = 1
        reader.read.return_value = telemetry(2)
        with patch("NetTrainer.pixel_classifier_v3.tools.data_collector.time.monotonic_ns" ,
                   side_effect = [0 , 0 , 200_000_000 , 200_000_001]):
            self.assertEqual(collector.sample() , "capture/telemetry too slow")
        reader.read.side_effect = [telemetry(3) , telemetry(4 , status = "paused")]
        self.assertEqual(collector.sample() , "telemetry unavailable after capture")
        writer.write.assert_not_called()


class NetworkTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        #Read function definitions without starting the top-level training script.
        path = Path(__file__).resolve().parents[1] / "train.py"
        source = ast.parse(path.read_text(encoding = "utf-8"))
        definitions = []
        for node in source.body:
            if isinstance(node , (ast.ClassDef , ast.FunctionDef)):
                definitions.append(node)
        cls.training = {"np": np , "torch": torch , "nn": nn}
        exec(compile(ast.Module(body = definitions , type_ignores = []) , str(path) , "exec") , cls.training)

    def test_three_heads_and_rgb_steering_backward(self):
        torch.manual_seed(7)
        model = UNet()
        images = torch.rand(2 , 3 , 64 , 96)
        features = model(images)
        self.assertEqual(tuple(model.head(images , features , "road").shape) , (2 , 3 , 64 , 96))
        self.assertEqual(tuple(model.head(images , features , "lane").shape) , (2 , 2 , 64 , 96))
        outputs = model.head(images , features , "steering")
        self.assertEqual(tuple(outputs.shape) , (2 , 1))
        self.assertTrue(torch.all(outputs.abs() <= 1))
        nn.SmoothL1Loss()(outputs , torch.tensor([[-0.8] , [0.7]])).backward()
        self.assertGreater(model.root[0].weight.grad.abs().sum().item() , 0)
        self.assertGreater(model.steering_head[-2].weight.grad.abs().sum().item() , 0)
        with self.assertRaises(ValueError):
            model(images , task = "typo")

    def test_interleaved_training_and_regression_metrics(self):
        model = UNet()
        images = torch.rand(2 , 3 , 64 , 96)
        datasets = {
            "road": TensorDataset(images , torch.zeros(2 , 64 , 96 , dtype = torch.long)) ,
            "lane": TensorDataset(images , torch.ones(2 , 64 , 96 , dtype = torch.long)) ,
            "steering": TensorDataset(images , torch.tensor([[-0.5] , [0.5]])) ,
        }
        loaders = {task: DataLoader(data , batch_size = 2) for task , data in datasets.items()}
        criteria = {"road": nn.CrossEntropyLoss() , "lane": self.training["LaneLoss"](torch.ones(2) , 1.0 , 0.5) ,
                    "steering": nn.SmoothL1Loss(beta = 0.1)}
        old = model.steering_head[-2].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters() , lr = 1e-4)
        self.training.update({
            "model": model , "optimizer": optimizer , "device": "cpu" ,
            "lane_prob_threshold": 0.7 , "steering_criterion": criteria["steering"]
        })
        classes = {"road": 3 , "lane": 2 , "steering": 0}
        training = {}
        for task in loaders:
            if task == "steering":
                training[task] = self.training["train_steering"](loaders[task])
            else:
                training[task] = self.training["train_mask"](loaders[task] , task , criteria[task] , classes[task])
        for batch in range(len(loaders["road"])):
            for task in training:
                next(training[task])
        train = {}
        for task in training:
            train[task] = next(training[task])
        self.assertFalse(torch.equal(old , model.steering_head[-2].weight))
        val = {}
        for task in loaders:
            if task == "steering":
                val[task] = self.training["evaluate_steering"](loaders[task])
            else:
                val[task] = self.training["evaluate_mask"](loaders[task] , task , criteria[task] , classes[task])
        self.assertEqual(set(train) , set(loaders))
        with torch.no_grad():
            expected = (model(images , task = "steering") - datasets["steering"].tensors[1]).abs().mean().item()
        self.assertAlmostEqual(val["steering"][1] , expected)
        json.dumps({"train": train["steering"] , "val": val["steering"]} , allow_nan = False)

    def test_v2_weights_match_shared_network(self):
        from Models.pc_v2 import UNet as UNetV2

        model = UNet()
        state = UNetV2().state_dict()
        missing , unexpected = model.load_state_dict(state , strict = False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all(key.startswith("steering_head.") for key in missing))


if __name__ == "__main__":
    unittest.main()
