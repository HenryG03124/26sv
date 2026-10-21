import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from MyAutoPilot import auto_config


controls_text = '''SiiNunit
{
input_config : _nameless.test {
 version: 2
 config_lines: 15
 config_lines[4]: "device keyboard `di8.keyboard`"
 config_lines[18]: "device joy `di8.'{12345678-1234-1234-1234-123456789ABC}'`"
 config_lines[91]: "constant c_relatsteer 1.000000"
 config_lines[95]: "constant c_mousesteer 1.000000"
 config_lines[27]: "constant c_steer_dz 0.150000"
 config_lines[43]: "constant c_asteersens 1.250000"
 config_lines[69]: "constant c_rsteersens 0.250000"
 config_lines[12]: "constant c_steer_func 0.730000"
 config_lines[101]: "mix dsteerleft `keyboard.larrow?0 | keyboard.a?0 | joy.b1?0`"
 config_lines[204]: "mix dsteerright `keyboard.rarrow?0 | keyboard.d?0 | joy.b2?0`"
 config_lines[87]: "mix ui_left `keyboard.larrow?0`"
 config_lines[76]: "mix photo_left `keyboard.larrow?0`"
 config_lines[54]: "mix replay_right `keyboard.rarrow?0`"
 config_lines[38]: "mix steering `joy.x`"
 config_lines[29]: "mix throttle `joy.z`"
}
}
'''


class AutoConfigTests(unittest.TestCase):
    def test_changes_requested_settings_without_changing_other_lines(self):
        updated = auto_config.update_controls(controls_text)
        expected_values = {"c_relatsteer": 0 , "c_mousesteer": 0 , "c_steer_dz": 0 , "c_asteersens": 3 , "c_rsteersens": 0.5}
        for name , value in expected_values.items():
            matches = re.findall(r'constant ' + name + r' ([^"]+)"' , updated)
            self.assertEqual(len(matches) , 1)
            self.assertEqual(float(matches[0]) , value)
        self.assertIn('"mix dsteerleft `unbound?0 | keyboard.a?0 | joy.b1?0`"' , updated)
        self.assertIn('"mix dsteerright `unbound?0 | keyboard.d?0 | joy.b2?0`"' , updated)
        for original_line , updated_line in zip(controls_text.splitlines() , updated.splitlines()):
            if not any(name in original_line for name in (*expected_values , "mix dsteerleft " , "mix dsteerright ")):
                self.assertEqual(original_line , updated_line)
        self.assertEqual(len(controls_text.splitlines()) , len(updated.splitlines()))

    def test_arrow_replacement_is_limited_to_exact_driving_tokens(self):
        source = controls_text.replace(
            "keyboard.larrow?0 | keyboard.a?0 | joy.b1?0" ,
            "keyboard.larrow | keyboard.rarrow?0 | keyboard.larrowextra?0 | joy.larrow?0 | keyboard.a?0"
        ).replace(
            "keyboard.rarrow?0 | keyboard.d?0 | joy.b2?0" ,
            "keyboard.rarrow | keyboard.larrow?0 | keyboard.rarrowextra?0 | keyboard.d?0"
        )
        updated = auto_config.update_controls(source)
        self.assertIn(
            '"mix dsteerleft `unbound?0 | unbound?0 | keyboard.larrowextra?0 | joy.larrow?0 | keyboard.a?0`"' ,
            updated
        )
        self.assertIn(
            '"mix dsteerright `unbound?0 | unbound?0 | keyboard.rarrowextra?0 | keyboard.d?0`"' ,
            updated
        )
        self.assertIn('"mix ui_left `keyboard.larrow?0`"' , updated)
        self.assertIn('"mix photo_left `keyboard.larrow?0`"' , updated)
        self.assertIn('"mix replay_right `keyboard.rarrow?0`"' , updated)

    def test_preserves_bom_and_crlf(self):
        source = "\ufeff" + controls_text.replace("\n" , "\r\n")
        updated = auto_config.update_controls(source)
        self.assertTrue(updated.startswith("\ufeffSiiNunit\r\n"))
        self.assertEqual(updated.count("\r\n") , source.count("\r\n"))
        self.assertNotIn("\n" , updated.replace("\r\n" , ""))
        self.assertEqual(auto_config.update_controls(updated) , updated)

    def test_sensitivity_uses_module_settings(self):
        with patch.object(auto_config , "steering_sensitivity" , 2.25):
            with patch.object(auto_config , "relative_steering_sensitivity" , 0.75):
                updated = auto_config.update_controls(controls_text)
        self.assertEqual(float(re.search(r'constant c_asteersens ([^"]+)' , updated).group(1)) , 2.25)
        self.assertEqual(float(re.search(r'constant c_rsteersens ([^"]+)' , updated).group(1)) , 0.75)

    def test_missing_or_duplicate_required_entries_are_rejected(self):
        names = ("constant c_relatsteer " , "constant c_mousesteer " , "constant c_steer_dz " , "constant c_asteersens " ,
                 "constant c_rsteersens " , "mix dsteerleft " , "mix dsteerright ")
        for name in names:
            line = next(line for line in controls_text.splitlines(keepends = True) if name in line)
            for source in (controls_text.replace(line , "") , controls_text.replace(line , line + line)):
                with self.subTest(name = name , duplicates = source.count(name)):
                    with self.assertRaises(ValueError):
                        auto_config.update_controls(source)

    def test_invalid_config_does_not_write_or_check_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.sii"
            original = controls_text.replace("constant c_steer_dz " , "constant c_steer_dz_removed ").encode("utf-8")
            path.write_bytes(original)
            with patch.object(auto_config.subprocess , "run") as run:
                with self.assertRaises(ValueError):
                    auto_config.auto_config(path)
                run.assert_not_called()
            self.assertEqual(path.read_bytes() , original)
            self.assertFalse(path.with_name("controls.sii.bak").exists())

    def test_running_game_does_not_write_or_create_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.sii"
            original = controls_text.encode("utf-8")
            path.write_bytes(original)
            result = SimpleNamespace(stdout = b'"eurotrucks2.exe","1234","Console","1","123 K"\r\n' , returncode = 0)
            with patch.object(auto_config.subprocess , "run" , return_value = result) as run:
                with self.assertRaises(RuntimeError):
                    auto_config.auto_config(path)
                run.assert_called_once()
            self.assertEqual(path.read_bytes() , original)
            self.assertFalse(path.with_name("controls.sii.bak").exists())

    def test_writes_config_and_original_backup_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.sii"
            original = ("\ufeff" + controls_text.replace("\n" , "\r\n")).encode("utf-8")
            path.write_bytes(original)
            result = SimpleNamespace(stdout = b'INFO: No tasks are running which match the specified criteria.\r\n' , returncode = 0)
            with patch.object(auto_config.subprocess , "run" , return_value = result) as run:
                self.assertTrue(auto_config.auto_config(path))
                self.assertFalse(auto_config.auto_config(path))
                run.assert_called_once()
            self.assertEqual(path.with_name("controls.sii.bak").read_bytes() , original)
            expected = auto_config.update_controls(original.decode("utf-8")).encode("utf-8")
            self.assertEqual(path.read_bytes() , expected)

    def test_existing_backup_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.sii"
            backup_path = path.with_name("controls.sii.bak")
            path.write_bytes(controls_text.encode("utf-8"))
            backup_path.write_bytes(b"previous backup")
            result = SimpleNamespace(stdout = b"" , returncode = 0)
            with patch.object(auto_config.subprocess , "run" , return_value = result):
                self.assertTrue(auto_config.auto_config(path))
            self.assertEqual(backup_path.read_bytes() , b"previous backup")
            self.assertEqual(path.read_bytes() , auto_config.update_controls(controls_text).encode("utf-8"))

    def test_unchanged_config_does_not_check_processes_or_create_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.sii"
            original = auto_config.update_controls(controls_text).encode("utf-8")
            path.write_bytes(original)
            with patch.object(auto_config.subprocess , "run") as run:
                self.assertFalse(auto_config.auto_config(path))
                run.assert_not_called()
            self.assertEqual(path.read_bytes() , original)
            self.assertFalse(path.with_name("controls.sii.bak").exists())

if __name__ == "__main__":
    unittest.main()
