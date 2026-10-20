from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    'prepare_olmo_study', Path(__file__).resolve().parents[1] / 'scripts/prepare_olmo_study.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class NativeTemplateChecks(unittest.TestCase):
    def test_native_variants_must_match_both_text_and_tokens(self):
        for text, ids, passes in (('same', [1, 2], True),
                                  ('changed', [1, 2], False), ('same', [1, 3], False)):
            with self.subTest(text=text, ids=ids):
                tokenizer = Mock()
                tokenizer.apply_chat_template.side_effect = (
                    lambda *args, tokenize, ids=ids, text=text, **kwargs: ids if tokenize else text)
                if passes:
                    prepare.verify_native_rendering(tokenizer, [], 'same', [1, 2], ['native'])
                else:
                    with self.assertRaisesRegex(ValueError, 'cannot hold prompts fixed'):
                        prepare.verify_native_rendering(tokenizer, [], 'same', [1, 2], ['native'])

    def test_metadata_pins_the_actual_instruct_and_think_branch_names(self):
        fixture = {'sha':'a'*40,'siblings':[{
            'rfilename':'model.safetensors','lfs':{'sha256':'b'*64}}]}
        with patch.object(prepare, 'fetch', return_value=json.dumps(fixture).encode()) as fetch:
            prepare.metadata(50, 'instruct')
            self.assertIn('Olmo-3-7B-Instruct/revision/step_050?', fetch.call_args.args[0])
            prepare.metadata(25, 'think')
            self.assertIn('Olmo-3-7B-Think/revision/step_0025?', fetch.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
