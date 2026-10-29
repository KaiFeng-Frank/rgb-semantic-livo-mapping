import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .probe_sam2_compile import compare


class AdmissionTests(unittest.TestCase):
    def test_missing_warm_data_and_omitted_layout_cost_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'eager';b=root/'candidate';a.mkdir();b.mkdir()
            mask=np.packbits(np.ones((1,4),np.uint8),axis=1)
            for i in range(32):
                for path in (a,b):
                    np.savez(path/f'{i:03d}.npz',current=mask,temporal=mask,shape=[1,2,2])
            (a/'report.json').write_text('{}')
            compare(a,b,root/'missing.json')
            r=json.loads((root/'missing.json').read_text())
            self.assertTrue(r['summary']['equivalent'])
            self.assertFalse(r['eligible_for_further_validation'])
            rows=[dict(frame=i,wall_both_arms_ms=80.,current_ms=10.) for i in range(32)]
            passes=[dict(repeat=i,frames=rows,sam_only_ms=dict(p95=20.)) for i in range(4)]
            (b/'report.json').write_text(json.dumps(dict(passes=passes)))
            compare(a,b,root/'underbilled.json')
            self.assertFalse(json.loads((root/'underbilled.json').read_text())['warm_runtime_passed'])
            for p in passes:
                for r in p['frames']:r['wall_both_arms_ms']=40.
            (b/'report.json').write_text(json.dumps(dict(passes=passes)))
            compare(a,b,root/'valid.json')
            self.assertTrue(json.loads((root/'valid.json').read_text())['eligible_for_further_validation'])


if __name__=='__main__':unittest.main()
