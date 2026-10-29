import unittest
from types import SimpleNamespace
import numpy as np
import torch
from .deva_memory import input_tensor,checkpoint_only_initialization,DEVAPair
from .geometry import InstanceFrame


class AdapterTests(unittest.TestCase):
    def test_rgb_normalization_and_aspect_ratio(self):
        image=np.zeros((9,16,3),np.uint8);image[:,:,2]=255
        x=input_tensor(image,'cpu',short_side=6)
        self.assertEqual(tuple(x.shape),(3,6,10))
        expected=torch.tensor([(1-.485)/.229,-.456/.224,-.406/.225])
        torch.testing.assert_close(x,expected[:,None,None].expand_as(x))

    def test_skip_initializer_downloads_restores_original_even_on_error(self):
        def f(*args,**kw):return kw
        module=SimpleNamespace(resnet18=f,resnet50=f)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with checkpoint_only_initialization(module):
                self.assertFalse(module.resnet50(pretrained=True)['pretrained'])
                self.assertEqual(module.resnet18(pretrained=True,extra_dim=1)['extra_dim'],1)
                raise RuntimeError('test')
        self.assertIs(module.resnet18,f);self.assertIs(module.resnet50,f)

    def test_repeated_or_old_timestamp_rejected_before_gpu_access(self):
        obj=DEVAPair.__new__(DEVAPair);obj.stamp=10
        for stamp in (9,10):
            f=InstanceFrame(stamp,'a'*32,np.zeros((2,3),np.uint16),np.empty(0,np.int64),
                            np.empty(0,np.uint16),np.empty(0,np.float32))
            with self.assertRaisesRegex(ValueError,'Non-increasing'):
                obj.step(np.zeros((2,3,3),np.uint8),f)


if __name__=='__main__':unittest.main()
