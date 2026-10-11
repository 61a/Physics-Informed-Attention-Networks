import importlib.util
import unittest
import tempfile
import csv
from pathlib import Path
import numpy as np
from metrics import binary_metrics,validation_threshold
from physics import delays_in_samples,align_numpy
from data import read_manifest


class MetricsTests(unittest.TestCase):
    def test_exact_counts(self):
        result=binary_metrics([1]*90+[0]*150,[0.9]*83+[0.1]*7+[0.9]+[0.1]*149,0.5)
        self.assertEqual([result[k] for k in ('tp','fn','fp','tn')],[83,7,1,149])
        self.assertAlmostEqual(result['recall'],83/90)
        self.assertAlmostEqual(result['f1'],166/174)

    def test_threshold_uses_validation(self):
        labels=[0,1,1,0]; probabilities=[0.1,0.4,0.8,0.3]
        self.assertEqual(validation_threshold(labels,probabilities),0.4)
        with self.assertRaises(ValueError): validation_threshold([1,1],[0.1,0.5])

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError): binary_metrics([1],[float('nan')],0.5)


class PhysicsTests(unittest.TestCase):
    def test_units(self):
        delay=delays_in_samples([1.,1.4],[100.],1.)
        self.assertAlmostEqual(delay[0,0],4.149*100*(1-1/1.4**2))
        self.assertEqual(delay[0,1],0.)

    def test_positive_phase_advances_pulse(self):
        x=np.zeros((1,1,32,2)); x[0,0,15,0]=1.; x[0,0,10,1]=1.
        aligned=align_numpy(x,[[5.,0.]])[0,0,0]
        np.testing.assert_allclose(aligned[10], [1.,1.],atol=1e-10)
        self.assertEqual(int(np.argmax(aligned[:,0])),10)

    def test_no_edge_wrap(self):
        x=np.zeros((1,1,32,2)); x[0,0,1,0]=1.
        y=align_numpy(x,[[5.,0.]])
        np.testing.assert_allclose(y,0.,atol=1e-10)

    def test_frequency_order(self):
        a=delays_in_samples([1.,1.2,1.4],[10],1.)
        b=delays_in_samples([1.4,1.2,1.],[10],1.)
        np.testing.assert_allclose(a,b[:,::-1])

    def test_window_support(self):
        with self.assertRaises(ValueError): align_numpy(np.zeros((1,1,8,2)),[[8.,0.]])


class ManifestTests(unittest.TestCase):
    def test_group_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); rows=[]
            for i,(split,label) in enumerate((s,y) for s in ('train','val','test') for y in (0,1)):
                name=f'{i}.npy'; np.save(base/name,np.array([i]))
                rows.append(dict(path=name,label=label,split=split,group='shared' if i<3 else str(i)))
            with (base/'manifest.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['path','label','split','group']); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'group crosses'): read_manifest(base/'manifest.csv')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch runtime required')
class TorchTests(unittest.TestCase):
    def test_bank_crops_release_padded_storage(self):
        import torch
        from unittest.mock import patch
        from models import DispersionBank
        stack=torch.stack
        def checked_stack(windows,**kwargs):
            for window in windows:
                self.assertEqual(window.untyped_storage().nbytes(),window.numel()*window.element_size())
            return stack(windows,**kwargs)
        layer=DispersionBank([1.,1.4],[0.,1.],1.,16)
        with patch('models.torch.stack',side_effect=checked_stack):
            output=layer(torch.randn(1,1,16,2))
        self.assertEqual(tuple(output.shape),(1,2,1,16,2))

    def test_input_shape_is_explicit(self):
        from models import validate_input_shape
        for shape in (None,[1,0,1024],[True,60,1024],[1,60]):
            with self.assertRaisesRegex(ValueError,'input_shape'):
                validate_input_shape({'input_shape':shape})
        self.assertEqual(validate_input_shape({'input_shape':[1,256,1024]}),(1,256,1024))

    def test_distinct_baseline_encoders(self):
        import torch
        from models import Classifier
        cfg=dict(input_shape=[2,60,1024],embedding_dim=32,attention_heads=4,conformer_layers=1,convolution_kernel=7,dropout=0.)
        classes=set()
        for name in ('resnet18','efficientnet_b0','convnext_tiny','mobilenet_v3_large','vit_small'):
            model=Classifier(cfg,name,'raw').eval()
            classes.add(type(model.encoder))
            with torch.no_grad(): output=model(torch.zeros(1,2,60,1024))
            self.assertEqual(tuple(output.shape),(1,2))
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(classes),5)

    def test_fft_reference_and_gradient(self):
        import torch
        from models import DispersionBank
        x=torch.randn(1,2,32,2,requires_grad=True)
        layer=DispersionBank([1.,1.4],[0.,1.],1.,32)
        expected=align_numpy(x.detach().numpy(),delays_in_samples([1.,1.4],[0.,1.],1.))
        actual=layer(x)
        np.testing.assert_allclose(actual.detach().numpy(),expected,atol=2e-6)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        # Continuous shifts also retain gradients when deliberately enabled.
        layer.delays.requires_grad_(True)
        layer(x.detach()).square().mean().backward()
        self.assertTrue(torch.isfinite(layer.delays.grad).all())

    def test_conformer_pathways(self):
        import torch
        from models import Classifier
        cfg=dict(input_shape=[2,16,32],frequencies_ghz=np.linspace(1.,1.4,32).tolist(),dm_values=[0.,1.],sampling_ms=1.,embedding_dim=32,attention_heads=4,conformer_layers=1,convolution_kernel=7,dropout=0.)
        for front_end in ('raw','uniform','learned'):
            model=Classifier(cfg,'conformer',front_end).eval()
            with torch.no_grad(): output=model(torch.randn(2,2,16,32))
            self.assertEqual(tuple(output.shape),(2,2))
            self.assertTrue(torch.isfinite(output).all())


if __name__=='__main__': unittest.main()
