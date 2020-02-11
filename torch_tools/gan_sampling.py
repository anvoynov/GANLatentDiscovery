import os
import torch

from .visualization import to_image
from .utils import wrap_with_tqdm, make_verbose


def sample_from_gan(
        generator, out_dir, num_samples, out_shape,
        batch_size=50,
        noise_shape=None, rand_sampler=None, verbosity=make_verbose()):
    if noise_shape is None and rand_sampler is None:
        raise Exception('Either noise shape or randomizer should be provided')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    i = 0
    with torch.no_grad():
        generator.cuda()
        for _ in wrap_with_tqdm(range(num_samples // batch_size + 1), verbosity):
                if rand_sampler is not None:
                    noise = rand_sampler()
                else:
                    noise = torch.randn([batch_size] + noise_shape).cuda()
                generated = generator(noise).cpu().view([batch_size] + out_shape)
                for sample in generated:
                    to_image(sample).save(os.path.join(out_dir, '{}.png'.format(i)))
                    i += 1
                    if i > num_samples:
                        return


class GeneratorDataloader(object):
    def __init__(self, generator, batch_size, length, noise_shape=None, rand_sampler=None):
        self.generator = generator
        self.batch_size = batch_size
        self.len = length

        if noise_shape is None and rand_sampler is None:
            raise Exception('Ether noise shape or randomizer should be provided')
        if rand_sampler is None:
            def _rs():
                return torch.randn([batch_size] + noise_shape).cuda()
            self.rand_sampler = _rs
        else:
            self.rand_sampler = rand_sampler

    def __len__(self):
        return self.len

    def __iter__(self):
        def iterator():
            for _ in range(self.len):
                with torch.no_grad():
                    yield self.generator(self.rand_sampler()).detach()
            raise StopIteration

        return iterator()
