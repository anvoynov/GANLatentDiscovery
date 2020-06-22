import types
from functools import wraps


def add_forward_with_shift(generator):
    def gen_shifted(self, z, shift, *args, **kwargs):
        return self.forward(z + shift, *args, **kwargs)

    generator.gen_shifted = types.MethodType(gen_shifted, generator)
    generator.dim_shift = generator.dim_z


def gan_with_shift(gan_factory):
    @wraps(gan_factory)
    def wrapper(*args, **kwargs):
        gan = gan_factory(*args, **kwargs)
        add_forward_with_shift(gan)
        return gan

    return wrapper
