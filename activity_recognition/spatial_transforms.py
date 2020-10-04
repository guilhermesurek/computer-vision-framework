from torchvision.transforms import transforms

# spatial transformations
class Compose(transforms.Compose):

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):

    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):

    def randomize_parameters(self):
        pass


class ScaleValue(object):

    def __init__(self, s):
        self.s = s

    def __call__(self, tensor):
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        pass


class Resize(transforms.Resize):

    def randomize_parameters(self):
        pass


class Scale(transforms.Scale):

    def randomize_parameters(self):
        pass


class CenterCrop(transforms.CenterCrop):

    def randomize_parameters(self):
        pass

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)